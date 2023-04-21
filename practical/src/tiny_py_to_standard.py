from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType,
      ArrayAttr, i32, i64, f32, f64, IndexType, DictionaryAttr,
      Float16Type, Float32Type, Float64Type, FloatAttr, UnitAttr,
      DenseIntOrFPElementsAttr, VectorType, SymbolRefAttr)
from xdsl.dialects import func, arith, cf, memref, scf, llvm
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, BlockArgument, MLContext
import tiny_py
from xdsl.passes import ModulePass
from util.list_ops import flatten
from util.visitor import Visitor
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import copy

"""
This is a transformation pass which converts the tiny_py dialect to standard MLIR dialects. This
is required so that the MLIR opt tool can then generate LLVM-IR that we feed into LLVM
to generate and executable
"""

# A match between operation names and their standard dialect representations, there are
# two entries for each representing the integer and float corresponding operation
binary_arith_op_matching={"add": [arith.Addi, arith.Addf], "sub":[arith.Subi, arith.Subf],
                          "mult": [arith.Muli, arith.Mulf], "div": [arith.DivSI, arith.Divf]}

builtin_function_name_mapping={"print": "printf"}

global_declarations=[]

class GetAssignedVariables(Visitor):
  def __init__(self):
    self.assigned_vars=[]

  def traverse_assign(self, assign:tiny_py.Assign):
    var_name=assign.var_name.data
    if var_name not in self.assigned_vars:
      self.assigned_vars.append(var_name)

@dataclass
class SSAValueCtx:
    """
    Context that relates identifiers from the AST to SSA values used in the flat representation.
    """
    dictionary: Dict[str, SSAValue] = field(default_factory=dict)
    parent_scope: Optional[SSAValueCtx] = None

    def __getitem__(self, identifier: str) -> Optional[SSAValue]:
        """Check if the given identifier is in the current scope, or a parent scope"""
        ssa_value = self.dictionary.get(identifier, None)
        if ssa_value:
            return ssa_value
        elif self.parent_scope:
            return self.parent_scope[identifier]
        else:
            return None

    def __setitem__(self, identifier: str, ssa_value: SSAValue):
        """Relate the given identifier and SSA value in the current scope"""
        self.dictionary[identifier] = ssa_value

    def copy(self):
      ssa=SSAValueCtx()
      ssa.dictionary=dict(self.dictionary)
      return ssa

def translate_program(input_module: Module) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for top_level_entry in input_module.ops:
      for module in top_level_entry.children.blocks[0].ops:
        translate_toplevel(global_ctx, module, block)

    assert (len(block.ops)) == 1 and isinstance(block.ops[0], func.FuncOp)

    block.add_ops(global_declarations)
    body.add_block(block)
    return ModuleOp.from_region_or_ops(body)

def translate_toplevel(ctx: SSAValueCtx, op: Operation, block) -> Operation:
    if isinstance(op, tiny_py.Function):
        block.add_op(translate_fun_def(ctx, op))

def translate_fun_def(ctx: SSAValueCtx,
                      fn_def: tinypy.Function) -> Operation:
    """
    Translates a function definition into the func standard dialect
    """
    routine_name = fn_def.attributes["fn_name"]

    body = Region()
    block = Block()

    # Create a new nested scope and relate parameter identifiers with SSA values of block arguments
    c = SSAValueCtx(dictionary=dict(), parent_scope=ctx)

    arg_types=[]
    arg_names=[]

    body_contents=[]
    for op in fn_def.body.blocks[0].ops:
        res=translate_def_or_stmt(c, op)
        if res is not None:
            body_contents.append(res)

    block.add_ops(flatten(body_contents))

    # A return is always needed at the end of the procedure
    block.add_op(func.Return.create())

    body.add_block(block)

    # To keep it very simple we have an empty list of arguments to our function definition
    function_ir=func.FuncOp.from_region(routine_name.data, arg_types, [], body)
    function_ir.attributes["sym_visibility"]=StringAttr("public")

    return function_ir

def translate_def_or_stmt(ctx: SSAValueCtx,
                          op: Operation) -> List[Operation]:
    """
    Translate an operation that can either be a definition or statement
    """
    ops = try_translate_stmt(ctx, op)
    if ops is not None:
        return ops
    # operation must have been translated by now
    raise Exception(f"Could not translate `{op}' as a definition or statement")

def try_translate_stmt(ctx: SSAValueCtx,
                       op: Operation) -> Optional[List[Operation]]:
    """
    Tries to translate op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Returns None otherwise.
    """
    if isinstance(op, tiny_py.CallExpr):
        return translate_call_expr_stmt(ctx, op)
    if isinstance(op, tiny_py.Return):
        return translate_return(ctx, op)
    if isinstance(op, tiny_py.Assign):
        return translate_assign(ctx, op)
    if isinstance(op, tiny_py.Loop):
        return translate_loop(ctx, op)

    return None

def translate_stmt(ctx: SSAValueCtx,
                  op: Operation) -> List[Operation]:
    """
    Translates op as a statement.
    If op is an expression, returns a list of the translated Operations.
    Fails otherwise.
    """
    ops = try_translate_stmt(ctx, op)
    if ops is None:
        raise Exception(f"Could not translate `{op}' as a statement")
    else:
        return ops

def translate_return(ctx: SSAValueCtx,
                     return_stmt: tiny_py.Return) -> List[Operation]:
    """
    Translates the return operation, currently assume not returning any value
    """
    return [func.Return.get([])]

def translate_loop(ctx: SSAValueCtx,
                  loop_stmt: tiny_py.Loop) -> List[Operation]:
    """
    Translates a loop into the standard dialect scf while construct
    """

    assigned_var_finder=GetAssignedVariables()
    for op in loop_stmt.body.blocks[0].ops:
      assigned_var_finder.traverse(op)

    start_expr, start_ssa=translate_expr(ctx, loop_stmt.from_expr.blocks[0].ops[0])
    ctx[loop_stmt.variable]=start_ssa

    block_arg_types=[i32]
    block_args=[ctx[loop_stmt.variable]]
    for var_name in assigned_var_finder.assigned_vars:
        block_arg_types.append(ctx[StringAttr(var_name)].typ)
        block_args.append(ctx[StringAttr(var_name)])

    block = Block(arg_types=block_arg_types)

    end_expr, end_ssa=translate_expr(ctx, loop_stmt.to_expr.blocks[0].ops[0])
    ctx["end_expr"]=end_ssa

    compare=arith.Cmpi.get(end_ssa, block.args[0], 1)
    condition=scf.Condition.get(compare.results[0], start_ssa)

    block.add_ops(end_expr+[compare, condition])
    body=Region()
    body.add_block(block)

    block_after = Block(arg_types=block_arg_types)

    ops: List[Operation] = []
    prev_ctx=ctx.copy()
    c = SSAValueCtx(dictionary=dict(), parent_scope=ctx)
    for idx, var_name in enumerate(assigned_var_finder.assigned_vars):
      c[StringAttr(var_name)]=block_after.args[idx+1]

    for op in loop_stmt.body.blocks[0].ops:
        stmt_ops = translate_stmt(c, op)
        ops += stmt_ops

    loop_increment=arith.Constant.from_int_and_width(1, 32)
    loop_variable_inc=arith.Addi.get(ctx[loop_stmt.variable], loop_increment.results[0])

    yield_stmt=generate_yield(c, assigned_var_finder.assigned_vars, loop_variable_inc)
    block_after.add_ops(ops+[loop_increment, loop_variable_inc, yield_stmt])
    body_after=Region()
    body_after.add_block(block_after)

    block_updated=None

    return start_expr+[scf.While.get([block_args], [[i32]], body, body_after)]

def generate_yield(ctx: SSAValueCtx, assigned_vars,
                  loop_variable_inc:Operation) -> List[Operation]:
    """
      Generates a yield statement for exiting a block, this exposes
      the updated variables to their origional values.
      It is required because in SSA form when we update a
      variable it creates a new SSA element, and it is this
      element that then needs to be used as the program progresses.
      This is the way that the standard dialects
      handle this, other approaches such as Flang contains a store in
      it's FIR dialect which will explicitly allocate
      a variable and then store into it whenever it is updated.
    """
    yield_list=[loop_variable_inc.results[0]]
    for var_name in assigned_vars:
        yield_list.append(ctx[StringAttr(var_name)])

    return scf.Yield.get(*yield_list)

def translate_assign(ctx: SSAValueCtx,
                      assign: tiny_py.Assign) -> List[Operation]:
    """
    Translates assignment
    """
    var_name = assign.var_name
    assert isinstance(var_name, StringAttr)

    expr, ssa=translate_expr(ctx, assign.value.blocks[0].ops[0])

    # Always update the SSA context as it is this new SSA element that subsequent references
    # to the variable should reference
    ctx[var_name] = ssa
    return expr

def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: tiny_py.CallExpr, is_expr=False) -> List[Operation]:
    """
    Translates a call expression or statement, it's slightly different how we handle these depending
    upon whether its a statement or expression
    """
    ops: List[Operation] = []
    args: List[SSAValue] = []
    arg_types = []

    # Generate arguments that will be passed to the call
    for arg in call_expr.args.blocks[0].ops:
        op, arg = translate_expr(ctx, arg)
        if op is not None: ops += op
        args.append(arg)
        arg_types.append(arg.typ)

    name = call_expr.attributes["func"].data

    if call_expr.builtin.data and name in builtin_function_name_mapping:
        # If this is a built in function and that name is in the mapping dictionary
        # then replace the name, for instance translating print to printf
        name=builtin_function_name_mapping[name]

    if is_expr:
        result_type=try_translate_type(call_expr.type)
        call = func.Call.create(attributes={"callee": SymbolRefAttr(name)},
                                operands=args, result_types=[result_type])
    else:
        call = func.Call.create(attributes={"callee": SymbolRefAttr(name)},
                                operands=args, result_types=[])
    ops.append(call)

    if call_expr.builtin.data:
      global_declarations.append(func.FuncOp.external(name, arg_types, []))

    return ops

def translate_expr(ctx: SSAValueCtx,
                   op: Operation) -> Tuple[List[Operation], SSAValue]:
    """
    Translates op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Fails otherwise.
    """
    res = try_translate_expr(ctx, op)
    if res is None:
        raise Exception(f"Could not translate `{op}' as an expression")
    else:
        ops, ssa_value = res
        return ops, ssa_value

def try_translate_expr(ctx: SSAValueCtx,
                       op: Operation) -> Optional[Tuple[List[Operation], SSAValue]]:
    """
    Tries to translate op as an expression.
    If op is an expression, returns a list of the translated Operations
    and the ssa value representing the translated expression.
    Returns None otherwise.
    """
    if isinstance(op, tiny_py.Constant):
        op = translate_constant(op)
        return op
    if isinstance(op, tiny_py.BinaryOperation):
        op = translate_binary_expr(ctx, op)
        return op
    if isinstance(op, tiny_py.Var):
        # A variable is very simple, so just handle it here
        if ctx[op.variable] is None:
            raise Exception(f"Variable `{op.variable}' being referenced before it is declared")
        return [], ctx[op.variable]

    return None

def translate_constant(op: tiny_py.Constant) -> Operation:
    """
    Translates a constant, literal, depending upon its type
    """
    value = op.attributes["value"]

    if isinstance(value, StringAttr):
        value=StringAttr(value.data+"\n")
        global_type=llvm.LLVMArrayType.from_size_and_type(len(value.data), IntegerType(8))
        global_op=llvm.GlobalOp.get(global_type, "str0", "internal", 0, True, value=value, unnamed_addr=0)
        global_declarations.append(global_op)

        global_lookup=llvm.AddressOfOp.get("str0", llvm.LLVMPointerType.typed(global_type))
        element_pointer=llvm.GEPOp.get(global_lookup.results[0], llvm.LLVMPointerType.typed(IntegerType(8)), [0,0])
        return [global_lookup, element_pointer], element_pointer.results[0]

    if isinstance(value, FloatAttr):
        const= arith.Constant.create(attributes={"value": value},
                                         result_types=[value.type])
        return [const], const.results[0]

    if isinstance(value, IntegerAttr):
        const= arith.Constant.create(attributes={"value": value},
                                         result_types=[value.typ])
        return [const], const.results[0]

    raise Exception(f"Could not translate `{op}' as a literal")

def translate_binary_expr(ctx: SSAValueCtx,
                          op: Operation) -> Operation:
    """
    Translates a binary expression
    """
    lhs, lhs_ssa=translate_expr(ctx, op.lhs.blocks[0].ops[0])
    rhs, rhs_ssa=translate_expr(ctx, op.rhs.blocks[0].ops[0])
    operand_type = lhs_ssa.typ
    if op.op.data in binary_arith_op_matching:
        # We match here on the LHS type, for a more advanced coverage if the types are different
        # then we should convert the lower to the higher type, but we ignore that in our simple example
        if isinstance(operand_type, IntegerType): index=0
        if isinstance(operand_type, Float16Type) or isinstance(operand_type, Float32Type
                        ) or isinstance(operand_type, Float64Type): index=1
        op_instance=binary_arith_op_matching[op.op.data][index]
        assert op_instance is not None, "Operation "+op.op.data+" not implemented for type"
        bin_op=op_instance.get(lhs_ssa, rhs_ssa)
        return [bin_op], bin_op.results[0]
    else:
        raise Exception(f"Could not translate operation `{op.op.data}' as it is unknown")

@dataclass
class LowerTinyPyToStandard(ModulePass):

  name = 'tiny-py-to-standard'

  def apply(self, ctx: MLContext, input_module: ModuleOp):
      res_module = translate_program(input_module)
      res_module.regions[0].move_blocks(input_module.regions[0])
      # Create program entry point
      #check_program_entry_point(input_module)

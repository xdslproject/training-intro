from __future__ import annotations
from xdsl.dialects.builtin import (StringAttr, ModuleOp, IntegerAttr, IntegerType, ArrayAttr, i32, i64, f32, f64, IndexType, DictionaryAttr,
      Float16Type, Float32Type, Float64Type, FlatSymbolRefAttr, FloatAttr, UnitAttr, DenseIntOrFPElementsAttr, VectorType, FlatSymbolRefAttr)
from xdsl.dialects import func, arith, cf, memref
from xdsl.ir import Operation, Attribute, ParametrizedAttribute, Region, Block, SSAValue, MLContext, BlockArgument
import tiny_py
from util.list_ops import flatten
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

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
        if identifier in self.dictionary:
            raise Exception()
        else:
            self.dictionary[identifier] = ssa_value

def tiny_py_to_standard(ctx: MLContext, input_module: ModuleOp):
    res_module = translate_program(input_module)
    res_module.regions[0].move_blocks(input_module.regions[0])
    # Create program entry point
    #check_program_entry_point(input_module)

def translate_program(input_module: Module) -> ModuleOp:
    # create an empty global context
    global_ctx = SSAValueCtx()
    body = Region()
    block = Block()
    for top_level_entry in input_module.ops:
      for module in top_level_entry.children.blocks[0].ops:
        translate_toplevel(global_ctx, module, block)

    body.add_block(block)
    return ModuleOp.from_region_or_ops(body)

def translate_toplevel(ctx: SSAValueCtx, op: Operation, block) -> Operation:
  if isinstance(op, tiny_py.Function):
    block.add_op(translate_fun_def(ctx, op))

def translate_fun_def(ctx: SSAValueCtx,
                      fn_def: tinypy.Function) -> Operation:
    routine_name = fn_def.attributes["fn_name"]

    body = Region()
    block = Block()

    # Create a new nested scope and relate parameter identifiers with SSA values of block arguments
    # For now create this empty, will add in support for arguments later on!
    c = SSAValueCtx(dictionary=dict(), #zip(param_names, block.args)),
                    parent_scope=ctx)

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

    function_ir=func.FuncOp.from_region(routine_name, arg_types, [], body)
    function_ir.attributes["sym_visibility"]=StringAttr("public")

    #if len(arg_names) > 0:
    #  arg_attrs={}
    #  for arg_name in arg_names:
    #    arg_attrs[StringAttr("fir.bindc_name")]=StringAttr(arg_name)
    #  function_fir.attributes["arg_attrs"]=DictionaryAttr.from_dict(arg_attrs)
    return function_ir

def translate_def_or_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
    """
    Translate an operation that can either be a definition or statement
    """
    # first try to translate op as a definition:
    #   if op is a definition this will return a list of translated Operations
    #ops = try_translate_def(ctx, op, program_state)
    #if ops is not None:
    #    return ops
    # op has not been a definition, try to translate op as a statement:
    #   if op is a statement this will return a list of translated Operations
    ops = try_translate_stmt(ctx, op)
    if ops is not None:
        return ops
    # operation must have been translated by now
    return None
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

def translate_stmt(ctx: SSAValueCtx, op: Operation) -> List[Operation]:
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

def translate_return(ctx: SSAValueCtx, return_stmt: tiny_py.Return) -> List[Operation]:
  return [func.Return.get([])]

def translate_call_expr_stmt(ctx: SSAValueCtx,
                             call_expr: psy_ir.CallExpr, is_expr=False) -> List[Operation]:
    ops: List[Operation] = []
    args: List[SSAValue] = []

    for arg in call_expr.args.blocks[0].ops:
        op, arg = translate_expr(ctx, arg)
        if op is not None: ops += op
        #if not isinstance(arg.typ, fir.ReferenceType) and not isinstance(arg.typ, fir.ArrayType):
        #  reference_creation=fir.Alloca.build(attributes={"in_type":arg.typ, "valuebyref": UnitAttr()}, operands=[[],[]], regions=[[]], result_types=[fir.ReferenceType([arg.typ])])
        #  store_op=fir.Store.create(operands=[arg, reference_creation.results[0]])
        #  ops+=[reference_creation, store_op]
        #  args.append(reference_creation.results[0])
        #else:
        #  args.append(arg)
        args.append(arg)

    name = call_expr.attributes["func"]
    # Need return type here for expression
    if is_expr:
      result_type=try_translate_type(call_expr.type)
      call = func.Call.create(attributes={"callee": FlatSymbolRefAttr.from_string_attr(name)}, operands=args, result_types=[result_type])
    else:
      call = func.Call.create(attributes={"callee": FlatSymbolRefAttr.from_string_attr(name)}, operands=args, result_types=[])
    ops.append(call)
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

def try_translate_expr(
        ctx: SSAValueCtx,
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

    assert False, "Unknown Expression"

def translate_constant(op: tiny_py.Constant) -> Operation:
    value = op.attributes["value"]

    if isinstance(value, StringAttr):
        global_memref=memref.Global.get("str0", memref.MemRefType.from_type_and_list(i32, [len(value.data)]), value)
        memref_get=memref.GetGlobal.get("str0", memref.MemRefType.from_type_and_list(i32, [len(value.data)]))
        return [global_memref, memref_get], memref_get.results[0]
        return arith.Constant.create(attributes={"value": value},
                                         result_types=[i32])

    raise Exception(f"Could not translate `{op}' as a literal")

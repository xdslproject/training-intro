from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, ArrayOfConstraint, AnyAttr, IntAttr, FloatAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, ParameterDef,
                       irdl_attr_definition, irdl_op_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer

"""
This is our bespoke Python dialect that we are calling tiny_py. As you will see it is
rather limited but is sufficient for our needs, and being simple means that we can easily
navigate it and understand what is going on.
"""

@irdl_attr_definition
class BoolAttr(Data[bool]):
    """
    Represents a boolean, MLIR does not by default have a boolean (it uses integer 1 and 0)
    and-so this can be useful in your own dialects
    """
    name = "bool"
    data: bool

    @staticmethod
    def parse_parameter(parser: Parser) -> BoolAttr:
        data = parser.parse_str_literal()
        if data == "True": return True
        if data == "False": return False
        raise Exception(f"bool parsing resulted in {data}")
        return None

    @staticmethod
    def print_parameter(data: bool, printer: Printer) -> None:
        printer.print_string(f'"{data}"')

    @staticmethod
    @builder
    def from_bool(data: bool) -> BoolAttr:
        return BoolAttr(data)

@irdl_attr_definition
class EmptyAttr(ParametrizedAttribute):
    """
    This represents an empty value, can be useful where you
    need a placeholder to explicitly denote that something is not filled
    """
    name="empty"

@irdl_op_definition
class Module(Operation):
    """
    A Python module, this is the top level Python container which is a region
    """
    name = "tiny_py.module"

    children = SingleBlockRegionDef()

    @staticmethod
    def get(contents: List[Operation],
            verify_op: bool = True) -> FileContainer:
        res = Module.build(regions=[contents])
        if verify_op:
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Function(Operation):
    """
    A Python function, our handling here is simplistic and limited but sufficient
    for the exercise (and keeps this simple!) You can see how we have a mixture of
    attributes and a region for the body
    """
    name = "tiny_py.function"

    fn_name = AttributeDef(StringAttr)
    args = AttributeDef(ArrayAttr)
    return_var = AttributeDef(AnyAttr())
    body = SingleBlockRegionDef()

    @staticmethod
    def get(fn_name: Union[str, StringAttr],
            return_var: Union[Operation, None],
            args: List[Operation],
            body: List[Operation],
            verify_op: bool = True) -> Routine:
        if return_var is None:
            # If return is None then use the empty token placeholder
            return_var=EmptyAttr()
        res = Function.build(attributes={"fn_name": fn_name, "return_var": return_var,
                            "args": ArrayAttr.from_list(args)}, regions=[body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Assign(Operation):
    """
    Represents variable assignment, where the LHS is the variable and RHS an expression. Note
    that we are fairly limited here to representing one variable on the LHS only.
    We also make life simpler by just storing the variable name as a string, rather than a reference
    to the token which is also referenced directly by other parts of the code. The later is
    more flexible, but adds additional complexity in the code so we keep it simple here.
    """
    name = "tiny_py.assign"

    var_name = AttributeDef(StringAttr)
    value = SingleBlockRegionDef()

    @staticmethod
    def get(var_name: str,
            value: Operation,
            verify_op: bool = True) -> Assign:
        res = Assign.build(attributes={"var_name":var_name}, regions=[[value]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Loop(Operation):
    """
    A Python loop, we take a restricted view here that the loop will operate on a variable
    between two bounds (e.g. has been provided with a Python range).
    """
    name = "tiny_py.loop"

    variable = AttributeDef(StringAttr)
    from_expr = SingleBlockRegionDef()
    to_expr = SingleBlockRegionDef()
    body = SingleBlockRegionDef()

    @staticmethod
    def get(variable: str,
            from_expr: Operation,
            to_expr: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        res = Loop.build(attributes={"variable": variable}, regions=[[from_expr], [to_expr], body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Var(Operation):
    """
    A variable reference in Python, we just use the string name as storage here rather
    than pointing to a token instance of the variable which others would also reference
    directly.
    """
    name = "tiny_py.var"

    variable = AttributeDef(StringAttr)

    @staticmethod
    def get(variable,
            verify_op: bool = True) -> If:
        res = Var.build(attributes={"variable": variable})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class BinaryOperation(Operation):
    """
    A Python binary operation, storing the operation type as a string
    and the LHS and RHS expressions as regions
    """
    name = "tiny_py.binaryoperation"

    op = AttributeDef(StringAttr)
    lhs = SingleBlockRegionDef()
    rhs = SingleBlockRegionDef()

    @staticmethod
    def get(op: str,
            lhs: Operation,
            rhs: Operation,
            verify_op: bool = True) -> BinaryExpr:
        res = BinaryOperation.build(attributes={"op": op}, regions=[[lhs], [rhs]])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Constant(Operation):
    """
    A constant value, we currently support integers, floating points, and strings
    """
    name = "tiny_py.constant"

    value = AttributeDef(AnyOf([StringAttr, IntegerAttr, FloatAttr]))

    @staticmethod
    def get(value: Union[None, bool, int, str, float], width=None,
            verify_op: bool = True) -> Literal:
        if width is None: width=32
        if type(value) is int:
            attr = IntegerAttr.from_int_and_width(value, width)
        elif type(value) is float:
            attr = FloatAttr.from_float_and_width(value, width)
        elif type(value) is str:
            attr = StringAttr.from_str(value)
        else:
            raise Exception(f"Unknown constant of type {type(value)}")
        res = Constant.create(attributes={"value": attr})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@irdl_op_definition
class Return(Operation):
    """
    Return from a function, we just support return without
    any values/expressions at the moment
    """
    name = "tiny_py.return"

@irdl_op_definition
class CallExpr(Operation):
    """
    Calling a function, in our example calling the print function, we store the target
    function name and whether this is a builtin function as attributes (the second is
    using the Boolean Attribute that we define in this dialect). The type of the call is
    handled, as this is needed if the call is used as an expression rather than a statement,
    and lastly the arguments to pass which are enclosed in a region.
    """
    name = "tiny_py.call_expr"

    func = AttributeDef(StringAttr)
    builtin = AttributeDef(BoolAttr)
    type= AttributeDef(AnyOf([AnyAttr(), EmptyAttr]))
    args = SingleBlockRegionDef()

    @staticmethod
    def get(func: str,
            args: List[Operation],
            type=EmptyAttr(),
            builtin=False,
            verify_op: bool = True) -> CallExpr:
        # By default the type is empty attribute as the default is to call as a statement
        res = CallExpr.build(regions=[args], attributes={"func": func, "type": type, "builtin": builtin})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

@dataclass
class tinyPyIR:
    ctx: MLContext

    def __post_init__(self):
        """
        We need to register the attributes and operations defined in our dialect for parsing
        so that xDSL can load in the text representation of the dialect and properly
        structure it.
        """
        self.ctx.register_attr(BoolAttr)
        self.ctx.register_attr(EmptyAttr)

        self.ctx.register_op(Module)
        self.ctx.register_op(Function)
        self.ctx.register_op(Return)
        self.ctx.register_op(Constant)
        self.ctx.register_op(Assign)
        self.ctx.register_op(Loop)
        self.ctx.register_op(Var)
        self.ctx.register_op(BinaryOperation)
        self.ctx.register_op(CallExpr)

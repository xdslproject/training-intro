from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type, Union

from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, ArrayOfConstraint, AnyAttr, IntAttr, FloatAttr
from xdsl.ir import Data, MLContext, Operation, ParametrizedAttribute
from xdsl.irdl import (AnyOf, AttributeDef, SingleBlockRegionDef, builder, ParameterDef,
                       irdl_attr_definition, irdl_op_definition)
from xdsl.parser import Parser
from xdsl.printer import Printer

@irdl_attr_definition
class BoolAttr(Data[bool]):
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
    name="empty"

@irdl_op_definition
class Module(Operation):
    name = "tiny_py.module"
    
    children = SingleBlockRegionDef()

    @staticmethod
    def get(contents: List[Operation],
            verify_op: bool = True) -> FileContainer:
      res = Module.build(regions=[contents])
      if verify_op:
        res.verify(verify_nested_ops=False)
      return res

    def verify_(self) -> None:
      pass

@irdl_op_definition
class Function(Operation):
    name = "tiny_py.function"

    fn_name = AttributeDef(StringAttr)    
    args = AttributeDef(ArrayAttr)
    return_var = AttributeDef(AnyAttr())    
    body = SingleBlockRegionDef()    

    @staticmethod
    def get(fn_name: Union[str, StringAttr],
            return_var,            
            args: List[Operation],            
            body: List[Operation],            
            verify_op: bool = True) -> Routine:
        if return_var is None:
          return_var=EmptyToken()
        res = Function.build(attributes={"fn_name": fn_name, "return_var": return_var, "args": ArrayAttr.from_list(args)},
                            regions=[body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass 


@irdl_attr_definition
class Token(ParametrizedAttribute):
    name = "tiny_py.token"

    var_name : ParameterDef[StringAttr]
    type : ParameterDef[AnyAttr()]

@irdl_attr_definition
class EmptyToken(EmptyAttr):
    name = "tiny_py.emptytoken"

@irdl_op_definition
class Constant(Operation):
    name = "tiny_py.constant"

    value = AttributeDef(AnyOf([StringAttr, IntegerAttr, FloatAttr]))

    @staticmethod
    def get(value: Union[None, bool, int, str, float], width=None,
            verify_op: bool = True) -> Literal:
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
    name = "tiny_py.return"

@irdl_op_definition
class CallExpr(Operation):
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
        res = CallExpr.build(regions=[args], attributes={"func": func, "type": type, "builtin": builtin})
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res

    def verify_(self) -> None:
      pass

@dataclass
class tinyPyIR:
    ctx: MLContext

    def __post_init__(self):                
        self.ctx.register_attr(BoolAttr)
        self.ctx.register_attr(Token)
        self.ctx.register_attr(EmptyToken)        
        self.ctx.register_attr(EmptyAttr)      

        self.ctx.register_op(Module)
        self.ctx.register_op(Function)
        self.ctx.register_op(Return)
        self.ctx.register_op(Constant)
        self.ctx.register_op(CallExpr)

    @staticmethod
    def get_type(annotation: str) -> Operation:
        return TypeName.get(annotation)

    @staticmethod
    def get_statement_op_types() -> List[Type[Operation]]:
        statements: List[Type[Operation]] = []            
        return statements + psyIR.get_expression_op_types()

    @staticmethod
    def get_expression_op_types() -> List[Type[Operation]]:
        return [
            CallExpr
        ]

    @staticmethod
    def get_type_op_types() -> List[Type[Operation]]:
        return []

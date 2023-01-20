import ast, inspect
import tiny_py
from xdsl.printer import Printer
import sys

def python_compile(func):
    def compile_wrapper():
        a=ast.parse(inspect.getsource(func))
        analyzer = Analyzer()
        tiny_py_ir=analyzer.visit(a)

        printer = Printer(stream=sys.stdout)
        printer.print_op(tiny_py_ir)
    return compile_wrapper

class Analyzer(ast.NodeVisitor):
    def generic_visit(self, node):
        print(node)

    def visit_Module(self, node):
        contents=[]
        for a in node.body:
          contents.append(self.visit(a))
        return tiny_py.Module.get(contents)

    def visit_FunctionDef(self, node):
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        return tiny_py.Function.get(node.name, None, [], contents)

    def visit_Constant(self, node):
      return tiny_py.Constant.get(node.value)

    def visit_Call(self, node):
        arguments=[]
        for arg in node.args:
            arguments.append(self.visit(arg))
        builtin_fn=False
        if node.func.id == "print":
          builtin_fn=True
        return tiny_py.CallExpr.get(node.func.id, arguments, builtin=builtin_fn)

    def visit_Expr(self, node):
        return self.visit(node.value)

import ast, inspect
import tiny_py
from xdsl.printer import Printer
from xdsl.dialects.builtin import ModuleOp
import sys

"""
This is our very simple Python parser which will parse the decorated
function and generate an IR based on our tiny_py dialect. We use the Python
ast library to do the parsing which keeps this simple. Note that there are
other MLIR/xDSL Python parsers which are more complete, such as the xDSL
frontend and pyMLIR
"""

def python_compile(func):
    """
    This is our decorator which will undertake the parsing and output the
    xDSL format IR in our tiny_py dialect
    """
    def compile_wrapper():
        a=ast.parse(inspect.getsource(func))
        analyzer = Analyzer()
        tiny_py_ir=analyzer.visit(a)
        # This next line wraps our IR in the built in Module operation, this
        # is required to comply with the MLIR standard (the top level must be
        # a built in module).
        tiny_py_ir=ModuleOp.from_region_or_ops([tiny_py_ir])

        # Now we use the xDSL printer to output our built IR to stdio
        printer = Printer(stream=sys.stdout, target=Printer.Target.XDSL)
        printer.print_op(tiny_py_ir)
        print("") # Adds a new line

        f = open("output.xdsl", "w")
        printer_file = Printer(stream=f, target=Printer.Target.XDSL)
        printer_file.print_op(tiny_py_ir)
        f.write("") # Terminates file on new line
        f.close()
    return compile_wrapper

class Analyzer(ast.NodeVisitor):
    """
    Our very simple Python parser based on the ast library. It's very simplistic but
    provides an easy to understand view of how our IR is built up from the tiny_py
    dialect that we have created for these practicals
    """
    def generic_visit(self, node):
        """
        A catch all to print out the node if there is not an explicit handling function
        provided
        """
        print(node)
        raise Exception("Unknown Python construct, no parser provided")

    def visit_Assign(self, node):
        """
        Handle assignment, we visit the RHS and then create the tiny_py Assign IR operation
        """
        val=self.visit(node.value)
        return tiny_py.Assign.get(node.targets[0].id, val)

    def visit_Module(self, node):
        """
        Handles the top level Python module which contains many operations (here the
        function that was decorated).
        """
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        return tiny_py.Module.get(contents)

    def visit_FunctionDef(self, node):
        """
        A Python function definition, note that we keep this simple by hard coding that
        there is no return type and there are no arguments (it would be easy to extend
        this to handle these and is left as an exercise).
        """
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        return tiny_py.Function.get(node.name, None, [], contents)

    def visit_Constant(self, node):
        """
        A literal constant value
        """
        return tiny_py.Constant.get(node.value)

    def visit_Name(self, node):
        """
        Variable name
        """
        return tiny_py.Var.get(node.id)

    def visit_For(self, node):
        """
        Handles a for loop, note that we make life simpler here by assuming that
        it is in the format for i in range(from, to), and that is where we get
        the from and to expressions.
        """
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        expr_from=self.visit(node.iter.args[0])
        expr_to=self.visit(node.iter.args[1])
        return tiny_py.Loop.get(node.target.id, expr_from, expr_to, contents)

    def visit_BinOp(self, node):
        """
        A binary operation
        """
        op_str=self.getOperationStr(node.op)
        if op_str is None:
            raise Exception("Operation "+str(node.op)+" not recognised")
        lhs=self.visit(node.left)
        rhs=self.visit(node.right)
        return tiny_py.BinaryOperation.get(op_str, lhs, rhs)

    def visit_Call(self, node):
        """
        Calling a function, we provide a boolean describing whether this is a
        built in Python function (e.g. print) or a user defined function.
        """
        arguments=[]
        for arg in node.args:
            arguments.append(self.visit(arg))
        builtin_fn=self.isFnCallBuiltIn(node.func.id)
        return tiny_py.CallExpr.get(node.func.id, arguments, builtin=builtin_fn)

    def visit_Expr(self, node):
        """
        Visit a generic Python expression (it will then call other functions
        in our parser depending on the expression type).
        """
        return self.visit(node.value)

    def isFnCallBuiltIn(self, fn):
        """
        Deduces whether a function is built in or not
        """
        if fn == "print":
            return True
        elif fn == "range":
            return True

        return False

    def getOperationStr(self, op):
        """
        Maps Python operation to string name, as we use the string name
        in the tiny_py dialect
        """
        if isinstance(op, ast.Add):
            return "add"
        elif isinstance(op, ast.Sub):
            return "sub"
        elif isinstance(op, ast.Mult):
            return "mult"
        elif isinstance(op, ast.Div):
            return "div"
        else:
            return None

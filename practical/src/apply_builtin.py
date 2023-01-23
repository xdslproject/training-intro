from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block
from xdsl.dialects.builtin import IntegerAttr, StringAttr, ArrayAttr, IntAttr
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)
import tiny_py

class ApplyBuiltinRewriter(RewritePattern):
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, 
                          call_expr: tiny_py.CallExpr, rewriter: PatternRewriter):
        """
        Matches on the tiny_py CallExpr IR operation, this is pretty simplistic but sufficient
        for our needs, where we check whether the builtin flag on the operation is true and then
        branch on the function name (here we care about the print function). We iterate over
        the argument operations and for each of these if it is a string without a newline
        at the end then add one.
        """
        if (call_expr.builtin):
            # Check that this is a built in function
            if (call_expr.func.data=="print"):
                # This is the print function
                call_expr.attributes["func"]=StringAttr("printf")
                for op in call_expr.args.blocks[0].ops:
                    if isinstance(op, tiny_py.Constant) and isinstance(op.value, StringAttr):
                        # Check that the string terminates with a new line, if not then add one
                        if op.value.data[-1] != "\\n":
                            op.attributes["value"]=StringAttr(op.value.data+"\\n")

def apply_builtin(ctx: tiny_py.MLContext, module: ModuleOp) -> ModuleOp:
    """
    This is the entry point of the pass, where we create the rewriter and then walk the IR
    to select specific operations of interest and manipulate them
    """
    applyRewriter=ApplyBuiltinRewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyRewriter]), apply_recursively=False)
    walker.rewrite_module(module)

    return module

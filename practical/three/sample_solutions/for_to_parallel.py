from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Operation, SSAValue, Region, Block, MLContext, BlockArgument
from xdsl.dialects import scf, arith
from dataclasses import dataclass
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (GreedyRewritePatternApplier,
                                   PatternRewriter, PatternRewriteWalker,
                                   RewritePattern, op_type_rewrite_pattern)

matched_operations={"arith.addf": arith.Addf, "arith.addi": arith.Addi}

class ApplyForToParallelRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self,
                          for_loop: scf.For, rewriter: PatternRewriter):
        """
        This will apply a rewrite to the for loop to convert it into a parallel for loop
        with reductions
        """
        # First we get the body of the for loop and detach it (as will attack to the
        # parallel loop when we create it)
        loop_body=for_loop.body.blocks[0]
        for_loop.body.detach_block(0)
        # Now get the arguments to the yield at the end of the for loop and the arguments
        # to the loop block too
        yielded_args=list(loop_body.ops.last.arguments)
        block_args=list(loop_body.args)

        ops_to_add=[]
        for op in loop_body.ops:
          # We go through each operation in the loop body and see if it is one that needs
          # a reduction operation applied to it
          if op.name in matched_operations.keys():
            # We need to find if it is the LHS or RHS that is based upon the argument to the block
            # if it is neither then ignore this as it is not going to be updated from one iteration
            # to the next so no need to wrap in a reduction
            if isinstance(op.lhs, BlockArgument):
              block_arg_op=op.lhs
              other_arg=op.rhs
            elif isinstance(op.rhs, BlockArgument):
              block_arg_op=op.rhs
              other_arg=op.lhs
            else:
              continue

            # Now detach op from the body and remove from those arguments yielded
            # and arguments to the top level block
            op.detach()
            yielded_args.remove(op.results[0])
            block_args.remove(block_arg_op)

            # Create a new block for this reduction operation which has the type of
            # operation LHS and RHS present
            block_arg_types=[block_arg_op.typ, other_arg.typ]
            block = Block(arg_types=block_arg_types)

            # Retrieve the dialect operation to instantiate
            op_instance=matched_operations[op.name]
            assert op_instance is not None

            # Instantiate the dialect operation and create a reduce return operation
            # that will return the result, then add these operations to the block
            new_op=op_instance.get(block.args[0], block.args[1])
            reduce_result=scf.ReduceReturnOp.get(new_op.results[0])
            block.add_ops([new_op, reduce_result])

            # Create the reduce operation and add to the top level block
            reduce_op=scf.ReduceOp.get(other_arg, block)
            ops_to_add.append(reduce_op)

        # Create a new top level block which will have far fewer arguments
        # as none of the reduction arguments are now present here
        new_block=Block(arg_types=[arg.typ for arg in block_args])
        new_block.add_ops(ops_to_add)

        for op in loop_body.ops:
            op.detach()
            new_block.add_op(op)

        # We have a yield at the end of the block which yields non reduction
        # arguments
        new_yield=scf.Yield.get(*yielded_args)
        new_block.erase_op(new_block.ops.last)
        new_block.add_op(new_yield)

        # Create our parallel operation and replace the for loop with this
        parallel_loop=scf.ParallelOp.get([for_loop.lb], [for_loop.ub], [for_loop.step], [new_block], for_loop.iter_args)
        rewriter.replace_matched_op(parallel_loop)


@dataclass
class ConvertForToParallel(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'for-to-parallel'

  def apply(self, ctx: MLContext, input_module: ModuleOp):
    applyRewriter=ApplyForToParallelRewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyRewriter]), apply_recursively=False)
    walker.rewrite_module(input_module)

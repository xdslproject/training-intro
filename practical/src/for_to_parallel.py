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
        loop_body=for_loop.body.blocks[0]
        for_loop.body.detach_block(0)
        yielded_args=list(loop_body.ops[-1].arguments)
        block_args=list(loop_body.args)

        ops_to_add=[]
        for op in loop_body.ops:
          if op.name in matched_operations.keys():
            op.detach()
            yielded_args.remove(op.results[0])

            if isinstance(op.lhs, BlockArgument):
              block_arg_op=op.lhs
              other_arg=op.rhs
            elif isinstance(op.rhs, BlockArgument):
              block_arg_op=op.rhs
              other_arg=op.lhs
            else:
              raise Exception("One of LHS or RHS must be a block argument")

            block_args.remove(block_arg_op)
            block_arg_types=[block_arg_op.typ, other_arg.typ]
            block = Block(arg_types=block_arg_types)
            op_instance=matched_operations[op.name]
            assert op_instance is not None

            new_op=op_instance.get(block.args[0], block.args[1])
            reduce_result=scf.ReduceReturnOp.get(new_op.results[0])
            block.add_ops([new_op, reduce_result])

            reduce_op=scf.ReduceOp.get(other_arg, block)
            ops_to_add.append(reduce_op)

        new_block=Block(arg_types=[arg.typ for arg in block_args])
        new_block.add_ops(ops_to_add)

        for op in loop_body.ops:
            op.detach()
            new_block.add_op(op)

        new_yield=scf.Yield.get(*yielded_args)
        new_block.erase_op(new_block.ops[-1])
        new_block.add_op(new_yield)

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

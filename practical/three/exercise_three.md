# Exercise Three

In this practical we are going to explore transformations in more detail to add OpenMP parallelism and/or vectorisation to our loop in an automated manner.

Learning objectives are:

* Exploring the role of transformations and how these can manipulate the IR
* To understand how transformations are developed
* Gain an understanding of the key ways in which the IR can be traversed and manipulated
* Awareness of the _parallel_ operation in the _scf_ dialect
* To further demonstrate reusability benefits of MLIR transformations

Sample solutions to this exercise are provided in [sample_solutions](sample_solutions) in-case you get stuck or just want to compare your efforts with ours.

>**Having problems?**  
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

## The starting point and the plan

We are starting with the same code in practical two as illustrated below, however now we are going to write a transformation that will convert the resulting _for_ operation in the _scf_ dialect into a _parallel_ operation of that same dialect. 

```python
@python_compile
def ex_three():
    val=0.0
    add_val=88.2
    for a in range(0, 100000):
      val=val+add_val
    print(val)

ex_three()
```

The _parallel_ operation represents a parallel for loop, and there are existing MLIR transformations run via _mlir-opt_ that will then parallelise this via OpenMP by lowering into the _omp_ dialect, apply vectorisation by lowering to the _vector_ dialect, or acclerate this via GPUs by lowering to the _gpu_ dialect. This is an illustration of the major reuse benefits of MLIR, where developers need not understand the underlying _omp_, _vector_, or _gpu_ dialects, but instead can convert a loop into this higher level _parallel_ operation and all other transformatiosn to exploit these facets are present and can be easily reused.

![Illustration of parallel lowering](https://github.com/xdslproject/training-intro/raw/main/practical/three/parallel_lowering.png)

## Driving the transformation

Of course, one way of leveraging the _parallel_ operation would be to edit our _tiny_py_to_standard_ transformation which lowers from tiny py down to the standard dialects, issuing _parallel_ instead of _for_. However, let's assume that we do not want to edit that and instead wish to apply an optimisation/transformation pass on the resulting IR that comes out of _tiny_py_to_standard_ in order to convert our sequential loop into a parallel one. 

If you take a look in [tinypy-opt](https://github.com/xdslproject/training-intro/blob/main/practical/src/tools/tinypy-opt) tool (which is in _src/tools_ from the _practical_ directory) you will see at line 17 the _register_all_passes_ function which is registering possible transformations that can be performed on the IR. The second of these, _ConvertForToParallel_ is the transformation that we will be working with in this exercise and have already started off for you.

This transformation can be found in (src/for_to_parallel.py)[https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py] and the transformation entry point is defined at the bottom of the file by the class _ConvertForToParallel_, where the _name_ field defines the name of the transformation as provided to _tinypy_opt_.


```python
@dataclass
class ConvertForToParallel(ModulePass):
  """
  This is the entry point for the transformation pass which will then apply the rewriter
  """
  name = 'for-to-parallel'

  def apply(self, ctx: MLContext, input_module: ModuleOp):
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([]), apply_recursively=False)
    walker.rewrite_module(input_module)
```

Here we are creating the _PatternRewriteWalker_ which walks the IR in the block and instruction order, and rewrite it in place if needed. As an argument we provide an instantiation of _GreedyRewritePatternApplier_ which applies a list of patterns in order until one pattern matches. Currently an empty list of patterns are provided (due to the empty list _[]_) and we need to provide our rewrite pattern here.

The rewrite pattern defined above, that we will be working with in a moment, is _ApplyForToParallelRewriter_, so instantiate this at the first line of the _apply_ method and then pass this as a member of the list argument to _GreedyRewritePatternApplier_.

>**Not sure or having problems?**
> Please feel free to ask if there is anything you are unsure about, or you can check the [sample solution](https://github.com/xdslproject/training-intro/blob/main/practical/three/sample_solutions/for_to_parallel.py)

## What is needed in the IR

We now want to replace the _for_ operation in the _scf_ dialect with a _parallel_ operation, based on what we generated for exercise two, you might assume that this would look something like the following:

```
%7 = "scf.parallel"(%4, %5, %6, %0) ({
^0(%8 : index, %9 : f32):
  %10 = "arith.addf"(%9, %1) : (f32, f32) -> f32
  "scf.yield"(%10) : (f32) -> ()
}) : (index, index, index, f32) -> f32
```

However, unfortunately it is not quite this easy! You can see from the IR above that we are updating the left hand side of the _addf_ operation on each loop iteration, effectively undertaking a sum reduction overall. Because of this loop carried dependency, this reduction must be wrapped in a _reduce_ operation which instructs MLIR to implement this as a reduction when it lowers to the _omp_, _vector_, or _gpu_ dialects.

Instead, the following is what we are after, where it can be seen that there is now only one argument to the top level block (which is the loop iteration count), and within this block sits the _reduction_ operation from the _scf_ dialect. This operation must contain one block with the left hand and right hand sides of the operation that is being reduced, in this case _addf_, and the result of this is returned out via the _reduce.return_ operation.

```
%7 = "scf.parallel"(%4, %5, %6, %0) ({
^0(%8 : index):
  "scf.reduce"(%1) ({
    ^1(%lhs : f32, %rhs : f32):
      %11 = "arith.addf"(%lhs, %rhs) : (f32, f32) -> f32
      "scf.reduce.return"(%11) : (f32) -> ()
    }) : (f32) -> ()
  "scf.yield"() : () -> ()
}) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
```

If we look at the first line, `"%7 =scf.parallel"(%4, %5, %6, %0)` in the above snippet, the first three arguments are the loop lower bounds, upper bounds, and step size respectively. All others, here _%0_ are values provided as arguments and MLIR will use each of these as the left hand side argument (_lhs_) of the _reduce_ operations. Therefore the number of value provided arguments, in this case 1, must match the number of reductions. This is similar for results of the _parallel_ operation, where the result of the _ith_ _reduce_ operation is mapped to the _ith_ overall result. Lastly, the argument provided in `"scf.reduce"(%1)` is set as the right hand side (_rhs_) of the subsequent operation.

## Developing the rewrite pass

We now need to develop the rewrite pass to convert the _for_ operation to a _parallel_ operation and extract out the values being updated each iteration and wrap these in a _reduce_ operation. We have started this for you in the _ApplyForToParallelRewriter_ class of the (src/for_to_parallel.py)[https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py] file.


```python
class ApplyForToParallelRewriter(RewritePattern):

    @op_type_rewrite_pattern
    def match_and_rewrite(self,
                          for_loop: scf.For, rewriter: PatternRewriter):

        # First we get the body of the for loop and detach it (as will attack to the
        # parallel loop when we create it)
        loop_body=for_loop.body.blocks[0]
        for_loop.body.detach_block(0)
        # Now get the arguments to the yield at the end of the for loop and the arguments
        # to the loop block too
        yielded_args=list(loop_body.ops[-1].arguments)
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
            block_arg_types=[] # Needs to be completed!
            block = Block(arg_types=block_arg_types)

            # Retrieve the dialect operation to instantiate
            op_instance=matched_operations[op.name]
            assert op_instance is not None

            # Instantiate the dialect operation and create a reduce return operation
            # that will return the result, then add these operations to the block
            new_op=op_instance.get(block.args[0], block.args[1])
            reduce_result=None # Needs to be completed!
            block.add_ops([new_op, reduce_result])

            # Create the reduce operation and add to the top level block
            reduce_op=None # Needs to be completed!
            #ops_to_add.append(reduce_op)

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
        new_block.erase_op(new_block.ops[-1])
        new_block.add_op(new_yield)

        # Create our parallel operation and replace the for loop with this
        parallel_loop=None # Needs to be completed! 
        #rewriter.replace_matched_op(parallel_loop)
```

If we look at line 55 of (src/for_to_parallel.py)[https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py], which is `block_arg_types=[] # Needs to be completed!`, we need to provide the two types of the left and right hand sides as arguments to the block. These are _block_arg_op.typ_ and _other_arg.typ_ respectively, and each should be a member of the list (with a comma separating them).

At line 65, which is `reduce_result=None # Needs to be completed!` we need to create the _reduce.return_ operation which will return the result of the calculation's operation. We can create this by calling the _get_ method on _scf.ReduceReturnOp_, with _new_op.results[0]_ as the argument (this provides the SSA result of the _new_op_ operation that we created at the line above. 

At line 69, `reduce_op=None # Needs to be completed!`, we need to create the overall _reduce_ operation. This is done by calling the _get_ method on _scf.ReduceOp_, and there are two arguments needed here. The first is the operand, _other_arg_, provided to this (_%1_ in our IR example of the previous section) and the second is the block, which is the _block_ variable in the code, that will comprise this operation.

Now we have done this we need to create the parallel loop operation itself, which is line 88, `parallel_loop=None # Needs to be completed!`. Again, we will be calling the _get_ method but this time on _scf.ParallelOp_. We can directly reuse the loop bounds and step from the for loop, _for_loop.lb_, _for_loop.ub_, and _for_loop.step_ as the first three arguments but crucially each of these needs to be wrapped in a list (so it will be [_for_loop.lb_]) - we will explain why that is the case a little later on. The _new_block_ variable is our block, that is the fourth argument and again must be wrapped in a list, and the fifth argument is the list of SSA argument values provided (in the IR example above this will be _%0_) and is _for_loop.iter_args_ which is already a list so need not be wrapped in one.

We are almost there, the last step is to instruct xDSL to replace the for loop with the new parallel loop. As the last line of this _match_and_rewrite_ method, just after you created the _parallel_ operation, you should add `rewriter.replace_matched_op(parallel_loop)`.


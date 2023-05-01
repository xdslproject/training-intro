# Exercise Three

In this practical we are going to explore transformations in more detail to add OpenMP parallelism and/or vectorisation to our loop in an automated manner.

Learning objectives are:

* Exploring the role of transformations and how these can manipulate the IR
* To understand how transformations are developed
* Gain an understanding of the key ways in which the IR can be traversed and manipulated
* Awareness of the _parallel_ operation in the _scf_ dialect
* To further demonstrate reusability benefits of MLIR transformations

Sample solutions to this exercise are provided in [sample_solutions](sample_solutions) in-case you get stuck or just want to compare your efforts with ours.

It is assumed that you have a command line terminal in the _training-intro/practical/three_ directory.

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

This transformation can be found in [src/for_to_parallel.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py) and the transformation entry point is defined at the bottom of the file by the class _ConvertForToParallel_, where the _name_ field defines the name of the transformation as provided to _tinypy_opt_.


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

We now need to develop the rewrite pass to convert the _for_ operation to a _parallel_ operation and extract out the values being updated each iteration and wrap these in a _reduce_ operation. We have started this for you in the _ApplyForToParallelRewriter_ class of the [src/for_to_parallel.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py) file.

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
        new_block.erase_op(new_block.ops.last)
        new_block.add_op(new_yield)

        # Create our parallel operation and replace the for loop with this
        parallel_loop=None # Needs to be completed!         
```

The method _match_and_rewrite_ defined as `def match_and_rewrite(self, for_loop: scf.For, rewriter: PatternRewriter)` will be called whenever the IR walker encounters a node which is of type _scf.For_. This is the argument _for_loop_ to the method, which we can then manipulate as required by the transformation

If we look at line 55 of [src/for_to_parallel.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/for_to_parallel.py), which is `block_arg_types=[] # Needs to be completed!`, we need to provide the two types of the left and right hand sides as arguments to the block. These are _block_arg_op.typ_ and _other_arg.typ_ respectively, and each should be a member of the list (with a comma separating them).

At line 65, which is `reduce_result=None # Needs to be completed!` we need to create the _reduce.return_ operation which will return the result of the calculation's operation. We can create this by calling the _get_ method on _scf.ReduceReturnOp_, with _new_op.results[0]_ as the argument (this provides the SSA result of the _new_op_ operation that we created at the line above. 

At line 69, `reduce_op=None # Needs to be completed!`, we need to create the overall _reduce_ operation. This is done by calling the _get_ method on _scf.ReduceOp_, and there are two arguments needed here. The first is the operand, _other_arg_, provided to this (_%1_ in our IR example of the previous section) and the second is the block, which is the _block_ variable in the code, that will comprise this operation.

Now we have done this we need to create the parallel loop operation itself, which is line 88, `parallel_loop=None # Needs to be completed!`. Again, we will be calling the _get_ method but this time on _scf.ParallelOp_. We can directly reuse the loop bounds and step from the for loop, _for_loop.lb_, _for_loop.ub_, and _for_loop.step_ as the first three arguments but crucially each of these needs to be wrapped in a list (so it will be [_for_loop.lb_]) - we will explain why that is the case a little later on. The _new_block_ variable is our block, that is the fourth argument and again must be wrapped in a list, and the fifth argument is the list of SSA argument values provided (in the IR example above this will be _%0_) and is _for_loop.iter_args_ which is already a list so need not be wrapped in one.

We are almost there, the last step is to instruct xDSL to replace the for loop with the new parallel loop. As the last line of this _match_and_rewrite_ method, just after you created the _parallel_ operation, you should add `rewriter.replace_matched_op(parallel_loop)`.

>**Not sure or having problems?**
> Please feel free to ask if there is anything you are unsure about, or you can check the [sample solution](https://github.com/xdslproject/training-intro/blob/main/practical/three/sample_solutions/for_to_parallel.py)

### Looking at the rewrite pass in more detail

A lot of the work being done here in the transformation is in extracting the arguments out of the top level block and removing them from the final _yield_. Effectively this transformation is manipulating the internal structure from the first IR to the second IR provided in the previous section, so have a look through the code and see if you can understand these it is making these modification, don't hesitate to ask one of the presenters about this if you are unsure about anything. For instance, _detach_ on an operation will remove it from its block (an operation can only be a member of one block), _erase_op_ will remove an operation from a block and _add_op_ adds an operation.

There are some things worth highlighting in the IR and rewrite pass that we have skipped over so far. Firstly, we passed the parallel loop's lower and upper bounds along with the step in a list. This is because a _parallel_ operation can represent a nested loop and the below IR example illustrates a parallel loop operating over a nested loop, with _%4_ being the lower bounds of the top loop and _%5_ the lower bounds of the inner loop. The upper bounds and step values follow a similar logic. You can see that the block now has two index argument, representing the current iteration of both the inner and outer loops. MLIR will transform this as it feels most appropriate, for instance with the _openmp_ lowering it will likely apply the _collapse_ clause.

```
%7 = "scf.parallel"(%4, %5, %6, %7, %8, %9) ({
  ^0(%10 : index, %11 : index):
  ...
  "scf.yield"() : () -> ()
}) {"operand_segment_sizes" = array<i32: 2, 2, 2, 0>} : (index, index, index, index, index, index) -> ()
```

You can see in the above IR that we have _operand_segment_sizes_ provided as an argument to the operation. This is required for _varadic_ operands, which are operands which can have any size. Here the attribute is informing the operation that it is two lower bound operands, two upper bound operands, and two step operands but no SSA value arguments to be passed in.

### Running our transformation pass

Now we have developed our pass, let's run it through `tinypy-opt` as per the following snippet. Note that here we are undertaking two transformations, first our previous _tiny-py-to-standard_ lowering and then the _for-to-parallel_ which because it comes second operates on the results of the first transformation.

```bash
user@login01:~$ tinypy-opt output.xdsl -p tiny-py-to-standard,for-to-parallel -f mlir -t mlir
```

The following is the IR outputted from these two transformations, you can see the _parallel_, _reduce_, and _reduce.return_ operations that we have added into our transformation in this section. The rest of the IR is the same as that generated in exercise two, and that is a major benefit of using _parallel_ because we can parallelise a loop without requiring extensive IR changes elsewhere.

```
"builtin.module"() ({
  "func.func"() ({
    %0 = "arith.constant"() {"value" = 0.0 : f32} : () -> f32
    %1 = "arith.constant"() {"value" = 88.2 : f32} : () -> f32
    %2 = "arith.constant"() {"value" = 0 : i32} : () -> i32
    %3 = "arith.constant"() {"value" = 100000 : i32} : () -> i32
    %4 = "arith.index_cast"(%2) : (i32) -> index
    %5 = "arith.index_cast"(%3) : (i32) -> index
    %6 = "arith.constant"() {"value" = 1 : index} : () -> index
    %7 = "scf.parallel"(%4, %5, %6, %0) ({
    ^0(%8 : index):
      "scf.reduce"(%1) ({
      ^1(%9 : f32, %10 : f32):
        %11 = "arith.addf"(%9, %10) : (f32, f32) -> f32
        "scf.reduce.return"(%11) : (f32) -> ()
      }) : (f32) -> ()
      "scf.yield"() : () -> ()
    }) {"operand_segment_sizes" = array<i32: 1, 1, 1, 1>} : (index, index, index, f32) -> f32
    %12 = "llvm.mlir.addressof"() {"global_name" = @str0} : () -> !llvm.ptr<!llvm.array<3 x i8>>
    %13 = "llvm.getelementptr"(%12) {"rawConstantIndices" = array<i32: 0, 0>} : (!llvm.ptr<!llvm.array<3 x i8>>) -> !llvm.ptr<i8>
    %14 = "arith.extf"(%7) : (f32) -> f64
    "func.call"(%13, %14) {"callee" = @printf} : (!llvm.ptr<i8>, f64) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<3 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "%f\n", "unnamed_addr" = 0 : i64} : () -> ()
  "func.func"() ({
  }) {"sym_name" = "printf", "function_type" = (!llvm.ptr<i8>, f64) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
```

## Compile and run

We are now ready to feed this into `mlir-opt` and generate LLVM IR to pass to Clang to build out executable. Similarly to exercise one you should create a file with the _.mlir_ ending, via 

```bash
user@login01:~$ tinypy-opt output.xdsl -p tiny-py-to-standard,for-to-parallel -f mlir -t mlir -o ex_three.mlir
```

### Threaded parallelism via OpenMP

Execute the following:

```bash
user@login01:~$ mlir-opt --pass-pipeline="builtin.module(loop-invariant-code-motion, convert-scf-to-openmp, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-arith-to-llvm{index-bitwidth=64}, convert-openmp-to-llvm, convert-func-to-llvm, reconcile-unrealized-casts)" ex-three.mlir | mlir-translate -mlir-to-llvmir | clang -fopenmp -x ir -o test -
```

This is similar to the _mlir-opt_ command that we issued in exercice two, but with a few additions. Firstly, _convert-scf-to-openmp_ will run the MLIR transformation to lower our parallel loop to the _omp_ dialect, and secondly _convert-openmp-to-llvm_ will then lower this to the _llvm_ dialect. Furthermore you can see that we have had to pass the _-fopenmp_ flag to clang as we must now link with the OpenMP runtime.

You can either run this on the login node (or local machine), or submit to the batch queue for execution on a compute node.

We can execute the _test_ executable direclty on the login node if we wish by (or if you are following the tutorial on your local machine):

```bash
user@login01:~$ export OMP_NUM_THREADS=8
user@login01:~$ ./test
```

A submission script called _sub_ex3.srun_ is prepared that you can submit to the batch queue and will run over all 128 cores of the node.

```bash
user@login01:~$ sbatch sub_ex3.srun
```

You can check on the status of your job in the queue via _squeue -u $USER_ and once this has completed an output file will appear in your directly that contains the stdio output of the job. You can cat or less this file, which ever you prefer.

In the submission file we have added the _time_ command which reports how long the executable took to run, and indeed if running this locally you can achieve this via `time ./test`. Experiment with running over different numbers of OpenMP threads, via the `OMP_NUM_THREADS` environment variable (which you will see is set to 128 in the _sub_ex3.srun_ and can be changed). How does this impact the runtime? You can also change the problem size (e.g. the number of loop iterations) by modifying the value in the origional _ex_three.py_ Python file and then regenerating and recompiling.

### Adding vectorisation

We can use the _scf-parallel-loop-specialization_ pass to apply vectorisation to our parallel loop, in order to do this (we do this instead of OpenMP, but the two can be mixed):

```bash
user@login01:~$ mlir-opt --pass-pipeline="builtin.module(loop-invariant-code-motion, scf-parallel-loop-specialization, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-arith-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts)" ex-three.mlir | mlir-translate -mlir-to-llvmir | clang -fopenmp -x ir -o test -
```

The executable is then run in the same manner as with OpenMP

### Running on a GPU

We don't have GPUs in ARHCER2, so their use is beyond the scope of this course, but if you have a GPU machine then you can transform your parallel loop into the _gpu_ dialect via the following

```bash
user@login01:~$ mlir-opt --pass-pipeline="builtin.module(scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)" ex-three.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

>**Note**
> Your LLVM must have been built with explicit support for GPUs via passing the `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"` flag to cmake

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

## Manipulating the IR

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


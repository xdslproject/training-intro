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

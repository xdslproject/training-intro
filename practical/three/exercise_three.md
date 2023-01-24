# Exercise Three

This practical is more open ended, and we are going to add in threaded parallelism via OpenMP around our loop. In doing so we will declare a new dialect and transformation pass, connecting these into _tinypy_opt_.

>**Having problems?**
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

## The starting point

We are starting with the same code in practical two as illustrated below, however now we are aiming for the loop over _a_ to be parallelised via OpenMP. To do this we are going to add our open OpenMP tiny python dialect and we will use a new transformation pass to wrap this around the for loop in the code. Lastly we will lower our new dialect down to the MLIR _omp_ standard dialect.

```python
@python_compile
def ex_three():
    val=0.0
    add_val=88.2
    for a in range(0, 100000):
      val=val+add_val

ex_three()
```

## Defining our new dialect

We are going to create a new dialect called openmp_python. If you look at the _tiny_py.py_ dialect (in _src/dialects_) then this gives you a good idea of what's needed. You will need to define a new operation, let's call it _OpenMPLoop_ and for this to contain a _body_ region (which will be the tiny_py loop that we are going to wrap). You should define this as a class which is decorated with the _irdl_op_definition_ decorator and inherits from _Operation_. It will have a single member called _body_ and of type _SingleBlockRegionDef_ and the name variable should be something like _openmp_python.loop_. You should then define the _get_ function (have a look in the _tiny_py_ dialect how it is done there).

Lastly, you should define the IR dialect itself, this will be something like below and is important so that the dialect can be parsed.

```python
@dataclass
class OpenMPPythonIR:
    ctx: MLContext

    def __post_init__(self):
        self.ctx.register_op(OpenMPLoop)
```

## Creating the transformation pass

Next we want to develop a pass that will identify all loops in the _tiny_py_ dialect and wrap them with our OpenMP loop. To do this we are going to write a new pass, and you can refer to the _apply_builtin.py_ pass in order to help. Let's call it _apply_openmp_, create this as a file and then based on the _apply_builtin.py_ pass define a _ApplyOpenMPRewriter_, where the IR node type is _tiny_py.Loop_. 

What we then need to do inside the function is to detach the loop, create our _OpenMPLoop_ and provide the origional loop to this as an argument, then adding our _OpenMPLoop_ in where the origional loop was before. Assuming you call the loop IR node argument _for_loop), the code to do this will look like the snippet below where we initially obtain the IR block parent of the for loop node, before retrieving its index in the block's list of operations (as we will insert our new OpenMP loop into the same location) and then detatch the for loop from the IR. We then create our new dialect's 'OpenMPLoop' operation, providing the _for_loop_ variable as an argument as that will constitute the operations within the _body_ region of our OpenMP loop. Lastly, we instruct the IR rewriter to insert our new _openmp_loop_ operation into the parent block at the same location that the origional loop was at.

```python
block = for_loop.parent
idx = block.ops.index(for_loop)
for_loop.detach()

openmp_loop = openmp_python.OpenMPLoop.get([for_loop])
rewriter.insert_op_at_pos(openmp_loop, block, idx)        
```

We now need to define the pass's entry point which drives the transformation, again you can look in the _apply_builtin.py_ file to get a feeling to how to do this, or base it on the code below.

```python
def apply_openmp(ctx: tiny_py.MLContext, module: ModuleOp) -> ModuleOp:
    applyRewriter=ApplyOpenMPRewriter()
    walker = PatternRewriteWalker(GreedyRewritePatternApplier([applyRewriter]), apply_recursively=False)
    walker.rewrite_module(module)

    return module
```

## Hooking it all up

Having created our new dialect and transformation pass, we now need to modify the _tinypy_opt_ tool so that it is aware of these. This script is held in the _src/tools_ directory, with the new dialect needing to be instantiated in the _register_all_dialects_ function and the transformation pass entry function appended to the _passes_native_ loop between lines 22 and 25.

Now you should run `tinypy_opt three.xdsl -p apply-openmp` (note that underscores in pass names become dashes in the command line arguments). This will run the pass over the xDSL IR file and the output will look like below, where you can see that the _openmp_python.loop_ operation now wraps _tiny_py.loop_.

```
builtin.module() {
  tiny_py.module() {
    tiny_py.function() ["fn_name" = "ex_two", "return_var" = !empty, "args" = []] {
      tiny_py.assign() ["var_name" = "val"] {
        tiny_py.constant() ["value" = 0.0 : !f32]
      }
      tiny_py.assign() ["var_name" = "add_val"] {
        tiny_py.constant() ["value" = 88.2 : !f32]
      }
      openmp_python.loop() {
        tiny_py.loop() ["variable" = "a"] {
          tiny_py.constant() ["value" = 0 : !i32]
        } {
          tiny_py.constant() ["value" = 100000 : !i32]
        } {
          tiny_py.assign() ["var_name" = "val"] {
            tiny_py.binaryoperation() ["op" = "add"] {
              tiny_py.var() ["variable" = "val"]
            } {
              tiny_py.var() ["variable" = "add_val"]
            }
          }
        }
      }
    }
  }
}
```

You should be able to see that, by operating on our high level _tiny_py_ dialect then it is obvious how to identify loops and for these to be transformed. If, by contrast, we were operating on the standard dialects (e.g. the _scf.while_ operation that our loop transforms into) then it would potentially be more complex to identify. This sort of tranformation can get quite complex, where we are identifying potential parallelism and then manipulating the IR as appropriate.

## Lowering to the standard MLIR dialect



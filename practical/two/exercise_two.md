# Exercise Two

In this practical we are going to get deeping into what is going on in our _tiny_py_ dialect and transformations by developing an enhancement to these in order to support a wider range of Python in our compiler. Specifically, we are going to add support for the loop construct which will provide an insight into how these things are coded.

Learning objectives are:

* A consolidation of the concepts explored in practical one
* To understand how dialects are expressed and can be modified
* Gain a more indepth understanding of transformations
* Awareness of the _for_ operation in the _scf_ dialect
* To show more advanced ways in which _mlir-opt_ can be driven to undertake additional transformations

Sample solutions to this exercise are provided in [sample_solutions](sample_solutions) in-case you get stuck or just want to compare your efforts with ours.

>**Having problems?**  
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

Remember, details about how to access your account on ARCHER2 and set up the environment that we will need for these practicals can be found on the [ARCHER2 setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/ARCHER2.md) page. If you are undertaking these tutorials locally then you can view the [local setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/local.md).

Irrespective of the machine you are using, it is assumed that you have a command line terminal in the _training-intro/practical/two_ directory.

## The starting point

The first thing to do is to have a look at the code _ex_two.py_ which loops up to 100000, adding a floating point value to a running total on each iteration and then printing out the final summed value at the end. 

```python
from python_compiler import python_compile   

@python_compile
def ex_two():
    val=0.0
    add_val=88.2
    for a in range(0, 100000):
      val=val+add_val
    print(val)

ex_two()
```

Firstly, let's generate the tiny py IR:

```bash
user@login01:~$ python3.10 ex_two.py
```

The output will look like the following, and you can also find it in the newly created _output.xdsl_ file.

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
      tiny_py.call_expr() ["func" = "print", "type" = !empty, "builtin" = !bool<"True">] {
        tiny_py.var() ["variable" = "val"]
      }
    }
  }
}
```

As you can see, we have the variable declarations and print statement at the end, but the loop and everything within it is missing. Don't worry, this is intentional and because our compiler is currently unaware of loops. The main purpose of this exercise is to add in support for handling loops. 

## Enhancing the frontend

### Supporting loops in the tiny py dialect

The first step is to enhance the _tiny_py_ dialect so that it is capable of representing a loop. Open up the _tiny_py.py_ file that is in _src/dialects_ and at line 125 you will see we have started the _Loop_ class. The _get_ function has been completed, as has the name, but we need to fill in the fields that will comprise the operation's fields (its operands, attributes, regions and results). 

Let's take a quick look at the code so far in this function (omitting the comments that are in the code to keep it a little shorter here) to explain what it is doing. This is below, and the _irdl_op_definition_ decorator annotates that this Operation follows the IRDL definition that we covered in the second lecture. The name of the operation is defined and the _get_ method creates an instance of this based upon the arguments provided. You can see that to create the operation a string (the variable name) and three operations are passed in. These comprise the four members of the operation, and it is these we need to add a definition for. 

```Python
@irdl_op_definition
class Loop(IRDLOperation):
    name = "tiny_py.loop"

    @staticmethod
    def get(variable: str | StringAttr,
            from_expr: Operation,
            to_expr: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        if isinstance(variable, str):
            # If variable is a string then wrap it in StringAttr
            variable=StringAttr(variable)

        res = Loop.build(attributes={"variable": variable}, regions=[Region([Block([from_expr])]),
            Region([Block([to_expr])]), Region([Block(body)])])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
```

You can see that we construct a string attribute (_StringAttr_) if the _variable_ is a raw string, and then these four fields are provided to the _Loop.build_ method call, where the _variable_ attribute is set and three regions specified. As we discussed in lecture one, each region contains a list of blocks which itself contains a list of operations. Hence we are constructing _Regions_ from a list of _Blocks_ and these from a list of operations. You can see from the method signature that _body_ is already a list of operations, and hence it is not wrapped in a list here.

There should be four fields, _variable_ which is an attribute of type _StringAttr_, and then three regions which are called _from_expr_, _to_expr_, and _body_. To understand the format these should follow then you can look at other operator definitions in the dialect, for instance _Var_ defines a string attribute member and _lhs_ and _rhs_ in _BinaryOperation_ are regions.

>**Not sure or having problems?**
> Please feel free to ask if there is anything you are unsure about, or you can check the [sample solution](https://github.com/xdslproject/training-intro/blob/main/practical/two/sample_solutions/tiny_py.py)

### Connecting up tiny py loop operation

Once you have completed the definition of this operation in the _tiny_py_ dialect then the next step is to generate this from the parser. If you open the _python_compiler.py_ file and navigate to line 100, you will see the function that handles a Python for loop. Again, we have started this off for you as illustrated by the code below (again we have removed comments from here for clarity). 

```Python
def visit_For(self, node):       
    contents=[]
    for a in node.body:
        contents.append(self.visit(a))
    expr_from=self.visit(node.iter.args[0])
    expr_to=self.visit(node.iter.args[1])
        
    # Now you need to construct the tiny_py Loop and return it
    return None  
```

In this code each member comprising the body of the loop is visited and the resulting operations are stored in the _contents_ list. We are then processing the loop bounds and for the purposes of this tutorial are simplifying things quite a bit here where we extract the lower and upper bounds from the Python _Range_ and set these as _expr_from_ and _expr_to_ respectively.

Currently this _visit_For_ function returns _None_ and instead we need to construct the _tiny_py_ dialect's _Loop_ operation. To do this you will use the _Loop.get_ function, providing the name of the loop variable (which you can obtain via _node.target.id_ and _expr_from_, _expr_to_, and _contents_ as the region arguments.

>**Not sure or having problems?**
> Please feel free to ask if there is anything you are unsure about, or you can check the [sample solution](https://github.com/xdslproject/training-intro/blob/main/practical/two/sample_solutions/python_compiler.py)

### Obtaining the tiny py IR

Now we have added support in our tiny py dialect for loops and instructed the parser how to generate these we can rerun our code to generate our IR which should now include the loop and its members. 

```bash
user@login01:~$ python3.10 ex_two.py
```

You should see the output below where, as you can see, the loop is now represented containing the _variable_ string as an attribute and the three regions (each of these are between the { } braces).

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
      tiny_py.call_expr() ["func" = "print", "type" = !empty, "builtin" = !bool<"True">] {
        tiny_py.var() ["variable" = "val"]
      }
    }
  }
}
```

## Lowering to the standard dialects

In [exercise one](https://github.com/xdslproject/training-intro/blob/main/practical/one/exercise_one.md) we ran the _tiny_py_to_standard.py_ to lower our tiny py IR down to the standard dialects in Static Single Assignment (SSA) form. We are now going to take a look at this transformation in more detail and add some support for transforming our loop to the _for_ operation in the _scf_ (structured control flow) dialect.

If you open the _tiny_py_to_standard.py_ file which is in the _src_ folder at the top level of the practical directory, then you will see the activities being undertaken to lower our _tiny_py_ dialect down to the standard MLIR dialects. Whilst this isn't particularly complicated, there is a reasonable amount going on in order to lower the different aspects.

Our objective is to transform the _Loop_ operation in our tiny py dialect into the _for_ operation of the standard _scf_ dialect, and if you look at line 173 of that file then you will see that we have started off the definition of this conversion. This function will first create an instance of the _BlockUpdatedVariables_ class (that we define at the start of that file) and take a copy of the SSA context. These are needed because we are about to go into a block, and standard MLIR dialects have no concept of variable storage. Instead, whenever a variable is updated then a new SSA context is created and this is refered to subsequently in the program. However blocks are a little different because their contents are private, so data is explicitly passed in as arguments and _yielded_ out of the block.

```python
def translate_loop(ctx: SSAValueCtx, block_description: BlockUpdatedVariables,
                  loop_stmt: tiny_py.Loop) -> List[Operation]:
    
    block_description=BlockUpdatedVariables()
    
    prev_ctx=ctx.copy()
    ops: List[Operation] = []
    for op in loop_stmt.body.blocks[0].ops:
        pass

    start_expr, start_ssa=translate_expr(ctx, block_description, loop_stmt.from_expr.blocks[0].ops[0])
    ctx[loop_stmt.variable]=start_ssa

    block_arg_types=[i32]
    block_args=[ctx[loop_stmt.variable]]
    for var_name in block_description.get():
        block_arg_types.append(ctx[var_name].typ)
        block_args.append(prev_ctx[var_name])

    block = Block()
    
    body=Region()
    body.add_block(block)

    loop_increment=arith.Constant.from_int_and_width(1, 32)
    loop_variable_inc=arith.Addi.get(ctx[loop_stmt.variable], loop_increment.results[0])

    block_after = Block.from_arg_types(block_arg_types)
    
    yield_stmt=generate_yield(ctx, block_description, loop_variable_inc)
    
    body_after=Region()
    body_after.add_block(block_after)

    block_updated=None

    return start_expr+[scf.While.get([block_args], [[i32]], body, body_after)]
```

In the code above you can see that we create the _ops_ list, which is currently empty and we need to complete the subsequent loop. To do this you call the _translate_stmt_ function with the SSA context _ctx_, _block_description_ member and _op_ operation. The result is then added to the _ops_ list (e.g. via _ops.append_). 

You can see that we translate the _from_expr_ region in the _tiny_py_ dialect, which results in both the expression to be placed in the resulting IR and also the SSA context which we can use in subsequent references. However the translation for the _to_expr_ region is missing, therefore add this in following the approach used with the _from_expr_ region. Once you have done this, we also need to generate the loop's exit condition check which is a two stage process, undertaking the integer comparison and then using the result to drive exiting the loop as appropriate. For the comparison we will use the _Cmpi_ operator from the standard _arith_ dialect, and you can generate this via ``compare=arith.Cmpi.get(end_ssa, start_ssa, 1)``. For the condition check we are going to use the _Condition_ operator from the _scf_ dialect, and you can create this via ``condition=scf.Condition.get(compare.results[0], start_ssa)`` . Note how we are not passing _compare_ to the _Condition.get_ function, but instead it's result. This _result[0]_ field contains the SSA context, so we are instructing the _Condition_ operator to check the result of the _compare_ operation. 

Now we have constructed these operators we need to add them to the block, _block_, just after it is created. To do this, you will need to use the _add_ops_ function on the _block_ variable, which adds a list of operations. Because _end_expr_ is already a list, then we can provide something like ``end_expr+[compare, condition]`` as arguments to the function.

With the standard MLIR while loop there are actually two blocks, a before and after. The former contains the exit condition check, whereas the later contains the loop body and at the end of this increments the loop variable. This increment operation can be seen in the Python code above, where we are using the _Addi_ operation from the standard _arith_ dialect to add a constant (in this case the _Constant_ operation defined in the line above that holds the value 1 and is 32 bits wide) to the loop variable. 

It can also be seen how we are creating this other block, _block_after_, with accepts a number of input arguments (the inputs to the block) based upon the body of the loop that has been translated and the _yield_stmt_ operation which contains the corresponding SSA contexts for these to make any updates visible outside the block. Just after this yield statement, before we construct the region, we need to add the operations to this block. This is similar to how we did it for the _before_ block, so we can use the _add_ops_ function on the _block_after_ variable, and we want to add the _ops_, _loop_increment_, _loop_variable_inc_ and _yield_stmt_ operations. Note that a list needs to be provided, one or more of these is already a list whereas others are individual operations (so as before you will need to wrap some in a list).

Once you have done this then the last thing to do is to make our translation pass call into the _translate_loop_ function whenever it encounters the _Loop_ operation in the _tiny_py_ dialect. If you go to the _try_translate_stmt_ function (line 135), you can see how we do this for other statements. You will need to add an extra condition checking whether _op_ is an instance of the _tiny_py.Loop_ class, and if so returning the results from calling _translate_loop_ with the same arguments as the other calls in that function.

Now run the translation pass, ``tinypy-opt two.xdsl -p tiny-py-to-standard -t mlir`` and as you can see we are outputting in MLIR format. You should see the following generated, where you can see the while loop from the scf dialect which contains the body, our loop comparisons, and the yield.

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
    %7 = "scf.for"(%4, %5, %6, %0) ({
    ^0(%8 : index, %9 : f32):
      %10 = "arith.addf"(%9, %1) : (f32, f32) -> f32
      "scf.yield"(%10) : (f32) -> ()
    }) : (index, index, index, f32) -> f32
    %11 = "llvm.mlir.addressof"() {"global_name" = @str0} : () -> !llvm.ptr<!llvm.array<3 x i8>>
    %12 = "llvm.getelementptr"(%11) {"rawConstantIndices" = array<i32: 0, 0>} : (!llvm.ptr<!llvm.array<3 x i8>>) -> !llvm.ptr<i8>
    %13 = "arith.extf"(%7) : (f32) -> f64
    "func.call"(%12, %13) {"callee" = @printf} : (!llvm.ptr<i8>, f64) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<3 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "%f\n", "unnamed_addr" = 0 : i64} : () -> ()
  "func.func"() ({
  }) {"sym_name" = "printf", "function_type" = (!llvm.ptr<i8>, f64) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
```

## Compile and run

We are now ready to feed this into LLVM and compile the code, similarly to exercise one you should create a file with the _.mlir_ ending, for instance _ex_two.mlir_ and into it copy the output from the above passes (redirecting stdio to the file is probably easiest). The execute the following (you can see we provide _-convert-std-to-llvm_ as an argument to _mlir_opt_ which converts all standard dialects to the LLVM dialect):

```bash
user@login01:~$ mlir-opt --pass-pipeline="builtin.module(loop-invariant-code-motion, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-arith-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts)" ex2.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

A submission script called _sub_ex2.srun_ is prepared and you will need to execute `sbatch sub_ex2.srun` at the command line. This will batch queue the job and, when a compute node is available, the executable will run and output stored in a file that you can be viewed.

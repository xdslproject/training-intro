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

Our objective is to transform the _Loop_ operation in our tiny py dialect into the _for_ operation of the standard _scf_ dialect, and if you look at line 173 of the [tiny_py_to_standard.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/tiny_py_to_standard.py) file then you will see that we have started off the definition of this conversion. This function is below, with the comment _Needs to be completed!_ highlighting the parts that are missing and you need to add:

```python
def translate_loop(ctx: SSAValueCtx,
                  loop_stmt: tiny_py.Loop) -> List[Operation]:    

    # First off lets translate the from (start) and to (end) expressions of the loop
    start_expr, start_ssa=translate_expr(ctx, loop_stmt.from_expr.blocks[0].ops[0])
    end_expr, end_ssa=None, None # Needs to be completed!
    # The scf.for operation requires indexes as the type, so we cast these to
    # the indextype using the IndexCastOp of the arith dialect
    start_cast = arith.IndexCastOp.get(start_ssa, IndexType())
    end_cast = None # Needs to be completed!
    # The scf.for operation requires a step (number of iterations to increment
    # each iteration, we just create this as 1)
    step_op = arith.Constant.create(attributes={"value": IntegerAttr.from_index_int_value(1)}, result_types=[IndexType()])

    # This is slightly more complex, we need to provide as arguments to the block
    # variables which are updated and then yield these out. We use a visitor
    # to visit all assignments and gather the names of the variables that are assigned
    assigned_var_finder=GetAssignedVariables()
    for op in loop_stmt.body.blocks[0].ops:
        assigned_var_finder.traverse(op)

    # Based on the above information we build the list of block arguments, the first
    # element is always the operand which represents the current loop iteration
    # which is of type index
    block_arg_types=[IndexType()]
    block_args=[]
    for var_name in assigned_var_finder.assigned_vars:
        block_arg_types.append(ctx[StringAttr(var_name)].typ)
        block_args.append(ctx[StringAttr(var_name)])

    # Create the block with our arguments, we will be putting into here the
    # operations that are part of the loop body
    block = Block(arg_types=block_arg_types)

    # In the SSA context that is passed into the translation of the loop
    # body we set each assigned variable to reference the corresponding argument
    # to the block
    c = SSAValueCtx(dictionary=dict(), parent_scope=ctx)
    for idx, var_name in enumerate(assigned_var_finder.assigned_vars):
      c[StringAttr(var_name)]=block.args[idx+1]

    # Now lets visit each operation in the loop body and build up the operations
    # which will be added to the block
    ops: List[Operation] = []
    for op in loop_stmt.body.blocks[0].ops:
        pass # Needs to be completed!        

    # We need to yield out assigned variables at the end of the block
    yield_stmt=generate_yield(c, assigned_var_finder.assigned_vars)
    block.add_ops(ops+[yield_stmt])
    body=Region()
    body.add_block(block)

    # Build the for loop operation here
    for_loop=None # Needs to be completed!

    # From now on, whenever the code references any variable that was assigned
    # in the body of the loop we need to use the corresponding loop result
    for i, var_name in enumerate(assigned_var_finder.assigned_vars):
      ctx[StringAttr(var_name)]=for_loop.results[i]

    return start_expr+end_expr+[start_cast, end_cast, step_op, for_loop]
```

There is quite a bit going on here, so let's first complete the missing parts and then we will explore what the other aspects are doing too. You can see at line 181 of this file the line `end_expr, end_ssa=None, None # Needs to be completed!`. This is for handling the upper loop bounds which is an expression, and we need to call the corresponding function to convert this from the tiny py dialect into the standard dialects. You can see from the line above how this is handled for start, or from, expression, and here we can do very similar for this end expression using `loop_stmt.to_expr.blocks[0].ops[0]` as the second argument to the `translate_expr` call. This `translate_expr` call returns two things, firstly the operations that the _to_ expression corresponds to, and secondly the resulting SSA value that can be used by subsequent operations to reference this.

Based upon how we have expressed this, the _start_ssa_ and _end_ssa_ values are of type integer, and the _for_ operation of the _scf_ dialect requires the lower and upper loop bound operands to be of type _index_ . Therefore we need to issue an operation that converts from an _integer_ to and _index_. If you look at line 185 of the file (line 10 of the snippet above) you will see the line `end_cast = None # Needs to be completed!` . The line above issues this conversion for _start_ssa_, so by following what was done there you should issue the same conversion operation for _end_ssa_.

In the code above you can see that we create the _ops_ list (line 219 of [tiny_py_to_standard.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/tiny_py_to_standard.py), and iterate through the operations of the loop body, but currently do not do anything with them. We therefore need to complete this, and to do that you call the _translate_stmt_ function with the SSA context _ctx_, and _op_ operation. The result from this call, which is a list, should then be added to the _ops_ list in the next line (e.g. if the result from the _translate_stmt_ function call is assigned to _stmt_ops_, then the code to add this would be `ops += stmt_ops`). 

Now we have done all of this we just need to create the _for_ operation in the _scf_ dialect, this missing code is towards the end of the snippet above (`for_loop=None # Needs to be completed!`) and at line 231 of [tiny_py_to_standard.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/tiny_py_to_standard.py). To create the operation we will call the _get_ method of the _for_ operations, i.e. `scf.For.get(..)` and provide to this operation five arguments. These arguments are the SSA result of the _start_cast_ operation, the SSA result of the _end_cast_ operation, the SSA result of the _step_op_ operation (which defines the step increment each iteration), _block_args_ (which we will describe in a moment), and _body_ which is a list of operations comprising the body of the loop. _block_args_ and _body_ can be passed directly as arguments 4 and 5, whereas for the other arguments we need to look up the SSA value from the operation which is avilable in the `results` member. For instance, for _start_cast_ you would pass `_start_cast.results[0]`.

We have completed the missing parts and are now ready to run the translation pass, ``tinypy-opt output.xdsl -p tiny-py-to-standard -t mlir`` and output MLIR formatted IR. You should see the following generated, where you can see the for loop from the scf dialect which contains the loop bounds and body.

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

>**Not sure or having problems?**
> Please feel free to ask if there is anything you are unsure about, or you can check the [sample solution](https://github.com/xdslproject/training-intro/blob/main/practical/two/sample_solutions/tiny_py_to_standard.py)

### Exploring the function in more detail

We have skipped over some of the details in this function that are worth highlighting. In a loop, or any region that is being iterated upon, values will likely change from one iteration to the next. Therefore we need to define which values are updated and ensure that these are being used per iteration. Therefore blocks can have arguments and the following snippet, taken from the IR above illustrates this.

```
%7 = "scf.for"(%4, %5, %6, %0) ({
    ^0(%8 : index, %9 : f32):
      %10 = "arith.addf"(%9, %1) : (f32, f32) -> f32
      "scf.yield"(%10) : (f32) -> ()
    }) : (index, index, index, f32) -> f32
```

Here we have the _for_ operation, with the lower bound, upper bound, and step passes as arguments. But furthermore, you can see _%0_ is also passed as an argument and this is the initial value of _val_ that we will be incrementing. The line below, `^0(%8 : index, %9 : f32):` defines a block with arguments provided to the block. With a _for_ operation, the first argument to it's body's block is the loop index (_%8_) and the second argument onwards are SSA values that are inputs to the block. At the end of this block you can see the _yield_ operation, with _%10_, the result of the floating point addition, as an argument. Effectively, this will set _%10_ to be the result of a single execution of the block, and on the next iteration of the loop the block argument (_%9%_) will refer to this value rather than the initial value of _%0_ that was provided. After the last iteration of the _for_ operation, this yielded value is set as the result odf the entire _for_ operation as _%7_. Zero, one or more SSA values can be yielded from a block.

The challenge is knowing which SSA values need to be included in the block as arguments, which need to be yielded, and then later on in the IR (e.g. when calling the _printf_ function) using the SSA value resulting from the loop rather than the initial SSA value. This is what the other parts of the _translate_loop_ function are doing, where we have written a simple _GetAssignedVariables_ visitor (which can be seen at line 32 of [tiny_py_to_standard.py](https://github.com/xdslproject/training-intro/blob/main/practical/src/tiny_py_to_standard.py)) which will visit all assignments to track which variables are updated. These are then used as the block and yield operation arguments.

## Compile and run

We are now ready to feed this into LLVM and compile the code, similarly to exercise one you should create a file with the _.mlir_ ending, for instance _ex_two.mlir_ and into it copy the output from the above passes (redirecting stdio to the file is probably easiest). The execute the following (you can see we provide _-convert-std-to-llvm_ as an argument to _mlir_opt_ which converts all standard dialects to the LLVM dialect):

```bash
user@login01:~$ mlir-opt --pass-pipeline="builtin.module(loop-invariant-code-motion, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-arith-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts)" ex2.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

A submission script called _sub_ex2.srun_ is prepared and you will need to execute `sbatch sub_ex2.srun` at the command line. This will batch queue the job and, when a compute node is available, the executable will run and output stored in a file that you can be viewed.

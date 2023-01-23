# Exercise Two

In this practical we are going to _get into the guts_ of what is going on in our _tiny_py_ dialect and transformations by developing an enhancement to these in order to support a wider range of Python in our compiler. Specifically, we are going to add support for the loop construct which will provide an insight into how these things are coded.

Sample solutions to this exercise are provided in [sample_solutions](sample_solutions) in-case you get stuck or just want to compare your efforts with ours.

>**Having problems?**  
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

## The starting point

The first thing to do is to have a look at the code _ex_two.py_ which loops up to 100000, adding a value to a running total on each iteration. Firstly let's generate the tiny py IR to see what this currently looks like, do this by issuing `python3.10 ex_two.py`, the output will look like the following:

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
    }
  }
}
```

As you can see, we have the variable declarations, but the loop and everything within it is missing, which is because our compiler is currently unaware of loops and as such unable to represent it. 

## Enhancing the dialect

The first step is to enhance the _tiny_py_ dialect so that it is capable of representing a loop. Open up the _tiny_py.py_ file that is in _src/dialects_ and at line 125 you will see we have started the _Loop_ class. The _get_ function has been completed, as has the name, but we need to fill in the field that will be present here. Let's take a quick look at the code so far in this function (omitting the comments that are in the code to keep it a little shorter here) to explain what it is doing. This is below, and the _irdl_op_definition_ decorator annotates that this Operation follows the IRDL definition that we covered in the second lecture. The name of the operation is defined and the _get_ method creates an instance of this based upon the arguments provided. You can see that to create the operation a string (the variable name) and three operations are passed in. There are four members, that we will need to specify, and these can be seen in the _Loop.build_ construct where the _variable_ attribute is set and three regions. As we discussed in lecture one, each region is a list of operations, hence _from_expr_ and _to_expr_ needed to be wrapped as a list (you can see that _body_ is already a list of operations).

```Python
@irdl_op_definition
class Loop(Operation):
    name = "tiny_py.loop"

    @staticmethod
    def get(variable: str,
            from_expr: Operation,
            to_expr: Operation,
            body: List[Operation],
            verify_op: bool = True) -> If:
        res = Loop.build(attributes={"variable": variable}, regions=[[from_expr], [to_expr], body])
        if verify_op:
            # We don't verify nested operations since they might have already been verified
            res.verify(verify_nested_ops=False)
        return res
```


There should be four fields, _variable_ which is an attribute of type _StringAttr_, and then three regions which are called _from_expr_, _to_expr_, and _body_. To understand the format these should follow then you can look at other operator definitions in the dialect, for instance _Var_ defines a string attribute member and _lhs_ and _rhs_ in _BinaryOperation_ are regions.

Once you have completed the definition of this operation in the _tiny_py_ dialect then the next step is to generate this from the parser. If you open the _python_compiler.py_ file and navigate to line 93, you will see the function that handles a Python for loop. Again, we have started this off for you and the code below illustrates this (again we have removed comments from here for clarity). It can be seen that each member of the loop's body is visited and the resulting operations are stored in a loop. For the loop bounds we are simplifying things quite a bit here to make our life easier, where even though a Python _Range_ is provided, we extract the low and high members as expressions and set these as _expr_from_ and _expr_to_ respectively. A better, and more flexible approach, would be to encode the range in the _tiny_py_ dialect and then have this evaluated each loop iteration but that would make things more complex and is left as an exercise to the interested participant if they so wish.

```Python
    def visit_For(self, node):       
        contents=[]
        for a in node.body:
            contents.append(self.visit(a))
        expr_from=self.visit(node.iter.args[0])
        expr_to=self.visit(node.iter.args[1])
        
        return None  
```

Instead of returning _None_ from this visit function we need to construct the _Loop_ operation from the _tiny_py_ dialect. You will use the _Loop.get_ function, providing the name of the loop variable (which you can obtain via _node.target.id_ and the three regions).

Once you have done this, rerun _python3.10 ex_two.py_ and you should see the output below where, as you can see, the loop is now represented containing the _variable_ string as an attribute and the three regions (each of these are between the { } braces).

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
    }
  }
}
```

## Lowering to the standard dialects

If you open the _tiny_py_to_standard.py_ file then you will see the activities being undertaken to lower our _tiny_py_ dialect down to the standard MLIR dialects. As we have added the _Loop_ operation into our dialect we also need to further enhance this transformation to be able to handle it. Our objective is to transform this into the _while_ operation from the standard _scf_ dialect, and if you look at line 173 of that file then you will see that we have started off the definition of this conversion. This function will first create an instance of the _BlockUpdatedVariables_ class (that we define at the start of that file) and take a copy of the SSA context. These are needed because we are about to go into a block, and standard MLIR dialects have no concept of variable storage. Instead, whenever a variable is updated then a new SSA context is created and this is refered to subsequently in the program. However blocks are a little different because their contents are private, so data is explicitly passed in as arguments and _yielded_ out of the block.

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
    %3 = "scf.while"(%2, %0) ({
      %4 = "arith.constant"() {"value" = 100000 : i32} : () -> i32
      %5 = "arith.cmpi"(%4, %2) {"predicate" = 1 : i64} : (i32, i32) -> i1
      "scf.condition"(%5, %2) : (i1, i32) -> ()
    }, {
    ^0(%6 : i32, %7 : f32):
      %8 = "arith.addf"(%0, %1) : (f32, f32) -> f32
      %9 = "arith.constant"() {"value" = 1 : i32} : () -> i32
      %10 = "arith.addi"(%2, %9) : (i32, i32) -> i32
      "scf.yield"(%10, %8) : (i32, f32) -> ()
    }) : (i32, f32) -> i32
    "func.return"() : () -> ()
  }) {"sym_name" = "ex_two", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
}) : () -> ()
```

## Compile and run

We are now ready to feed this into LLVM and compile the code, similarly to exercise one you should create a file with the _.mlir_ ending, for instance _ex_two.mlir_ and into it copy the output from the above passes (redirecting stdio to the file is probably easiest). The execute the following (you can see we provide _-convert-std-to-llvm_ as an argument to _mlir_opt_ which converts all standard dialects to the LLVM dialect):

```bash
user@login01:~$ mlir-opt -convert-std-to-llvm ex_two.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

A submission script called _sub_ex2.srun_ is prepared and you will need to execute `sbatch sub_ex2.srun` at the command line. This will batch queue the job and, when a compute node is available, the executable will run and output stored in a file that you can be viewed.

# Exercise Two

In this practical we are going to _get into the guts_ of what is going on in our _tiny_py_ dialect and transformations by developing an enhancement to these in order to support a wider range of Python in our compiler. Specifically, we are going to add support for the loop construct which will provide an insight into how these things are coded.

Sample solutions to this exercise are provided in _sample_solutions_ in-case you get stuck or want to compare your efforts with ours.

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
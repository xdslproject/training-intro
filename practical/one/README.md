# Practical One

## Getting started

Firstly you should connect to ARCHER2 via the login credentials supplied to you by the instructors, connecting to the machine via `ssh username@login.archer2.ac.uk`. Once you have connected you should then change into your _work_ directory, via issuing `cd /work/y14/y14/$USER`.

The guest accounts that you are using have been configured to contain the requires files for this practical and the correct environment settings.

## First steps

If you _cd_ into the _training-intro/practical/one_ directory that should be visible from your work directory, you will see the _code.py_ file. This is a simple, contrived, example but is useful to illustrate the central concepts behind MLIR and xDSL. Imagine that we want to write a Python compiler which enables us to decorate functions and for these to then be compiled and executed natively. Let's take a look inside the _code.py_ file to see how this is driven.

```python
from python_compiler import python_compile   

@python_compile
def hello_world():
    print("Hello world!")

hello_world()
```

You can see that we have created a function, called _hello_world_ which prints a message to standard out. We import our own _python_compiler_ Python script which defines the _python_compile_ decorator and this is used to decorate the functions that we wish to compile. The _python_compiler_ script can be found in the _training-intro/src_ directory, and we will look at this a bit later on.

Now let's run this by issuing `python3.10 code.py`, with the result looking like:

```
builtin.module(){
  tiny_py.module() {
    tiny_py.function() ["fn_name" = "hello_world", "return_var" = !tiny_py.emptytoken, "args" = []] {
      tiny_py.call_expr() ["func" = "print", "type" = !empty, "builtin" = !bool<"True">] {
        tiny_py.constant() ["value" = "Hello world!"]
      }
    }
  }
}
```

Congratulations, you have run your first bit of xDSL/MLIR! Let's take a look at what this actually means, firstly as we talked about in the first presentation all operations are prefixed with the dialect name that they belong to. Here you can see that we are using two dialects, the _builtin_ dialect for the _module_ operation and the _tinypy_ dialect for other operations. Hopefully from looking at this IR representation you can see how it corresponsd to the origional Python code, where _tiny_py.function_ defines the _hello_world_ function and _tiny_py.call_expr_ represents the call into the _print_ function, which is using the attributed _builtin_ to represent whether it is a built in Python function or user defined. Lastly, you can also see how the value _"Hello World!"_ is represented by the _tiny_py.constant_ operation and sits hierarchically inside the _tiny_py.call_expr_ operation. This is an example of nesting regions, where the _tiny_py.call_expr_ operation contains a region, which contains blocks (one in this case) which then contains this operation.

## Moving towards an executable

The _tiny_py_ dialect that you have seen above is one that we have defined for this exercise. It's intentionally very simple, can be found at _src/dialects/tiny_py.py_ and we will modify it a bit later on in the practical. Now thinking about wanting to get an executable out of this, we have a few options. Firstly we could write some transformations that directly generate LLVM-IR (or something else such as C) from this dialect. That would be OK, and indeed this is how the Devito DSL uses xDSL, but potentially time consuming and there is an easier way. Instead, we can transform to the standard MLIR dialects will all have existing transformations from these into LLVM-IR. To this end we need to undertake a transformation pass on the IR generated above to undertake this transformation. 

To this end we use the _tinypy_opt_ tool, which enables us to drive the parsing, printing, and manipulations of IRs from the command line. LLVM and MLIR have adopted _-opt_ tools as a standard, which xDSL adopts. At the command line execute _./tinypy-opt eg_one.xdsl -p tiny-py-to-standard -t mlir_ , the output will appear as:

```
"builtin.module"() ({
"llvm.mlir.global"() ({
  }) {addr_space = 0 : i32, constant, global_type = !llvm.array<12 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str0", unnamed_addr = 0 : i64, value = "Hello world!"} : () -> ()
  "func.func"() ({    
    %0 = "llvm.mlir.addressof"() {global_name = @str0} : () -> !llvm.ptr<array<12 x i8>>
    %1 = "llvm.getelementptr"(%0) {rawConstantIndices = array<i32: 0, 0>} : (!llvm.ptr<array<12 x i8>>) -> !llvm.ptr<i8>
    "llvm.call"(%1) {"callee" = @print} : (!llvm.ptr<i8>) -> (i32)
    "func.return"() : () -> ()
  }) {"sym_name" = "hello_world", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
}) : () -> ()
```

There are a few things going on here, so let's unpack it step by step. Firstly, we are providing the IR stored in the _eg_one.xdsl_ file as an input to the _tinypy-opt_ tool. This is simply the IR that was generated in the previous section and then, using the _-p_ flag we are instructing the that the _tiny-py-to-standard_ pass should be run over that IR to transform it. Lastly, the _-t_ flag instructs the tool which target to output the IR for, in this case we are selecting MLIR format so that it can be fed directly into the MLIR tooling itself.

You can see that this IR looks quite different to the IR previously where it is in a much flatter form and using SSA. It's also closer to the concrete implementation level, for instance the argument _"Hello World!"_ is now declared as a global and the memory reference of this is passed as the argument to the _print_ function. Whilst this is still human readable and debuggable, effectively we have removed a lot of the Python-ness that is present in the _tiny_py_ dialect in this pass.

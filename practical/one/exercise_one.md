# Practical One

In this practical we are going to look at compiling a simple Python function to run on the compute node of ARCHER2, a Cray EX. This will give you an introduction to MLIR and xDSL, as well as getting you onto the ARCHER2 supercomputer where we will be doing our practicals today.

## Getting started

Firstly, you should connect to ARCHER2 via the login credentials supplied to you by the instructors. You will need to connect to the machine via ssh, for instance `ssh username@login.archer2.ac.uk`. Once you have connected you should then change into your _work_ directory, via issuing `cd /work/y14/y14/$USER`, as this is the filesystem that is visible to the compute nodes.

The guest accounts that you are using have been configured to contain the requires files for this practical and the correct environment settings such as PYTHONPATHS. If you do this exercise on your own machines then you will need to download these from our website [xdsl.dev](https://www.xdsl.dev).

## First steps

Once you have changed to your work directory, now _cd_ into the _training-intro/practical/one_ directory that should be visible. In this you will see the _code.py_ file which we are going to work with in this exercise. This is a simple, contrived, Python example but is useful to illustrate the central concepts behind MLIR and xDSL. Our objective is to write a Python compiler which enables programmers to decorate functions and for these to then be compiled and executed natively. 

Let's take a look inside the _code.py_ file to see how this is driven.

```python
from python_compiler import python_compile   

@python_compile
def hello_world():
    print("Hello world!")

hello_world()
```

You can see that we have started with a function, called _hello_world_, which prints a message to standard out. The code imports the _python_compiler_ Python script which defines the _python_compile_ decorator, and this is used to decorate the functions that we wish to compile. The _python_compiler_ script can be found in the _training-intro/src_ directory, this parses the Python code and we will look at this a bit later on.

Now let's run this by issuing `python3.10 code.py`, you should see a result similar to:

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

Congratulations, you have run your first bit of xDSL/MLIR, now let's take a look at what this means! Firstly, as we talked about in the initial presentation, all operations are prefixed with the dialect name that they belong to. Here you can see that we are using two dialects, the _builtin_ dialect for the _module_ operation and the _tinypy_ dialect for other operations. Hopefully from looking at this IR representation you can see how it corresponsd to the origional Python code, where _tiny_py.function_ defines the _hello_world_ function and _tiny_py.call_expr_ represents the call into the _print_ function, which is using the attributed _builtin_ to represent whether it is a built in Python function or user defined. 

Lastly, you can also see how the value _"Hello World!"_ is represented by the _tiny_py.constant_ operation and sits hierarchically inside the _tiny_py.call_expr_ operation. This is an example of nesting regions, where the _tiny_py.call_expr_ operation contains a region, which contains blocks (one in this case) which then contains this operation.

## Moving towards an executable

The _tiny_py_ dialect that you have seen above is one that we have defined for this exercise. It's intentionally very simple, can be found at _src/dialects/tiny_py.py_ and we will modify it a bit later on in the practical. Wanting to get an executable out of this that we can actually run on a supercomputer, we have a few options. Firstly we could write some transformations that directly generate LLVM-IR (or something else such as C) from this dialect. That would be OK, and indeed this is how the Devito DSL uses xDSL, but potentially time consuming and there is an easier way to achieve what we want. Instead, we can transform to the standard MLIR dialects which themselves have existing transformations into LLVM-IR. To this end we need to undertake a transformation pass on the IR generated above to undertake such a transformation. 

We use the _tinypy_opt_ tool, which enables us to drive the parsing, printing, and manipulations of IRs from the command line. LLVM and MLIR have adopted _-opt_ tools as a standard, which xDSL also adopts. Create a new file called _ex_one.xdsl_ and in it place the output of the previous section (or alternatively redirect stdio from the command to this file). At the command line execute _./tinypy-opt ex_one.xdsl -p tiny-py-to-standard -t mlir_ , the output will appear similar to:

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

You can see that this IR looks quite different to the IR previously where it is in a much flatter form and using SSA. This goes back to the point we made in the lectures about progressive lowering, where we have taken something more structured (the _tiny_py_) and lowered it to a representation that is closer to the concrete implementation level. For instance, the argument _"Hello World!"_ is now declared as a global and its memory reference is passed as the argument to the _print_ function. Whilst this is still human readable and debuggable, effectively we have removed a lot of the Python-ness that is present in the _tiny_py_ dialect in this pass.

## Handling built in function calls

There is just one more activity that needs to happen before we can build the code, and that is to handle the the _print_ function. Specifically, _print_ is an inbuilt Python function whereas when we compile our code we want it to call _printf_ from the standard library. Furthermore, as this is not a user defined function, we need to tell the compiler that it is external.

We have developed another transformation pass, _apply-builtin_, which will manipulate the IR to handle calling built in functions. Specifically, this transformation is looking for nodes in the IR of type _tiny_py.call_expr_ and then checking the _builtin_ flag. If this is true it will change the name of the function to _printf_, add a newline at the end of the string, and also prefix an operation at the top level of the module to direct that this is an external function. 

To use this pass then execute `./tinypy-opt eg_one.xdsl -p apply-builtin,tiny-py-to-standard -t mlir_` which will run the _apply-builtin_ pass first and then the _tiny-py-to-standard_ pass second. The output will look something like:

```
"builtin.module"() ({
"llvm.mlir.global"() ({
  }) {addr_space = 0 : i32, constant, global_type = !llvm.array<14 x i8>, linkage = #llvm.linkage<internal>, sym_name = "str0", unnamed_addr = 0 : i64, value = "Hello world!\0A\00"} : () -> ()
"llvm.func"() ({
  }) {CConv = #llvm.cconv<ccc>, function_type = !llvm.func<i32 (ptr<i8>, ...)>, linkage = #llvm.linkage<external>, sym_name = "printf"} : () -> ()
  "func.func"() ({    
    %0 = "llvm.mlir.addressof"() {global_name = @str0} : () -> !llvm.ptr<array<14 x i8>>
    %1 = "llvm.getelementptr"(%0) {rawConstantIndices = array<i32: 0, 0>} : (!llvm.ptr<array<14 x i8>>) -> !llvm.ptr<i8>
    "llvm.call"(%1) {"callee" = @printf} : (!llvm.ptr<i8>) -> (i32)
    "func.return"() : () -> ()
  }) {"sym_name" = "hello_world", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
}) : () -> ()
```

Compared to the previous IR, you can see that we have the addition of the _llvm.func_ operation at the top level of the module, the _Hello world!_ string has been extended with a new line and it is calling into _printf_ rather than _print_ now.

## Generating the executable and running

The IR that we are generating from our tool is now is ready to be fed into LLVM and compiled. To do this you should create a file with the _.mlir_ ending, for instance _ex_one.mlir_ and into it copy the output from the above passes (redirecting stdio to the file is probably easiest). Next we execute:

```bash
user@login01:~$ mlir-opt --convert-func-to-llvm ex_one.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

Here we are calling the _mlir-opt_ tool, part of the core LLVM distribution, to undertake some conversion of the standard dialects into the LLVM dialect, which is then fed into the _mlir-translate_ tool to generate LLVM-IR. This IR is then used as an input to Clang which builds the executable called _test_. If you are interested you can run the commands in isolation to see the different IRs that are generated from the commands.

### Running on compute nodes

Now we are good to go and it's time to run our executable on a compute nodes of ARCHER2. A submission script called _sub_ex1.srun_ is prepared and you will need to execute `sbatch sub_ex1.srun` at the command line. This will batch queue the job and, when a compute node is available, the executable will run and output stored in a file that you can be viewed.

## Well done!

Well done - you have got your first end to end code compiling and running on a supercomputer via xDSL and MLIR. In the next part of this practical we are going to enhance the dialects and transformations to extend the subset of Python that our tool supports

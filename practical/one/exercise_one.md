# Exercise One

In this practical we are going to look at compiling a simple Python function to run on the compute node of ARCHER2, a Cray EX. This will give you an introduction to MLIR and xDSL, as well as getting you onto the ARCHER2 supercomputer where we will be doing our practicals today. 

Learning objectives are:

* To provide a concrete end-to-end flow from source code to executable via xDSL and MLIR
* To illustrate how different dialects and transformations can be mixed and interact within the compilation flow
* To show how the different xDSL, MLIR, and LLVM tools can be used to manipulate the IR
* To introduce the central MLIR concepts in a practical fashion

>**Having problems?**
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

## Getting started

Details about how to access your account on ARCHER2 and set up the environment that we will need for these practicals can be found on the [ARCHER2 setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/ARCHER2.md) page. If you are undertaking these tutorials locally then you can view the [local setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/local.md) and if on a local machine ensure you have sourced the _environment.sh_ file from the _practical_ directory to set up your environment.

Irrespective of the machine you are using, it is assumed that you have a command line terminal in the _training-intro/practical/one_ directory.

## The structure of the practical content

If you look in the [practical](https://github.com/xdslproject/training-intro/edit/main/practical) directory you will see a number of sub-folders. [one](https://github.com/xdslproject/training-intro/edit/main/practical/one), [two](https://github.com/xdslproject/training-intro/edit/main/practical/two) and [three](https://github.com/xdslproject/training-intro/edit/main/practical/three) correspond to the three practical exercises with [general](https://github.com/xdslproject/training-intro/edit/main/practical/general) containing general instructions that are linked from the practicals. The [src](https://github.com/xdslproject/training-intro/edit/main/practical/src) directory contains code to drive our Python compiler and transformations, and [src/dialect](https://github.com/xdslproject/training-intro/edit/main/practical/src/dialects) contains the tiny py dialect that we will be using here.

Practicals two and three will require you to edit the source code files in the [src](https://github.com/xdslproject/training-intro/edit/main/practical/src) directory.

## First steps

Once you have changed to your work directory, now _cd_ into the _training-intro/practical/one_ directory that should be visible. In this you will see the _code.py_ file which we are going to work with in this exercise. This is a simple, contrived, Python example but is useful to illustrate the central concepts behind MLIR and xDSL. Our objective is to write a Python compiler which enables programmers to decorate functions and for these to then be compiled and executed natively. 

Let's take a look inside the _ex_one.py_ file to see how this is driven.

```python
from python_compiler import python_compile

@python_compile
def hello_world():
    print("Hello world!")

hello_world()
```

You can see that we have started with a function, called _hello_world_, which prints a message to standard out. The code imports the _python_compiler_ Python script which defines the _python_compile_ decorator, and this is used to decorate the functions that we wish to compile. The _python_compiler_ script can be found in the _training-intro/src_ directory, this parses the Python code and we will look at this a bit later on.

Let's do the first step of our compilation by issuing the following:

```bash
user@login01:~$ python3.10 ex_one.py
```

Congratulations, you have run your first bit of xDSL/MLIR! This won't actually execute the code, but instead builds up an Intermediate Representation (IR) of the code that we wish to compile in that function. The output of this is as follows:

```
builtin.module() {
  tiny_py.module() {
    tiny_py.function() ["fn_name" = "hello_world", "return_var" = !empty, "args" = []] {
      tiny_py.call_expr() ["func" = "print", "type" = !empty, "builtin" = !bool<"True">] {
        tiny_py.constant() ["value" = "Hello world!"]
      }
    }
  }
}
```

Now let's take a look at what this means. Firstly, as we talked about in the initial presentation, all operations are prefixed with the dialect name that they belong to. Here you can see that we are using two dialects, the _builtin_ dialect for the _module_ operation and the _tinypy_ dialect for other operations. Hopefully from looking at this IR representation you can see how it corresponds to the origional Python code, where _tiny_py.function_ defines the _hello_world_ function and _tiny_py.call_expr_ represents the call into the _print_ function, which is using the attributed _builtin_ to represent whether it is a built in Python function or user defined. 

Lastly, you can also see how the value _"Hello World!"_ is represented by the _tiny_py.constant_ operation and sits hierarchically inside the _tiny_py.call_expr_ operation. This is an example of nesting regions, where the _tiny_py.call_expr_ operation contains a region, which contains blocks (one in this case) which then contains this operation.

## Moving towards an executable

The _tiny_py_ dialect that we are using here is one that we have defined for this exercise. It's intentionally very simple and can be found at _src/dialects/tiny_py.py_ . We will modify this a bit later on in practical two, but first let's focus on the full compiler flow to get an executable out and run on the ARCHER2 supercomputer. We actually have a few options here, firstly we could write some transformations that directly generate LLVM-IR (or something else such as C) from this dialect. That would be OK, but potentially time consuming as one would need to explicitly write the transformations that convert this high level representation of a user's program into the very low level LLVM-IR. Instead, there is a much easier way and this is where MLIR comes in. We can transform this IR into standard MLIR dialects which themselves have transformations, written by the MLIR/LLVM community, to convert these into LLVM-IR. These transformations tend to be called _lowerings_, as we will initially lower from this very language (here Tiny Python) specific dialect into relatively lower level dialects, the standard MLIR dialects, and from there lower into the much lower level LLVM dialect. To this end we need to undertake a transformation pass on the IR that was generated above to undertake such a lowering. 

## The opt driver tool

LLVM and MLIR have adopted _-opt_ tools as a standard to be used to drive manipulation of IR, and xDSL also follows this naming convention. We have provided a corresponding tool for our tinypy compiler called _tinypy_opt_ , and this provides a convenient way in which to drive the parsing, printing, and manipulation from the command line. If you look in the current directory you will see a new file called _output.xdsl_, this is a text file that was created in the last step when we ran the _python3.10 ex_one.py_ command, and contains exactly the same output as was printed to the screen and we saw above. We will now operate on that file to lower it to the standard dialects, at the command line execute to following: 

```bash
user@login01:~$ ./tinypy-opt output.xdsl -p tiny-py-to-standard -t mlir
```

You will see that the following output will be displayed to screen:

```
"builtin.module"() ({
  "func.func"() ({
    %0 = "llvm.mlir.addressof"() {"global_name" = @str0} : () -> !llvm.ptr<!llvm.array<13 x i8>>
    %1 = "llvm.getelementptr"(%0) {"rawConstantIndices" = array<i32: 0, 0>} : (!llvm.ptr<!llvm.array<13 x i8>>) -> !llvm.ptr<i8>
    "func.call"(%1) {"callee" = @printf} : (!llvm.ptr<i8>) -> ()
    "func.return"() : () -> ()
  }) {"sym_name" = "main", "function_type" = () -> (), "sym_visibility" = "public"} : () -> ()
  "llvm.mlir.global"() ({
  }) {"global_type" = !llvm.array<13 x i8>, "sym_name" = "str0", "linkage" = #llvm.linkage<"internal">, "addr_space" = 0 : i32, "constant", "value" = "Hello world!\n", "unnamed_addr" = 0 : i64} : () -> ()
  "func.func"() ({
  }) {"sym_name" = "printf", "function_type" = (!llvm.ptr<i8>) -> (), "sym_visibility" = "private"} : () -> ()
}) : () -> ()
```

There are a few things going on here, so let's unpack it step by step. Firstly, we are providing the IR stored in the _output.xdsl_ file as an input to the _tinypy-opt_ tool. Using the _-p_ flag, we are instructing that the _tiny-py-to-standard_ pass should be run over this IR and transform it. Lastly, the _-t_ flag instructs the tool which technology to structure the output IR for, in this case we are selecting the MLIR format which can be fed directly into the MLIR tooling itself.

You can see that this IR looks quite different to the IR previously where it is in a much flatter form. This is known as Static Single-Assignment (SSA) form and is a common standard used across many compilers for structuring the IR. This goes back to the point we made in the lectures about progressive lowering, where we have taken something more structured (the _tiny_py_) and lowered it to a representation that is closer to the concrete implementation level. 

## Describing the output MLIR in more detail

The SSA IR that we have just generated is actually rather simple, even if it looks a little off putting at first. At the top level is a _builtin.module_, where all contents must reside within regions of a module. The first region contains the _func_ operation that is part of the _func_ dialect and used to define functions. A couple of lines down you can see that we are issuing the _call_ operation of the _func_ dialect, with the _callee_ (the function name) being _printf_. As an argument we pass the SSA value _%1_ to this as an argument. This is where this gets slightly more complex, you can see further down that we have the _mlir.global_ operation (which is part of the _llvm_ dialect) and this defines our string _"Hello World!"_ and calls it _str0_. The _mlir.addressof_ operation is retrieving this and then the _getelementptr_ operation looks up the pointer to the first element of the array which is then passed to the _printf_ call as _%1_. 

The _getelementptr_ operation is a good example to explain to give you a clearer idea of how operations are generally working. Here you can see that we are providing SSA value _%0_ as an argument (wich is the result of _mlir.addressof_) and of type _!llvm.ptr<!llvm.array<13 x i8>>_, which is an LLVM pointer to an array of thirteen 8 bit elements. The attribute _rawConstantIndices_ insructs the operation which element of this input it should return the pointer to, in this case element 0, and the result of this (referenced as _%1_ in the SSA) is _!llvm.ptr<i8>_ i.e. an LLVM pointer to an 8-bit value. From this explanation you should be able to see where the arguments to the operations are, where their types are specified, where the return type(s) if present are provided and how an operation's attributes are defined.

You can see right at the end of the IR we have another _func_ operation, this time for the _printf_ function but without any body. We need this because _printf_ is an external function that will only be brought in at link time. Therefore this _func_ operation without a routine will define an external function, providing MLIR with the type signature information.

## Generating the executable and running

The IR that we are generating from our tool is now is ready to be fed into MLIR and compiled into an executable. To do this we will execute _./tinypy-opt output.xdsl -p tiny-py-to-standard -t mlir -o ex_one-mlir_ . You can see that this command is identical to the previous one apart from the _-o_ argument, which informs the tool to store the generated IR in a file rather than output to screen. Next we execute:

```bash
user@login01:~$ mlir-opt --convert-func-to-llvm ex_one.mlir | mlir-translate -mlir-to-llvmir | clang -x ir -o test -
```

Here we are calling the _mlir-opt_ tool, which is part of MLIR, to undertake some conversion of the standard dialects into the LLVM dialect (here converting the _func_ dialect into its _llvm_ dialect counterpart), which is then fed into the _mlir-translate_ tool which generates LLVM-IR. This IR is then used as an input to Clang which builds the executable called _test_. If you are interested you can run the commands in isolation to see the different IRs that are generated from the commands. If you take a look in your directory you will see that there is now a new executable called _test_. 

### Running on ARCHER2

We can execute the _test_ executable direclty on the login node if we wish by (or if you are following the tutorial on your local machine):

```bash
user@login01:~$ ./test
```

But being a supercomputer it is also nice to run on the compute nodes too, and indeed we will be needing these in subsequent exercises. A submission script called _sub_ex1.srun_ is prepared that you can submit to the batch queue. 

```bash
user@login01:~$ sbatch sub_ex1.srun
```

You can check on the status of your job in the queue via _squeue -u $USER_ and once this has completed an output file will appear in your directly that contains the stdio output of the job. You can cat or less this file, which ever you prefer.


## Well done!

Well done - you have got your first end to end code compiling and running on a supercomputer via xDSL and MLIR. In the [next exercise](https://github.com/xdslproject/training-intro/edit/main/practical/two) we are going to enhance the dialects and transformations to extend the subset of Python that our tool supports to make things a bit more interesting and look at the different aspects in more depth.

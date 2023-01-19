# Practical One

## Getting started

Firstly you should connect to ARCHER2 via the login credentials supplied to you by the instructors, connecting to the machine via `ssh username@login.archer2.ac.uk`. Once you have connected you should then change into your _work_ directory, via issuing `cd /work/y14/y14/$USER`.

The guest accounts that you are using have been configured to contain the requires files for this practical and the correct environment settings.

## Step one

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

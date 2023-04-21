# A local environment

Whilst the intention of this tutorial is to undertake the practical elements on the ARCHER2 supercomputer, we also provide instructions so that people can undertake the practicals on a local machine too. We assume that you are using a Linux based system here, for other operating systems the instructions will be similiar but likely with some small variations.

## Prerequisites

In order to undertake the practicals on your local machine you will need to have the following installed:

| Package        | Minimal version          |
| ------------- |:-------------:| 
| cmake      | 3.13.4 | 
| gcc      | 7.1.0     | 
| python | 3.10     |  
| zlib | 1.2.3.4      |  
| gnu make | 3.79      |  

Optionally you can also install Ninja, which can be used instead of GNU make for building LLVM.

## Downloading and building LLVM

You first need to download and build LLVM, whilst this is fairly time consuming due to the size of LLVM it is a highly automated process. Our tutorials all leverage LLVM version 16, which at the time of writing is the latest version of LLVM. By executing the following you will obtain LLVM and switch to the version 16 release branch.

```bash
user@local:~$ git clone https://github.com/llvm/llvm-project.git
user@local:~$ git checkout --track origin/release/16.x
```

We will now create the build directory and issue cmake to configure the LLVM build, using GNU make to undertake the actual build and building clang, mlir, and openmp.

```bash
user@local:~$ cd llvm-project
user@local:~$ mkdir build
user@local:~$ cd build
user@local:~$ cmake  -G "Unix Makefiles"   -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON   -DCMAKE_CXX_LINK_FLAGS="-Wl,-rpath,$LD_LIBRARY_PATH"   -DFLANG_ENABLE_WERROR=ON   -DLLVM_ENABLE_ASSERTIONS=ON   -DLLVM_TARGETS_TO_BUILD=host   -DLLVM_LIT_ARGS=-v   -DLLVM_ENABLE_PROJECTS="clang;mlir;openmp"   -DLLVM_ENABLE_RUNTIMES="compiler-rt" ../llvm
```

>**Note**
> If you would rather use Ninja for the build you can substitute _"Unix Makefiles"_ with _Ninja_. To build additional LLVM components you can add these in the comma separated list of _LLVM_ENABLE_PROJECTS_ . You can explicitly specify the install directory via the _-DMAKE_INSTALL_PREFIX_ argument to cmake.

Once configuration has completed you can build LLVM by issuing:

```bash
user@local:~$ make install -j 12
```

This will build LLVM across 12 cores, you can change this number based upon the configuration of your system. Alternatively, you can issue `ninja -C $builddir install` to build via Ninja.

For more details about building LLVM or troubleshooting you can visit the (LLVM getting started page)[https://llvm.org/docs/GettingStarted.html]

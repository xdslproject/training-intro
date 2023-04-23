# xDSL introduction tutorial practicals

These are the practical exercises for the xDSL introduction tutorial, where we have the following directories

* [one](one) is where participants will obtain an overview of building the IR, compiling, and executing for a simple _Hello World_ Python example.
* [two](two) is where we get more in-depth into the details of the dialects and transformations as we add support for the Python _For_ construct, supporting loops in our simple Python compiler using the _scf.for_ operation.
* [three](three) is where we leverage threaded parallelism via OpenMP and vectorisation by transforming our for loop into an _scf.parallel_ operation and then use existihg MLIR transformations to lower to the _omp_ or _vector_ dialects.
* [src](src) contains the source code (dialect, transformations, and _tinypy_opt_ tool) that will be used throughout these exercises. If you are participating in one of our organised tutorials then this will all be preinstalled for you to use.
* [general](general) contains general instructions for accessing our machines and/or installing locally

Details about how to access your account on ARCHER2 and set up the environment that we will need for these practicals can be found on the [ARCHER2 setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/ARCHER2.md) page. If you are undertaking these tutorials locally then you can view the [local setup instructions](https://github.com/xdslproject/training-intro/blob/main/practical/general/local.md).

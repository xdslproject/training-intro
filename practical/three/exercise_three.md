# Exercise Three

In this practical we are going to explore transformations in more detail to add OpenMP parallelism and/or vectorisation to our loop in an automated manner.

Learning objectives are:

* Exploring the role of transformations and how these can manipulate the IR
* To understand how transformations are developed
* Gain an understanding of the key ways in which the IR can be traversed and manipulated
* Awareness of the _parallel_ operation in the _scf_ dialect
* To further demonstrate reusability benefits of MLIR transformations

Sample solutions to this exercise are provided in [sample_solutions](sample_solutions) in-case you get stuck or just want to compare your efforts with ours.

>**Having problems?**  
> As you go through this exercise if there is anything you are unsure about or are stuck on then please do not hesitate to ask one of the tutorial demonstrators and we will be happy to assist!

## The starting point and the plan

We are starting with the same code in practical two as illustrated below, however now we are going to write a transformation that will convert the resulting _for_ operation in the _scf_ dialect into a _parallel_ operation of that same dialect. 

```python
@python_compile
def ex_three():
    val=0.0
    add_val=88.2
    for a in range(0, 100000):
      val=val+add_val
    print(val)

ex_three()
```

The _parallel_ operation represents a parallel for loop, and there are existing MLIR transformations that will then parallelise this via OpenMP by lowering into the _omp_ dialect, apply vectorisation by lowering to the _vector_ dialect, or acclerate this via GPUs by lowering to the _gpu_ dialect. 



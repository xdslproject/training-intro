# Accessing ARCHER2

You have been provided with a visitor account on ARCHER2 for the duration of the tutorial. This has all the prerequisites installed and ready to be used. The username and password of your guest account will be provided to you by the tutorial team. 

## Logging into ARCHER2

In order to login to ARCHER2 you should execute the following, where _[user-id]_ is the username that has been allocated to you. You should then enter the password when prompted. 

```bash
user@login01:~$ ssh [user-id]@login.archer2.ac.uk
```

For more details on how to access ARCHER2, for instance if you are on a Windows or Mac machine then you can visit the [connecting guide](https://docs.archer2.ac.uk/user-guide/connecting/) for more information, or ask one of the demonstrators who will be happy to help.

## Setting up your environment

Once you have connected you should then change into your _work_ directory because this is the filesystem visible to the compute nodes and the area that we will be working in, you can do this via , via issuing the following:

```bash
user@login01:~$ cd /work/d011/d011/$USER
```

We have preprepared a series of modules which will provide us with LLVM 16, xDSL, and the correct Python version. You should therefore issue:

```bash
user@login01:~$ module use /work/d011/shared/modules
user@login01:~$ module load llvm
user@login01:~$ module load xdsl
```

This will set up all the required environment variables so that you can now execute LLVM binaries, link to LLVM libraries, and Python will find xDSL. 

>**Important**
> These environment setup steps (changing to your work directory and loading the llvm and xdsl modules) should be done for each ssh ARCHER2 session.

## Tutorial materials

This GitHub repository has already been cloned into your _work_ directory on ARCHER2, so if you _ls_ you should see the _training-intro_ directory that you can cd into, for instance to go into the practical directory and set up your environment specific to the tutorial:

```bash
user@login01:~$ cd training-intro/practical
user@login01:~$ source environment.sh
```

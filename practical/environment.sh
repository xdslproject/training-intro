#!/bin/bash

BASE_PATH=`pwd`

export PYTHONPATH=$BASE_PATH/src:$BASE_PATH/src/dialects:$BASE_PATH/src/utils:$PYTHONPATH
export PATH=$BASE_PATH/src/tools:$PATH
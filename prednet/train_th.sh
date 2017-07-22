#!/bin/bash

export LD_PRELOAD=/usr/local/cuda/lib64/libcudnn.so
export CUDA_ROOT=/home/hugdic/cuda
export CUDA_HOME=~/home/hugdic/cuda
export PATH=${CUDA_HOME}/bin/:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64/:$LD_LIBRARY_PATH
export CPATH=${CUDA_HOME}/include/:$CPATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH

KERAS_BACKEND=theano THEANO_FLAGS=device=gpu1,floatX=float32,nvcc.flags=-D_FORCE_INLINES ../anaconda2/bin/python vim2_theano.py
#/usr/lib64/nvidia:

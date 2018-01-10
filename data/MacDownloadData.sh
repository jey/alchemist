#!/usr/bin/env bash

curl -L "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2" -o "mnist.t.bz2"
bzip2 -d mnist.t.bz2


#!/usr/bin/env bash

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d mnist.scale.t.bz2
mv mnist.scale.t mnist.t

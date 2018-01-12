#!/bin/bash
# Assumes that spark-submit is in your PATH
# and the MNIST dataset has been downloaded using the script in the data directory

export NUM_ALCHEMIST_RANKS=3
export TMPDIR=/tmp

filepath=`pwd`/data/mnist.t
format=LIBSVM
numFeatures=10000
gamma=.001
numClass=10

spark-submit --verbose\
  --master local[*] \
  --driver-memory 2G\
  --class alchemist.test.regression.AlchemistRFMClassification\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $filepath $format $numFeatures $gamma $numClass
exit


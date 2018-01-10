#!/bin/bash
# Assumes that spark-submit is in your PATH
# and the MNIST dataset has been downloaded using the script in the data directory

export NUM_ALCHEMIST_RANKS=3
export TMPDIR=/tmp

filepath=`pwd`/data/mnist.t
numFeatures=1000
gamma=1.0

spark-submit --verbose\
  --master local[*] \
  --driver-memory 2G\
  --class amplab.alchemist.test.regression.AlchemistRFMClassification\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $filepath $numFeatures $gamma
exit


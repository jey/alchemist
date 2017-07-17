#!/usr/bin/env bash
# assume spark-submit is in path

export ALPREFIX=$HOME/Documents/alchemist/bins # or whatever you used during install
export TMPDIR=/tmp # avoid a Mac specific issue with tmpdir length

sbt -batch assembly

# user specified
PROJ_HOME="$HOME/Documents/alchemist"
MASTER="local[3]"
TARGET_DIM="30"

# the rest does not need change

# data
DATA_FILE="$PROJ_HOME/data/mnist.t"

# .jar file
JAR_FILE="$PROJ_HOME/test/target/scala-2.11/test-svd-assembly-0.0.2.jar"

spark-submit \
    --master $MASTER \
    $JAR_FILE $TARGET_DIM $DATA_FILE \
    > ResultTestSVD.out

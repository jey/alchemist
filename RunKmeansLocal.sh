#!/usr/bin/env bash

export ALPREFIX=$HOME/Documents/alchemist/bins # or whatever you used during install
export TMPDIR=/tmp # avoid a Mac specific issue with tmpdir length


sbt -batch assembly

# user specified
PROJ_HOME="$HOME/Documents/alchemist"
SPARK_HOME="$HOME/local/spark-2.1.1"
MASTER="local[8]"
CLUSTER_NUM="10"
MAX_ITERS="100"

# the rest does not need change

# data
export DATA_FILE="$PROJ_HOME/data/mnist.t"

# save clustering results
OUTPUT_DIR="$PROJ_HOME/.tmp"
mkdir $OUTPUT_DIR
export OUTPUT_FILE="$OUTPUT_DIR/kmeans_result"

# .jar file
JAR_FILE="$PROJ_HOME/test/target/scala-2.11/test-kmeans-assembly-0.0.2.jar"

$SPARK_HOME/bin/spark-submit \
    --master $MASTER \
    $JAR_FILE $CLUSTER_NUM $MAX_ITERS \
    > ResultTestKmeans.out


# Compute NMI using a Python script
MNI_PY="$PROJ_HOME/test/src/main/scala/amplab/alchemist/nmi.py"
    
python $MNI_PY -f $OUTPUT_FILE"_spark.txt" \
    >> ResultTestKmeans.out
    
python $MNI_PY -f $OUTPUT_FILE"_alchemist.txt" \
    >> ResultTestKmeans.out
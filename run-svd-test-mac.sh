#!/bin/bash
# Assumes that spark-submit is in your PATH

export NUM_ALCHEMIST_WORKERS=5
export TMPDIR=/tmp

method=SVD
# 2.5M by 10K double matrix is 200 GB
# m=5000000
#m=10000000
m=250000 
n=1000
k=20

# seems like if the partitions are too large, Spark will hang, so go for 2GB/partition
# 0 tells Spark to use default parallelism
#partitions=200
partitions=0

spark-submit --verbose\
  --master local[*] \
  --driver-memory 1G\
  --executor-memory 2G\
  --num-executors 2 \
  --class amplab.alchemist.BasicSuite\
  --conf spark.memory.fraction=0.8 \
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test.log
exit

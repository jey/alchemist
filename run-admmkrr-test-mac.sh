#!/bin/bash
# Assumes that spark-submit is in your PATH

export NUM_ALCHEMIST_RANKS=4
export TMPDIR=/tmp

method=ADMMKRR
# 2.5M by 10K double matrix is 200 GB
# m=5000000
#m=10000000
n=125000 
p=1000
m=4

# seems like if the partitions are too large, Spark will hang, so go for 2GB/partition
# 0 tells Spark to use default parallelism
#partitions=200
partitions=0

spark-submit --verbose\
  --master local[*] \
  --driver-memory 3G\
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $n $p $m $partitions 2>&1 | tee test.log
exit

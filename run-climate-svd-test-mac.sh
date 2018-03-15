#!/bin/bash
# Assumes that spark-submit is in your PATH

export NUM_ALCHEMIST_RANKS=5
export TMPDIR=/tmp

# testclimatesvd.h5 has 
k=100
fname=testclimatesvd.h5
useAlc=1
loadAlc=1
varname=/rows
colreplicas=2

spark-submit --verbose\
  --master local[3]\
  --driver-memory 4G\
  --class org.apache.spark.mllib.linalg.distributed.ClimateSVD\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $k $fname $useAlc $loadAlc $varname $colreplicas 2>&1 | tee test.log
exit

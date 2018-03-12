#!/bin/bash

#SBATCH -p debug
#SBATCH -N 14
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

source setup/cori-start-alchemist.sh 12 2
sleep 15

# 6177583 by 8096 => 400 GB dataset
# need about 3x times memory to store, relayout the matrix to do the GEMM needed in the alchemist SVD vs just store the matrix
k=200
fname=/global/cscratch1/sd/gittens/large-datasets/ocean.h5
useAlc=1
loadAlc=1
varname=/rows
colreplicas=2

spark-submit --verbose\
  --driver-memory 120G\
  --executor-memory 120G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 1 \
  --conf spark.driver.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.executor.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class org.apache.spark.mllib.linalg.distributed.ClimateSVD\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $k $fname $useAlc $loadAlc $varname $colreplicas 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

#!/bin/bash

#SBATCH -p debug
#SBATCH -N 20
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

export LD_LIBRARY_PATH=$LD_LBRARY_PATH:$PWD/lib

module unload darshan
source setup/cori-start-alchemist.sh 18 2

# 6177583 by 8096 => 400 GB dataset
k=20
fname=/global/cscratch1/sd/gittens/large-datasets/ocean.h5
varname=/rows

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
  --class amplab.alchemist.ClimateSVD\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $k $fname $varname 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

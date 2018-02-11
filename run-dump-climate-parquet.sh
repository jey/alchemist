#!/bin/bash

#SBATCH -p debug
#SBATCH -N 35
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

export LD_LIBRARY_PATH=$LD_LBRARY_PATH:$PWD/lib

module unload darshan
source setup/cori-start-alchemist.sh 24 2

spark-submit --verbose\
  --driver-memory 120G\
  --executor-memory 120G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 10 \
  --conf spark.driver.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.executor.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class amplab.alchemist.dumpClimateToParquet\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

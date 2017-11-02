#!/bin/bash

#SBATCH -p debug
#SBATCH -N 6
#SBATCH -t 00:15:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module unload spark/hist-server
#module load collectl
#start-collectl.sh 

source setup/cori-start-alchemist.sh

export LD_LIBRARY_PATH=$LD_LBRARY_PATH:$PWD/lib

spark-submit --verbose\
  --master $SPARKURL\
  --name $app_name \
  --driver-memory 100G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors=2 \
  --executor-memory 105G\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar 2>&1 | tee test.log

stop-all.sh
#stop-collectl.sh

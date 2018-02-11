#!/bin/bash

#SBATCH -p debug
#SBATCH -N 15
#SBATCH -t 01:00:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

# How to choose number of nodes (based on memory pressure)
#
# For Spark: idk how to choose the number of executors. Take more than you should need, i guess.
# Spark is horrible w.r.t. using memory efficiently.
#
# For Alchemist: note that each process can access roughly 16GB chunk of each matrix 
# (from my experiments, trying to pack more on a single process causes std::alloc array length failures)
# so choose number of processes and hence number of nodes accordingly, based on how much memory you need
# for the data, any relayouts that may be needed, etc.
#
# NB: you should ensure the data isn't concentrated on just
# a few spark nodes, otherwise the communication to Alchemist will be slow

module unload darshan
# x y means start x machines with y cores per process
source setup/cori-start-alchemist.sh 10 2

filepath=/global/cscratch1/sd/wss/data_timit/timit-train.csv
format=CSV
numFeatures=10000
gamma=.001
numClass=147

spark-submit --verbose\
  --driver-memory 124G\
  --executor-memory 124G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 4 \
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class alchemist.test.regression.AlchemistADMMTest\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $filepath $format $numFeatures $gamma $numClass 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

#!/bin/bash

#SBATCH -p debug
#SBATCH -N 21
#SBATCH -t 00:35:00
#SBATCH -J matmul
#SBATCH --mail-user=gittea@rpi.edu
#SBATCH --mail-type=ALL
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

module unload darshan
source setup/cori-start-alchemist.sh 10 2

method=MATMUL
# 10K by 10K double matrix is 800 MB
# 300K by 10K times 10K by 60K results in 144 GB matrix (5 alchemist nodes didn't work, oom, 10 does)
m=300000
n=10000
k=60000

# seems like if the partitions are too large, Spark will hang, so go for 2GB/partition
# 0 tells Spark to use default parallelism
#partitions=200
partitions=0

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
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test-matmul.log

stop-all.sh
exit
#stop-collectl.sh

#!/bin/bash

#SBATCH -q premium
#SBATCH -N 278
#SBATCH -t 00:07:00
#SBATCH -J eight-replicated
#SBATCH --mail-user=gittea@rpi.edu
#SBATCH --mail-type=ALL
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

source setup/cori-start-alchemist.sh 276 2
sleep 15

# 2.2TB dataset => (18 machines to hold in memory one copy of this dataset)
# need about 2.24x times the memory necessary for storage (of the replicated matrix) to load and work with the matrix 
k=20
fname=/global/cscratch1/sd/gittens/large-datasets/rda_ds093.0_dataset/outputs/ocean2T.h5
useAlc=1
loadAlc=1
varname=/rows
colreplicas=8

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

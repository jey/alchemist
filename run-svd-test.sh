#!/bin/bash

#SBATCH -p debug
#SBATCH -N 43
#SBATCH -t 00:30:00
#SBATCH -e mysparkjob_%j.err
#SBATCH -o mysparkjob_%j.out
#SBATCH -C haswell
#module load collectl
#start-collectl.sh 

# NB on choosing parameters: from
# echo | gcc -E -xc -include 'stddef.h' - | grep size_t
# the max array size on Cori is 2^32, which means can only
# have each chunk of the Elemental matrices be of size ~<4GB
# Example calculation: I want to use as few nodes as possible to hold a 400GB dataset
# say I give 4 cores per alchemist process, then 8 alchemist processes fit on one Cori node
# then to get less than 4GB/process, that means I'll need 100 processes at least, so 
# will need 100/8 = 13 nodes. Note this is very memory inefficient, since will only be using
# 32 GB per node. 
# Alternatively, doing flat MPI, can fit 32 processes per node. To get less than 4GB/process,
# need at least 100 processes, so need at least 100/32 = 4 nodes. 
# To give Alchemist some memory room for intermediate operations, lets double the node count to 8 (and double the number of 
# cores to 2 per process to ensure we actually get more memory per process)
#
# As for Spark, idk how to choose the number of executors. Take more than you should need, i guess.
# Spark is horrible w.r.t. using memory efficiently.

module unload darshan
source setup/cori-start-alchemist.sh 30 2

method=SVD
# 2.5M by 10K double matrix is 200 GB
# m=5000000
m=10000000
#m=2500000
n=10000
k=20
# seems like if the partitions are too large, Spark will hang, so go for 2GB/partition
partitions=200

spark-submit --verbose\
  --driver-memory 120G\
  --executor-memory 120G\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 12 \
  --conf spark.driver.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.executor.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

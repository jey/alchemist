#!/bin/bash

#SBATCH -p debug
#SBATCH -N 30
#SBATCH -t 00:15:00
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
source setup/cori-start-alchemist.sh 8 2

method=SVD
m=5000000
n=10000
k=100
partitions=200

spark-submit --verbose\
  --jars /global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/fits.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarfitsobj.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarh4obj.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarh5obj.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarhdf-2.11.0.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarhdf5-2.11.0.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarhdfobj.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jarnc2obj.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/jhdfview.jar,/global/cscratch1/sd/gittens/alchemistSHELL/alchemist/lib/netcdfAll-4.6.5.jar\
  --driver-memory 115g\
  --executor-memory 115g\
  --executor-cores 32 \
  --driver-cores 32  \
  --num-executors 21 \
  --conf spark.driver.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.executor.extraLibraryPath=$SCRATCH/alchemistSHELL/alchemist/lib\
  --conf spark.eventLog.enabled=true\
  --conf spark.eventLog.dir=$SCRATCH/spark/event_logs\
  --class amplab.alchemist.BasicSuite\
  test/target/scala-2.11/alchemist-tests-assembly-0.0.2.jar $method $m $n $k $partitions 2>&1 | tee test.log

stop-all.sh
exit
#stop-collectl.sh

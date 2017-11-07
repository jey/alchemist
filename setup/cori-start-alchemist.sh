# This script is for running Alchemist on a Cori allocation
# It loads the spark module and splits the nodes between Alchemist and Spark 
# It is to be run immediately after you have launched a batch/interactive job on Cori
# It also loads the modules needed to run/compile Alchemist and sets the appropriate paths and variables
# (just in case you need to compile during an allocation)
#
# the first argument is the number of nodes to assign to Alchemist: it should be at least two (for a driver and one worker)
# the second argument decides how many cores per alchemist worker (and driver), and therefore how many alchemist workers per node
#
# Example use: launch 15 Spark nodes (14 executors) and 5*(32/16) - 1 = 9 Alchemist workers (on 5 nodes)
# salloc -N 20 -t 15 --qos=interactive -C haswell
# source setup/cori-start-alchemist.sh 5 16

###############################################
##### Setup building environment ##############
###############################################

# need to update this variable depending on where you installed alchemist
export ALPREFIX=$SCRATCH/alchemistSHELL/bins

module unload PrgEnv-intel
module load PrgEnv-gnu
module load gcc
module load java
module load python
module load boost
module load cmake
module load sbt

# the library paths probably don't need to be set, as alchemist is build with everything in rpath
export PATH=$ALPREFIX/bin:$PATH
export CPATH=$ALPREFIX/include:$CPATH
export LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
# probably don't need the next two lines
export LIBRARY_PATH=$BOOST_DIR/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin/:$LD_LIBRARY_PATH
export CC="cc"
export CXX="CC"
export FC="ftn"

###############################################
##### Start Spark and Alchemist  ##############
###############################################

# slaves is automatrically generated when the spark module is loaded, and contains 
# the machines on which the spark executors will run (so one less than the number of machines allocated in this Cori batch job)
# split this into two machine files: one for Spark, one for Alchemist

module load spark/2.1.0

ALCHEMISTNODECOUNT=$1 
[[ $SPARKURL =~ spark://(.*):(.*) ]]
export SPARK_MASTER_NODE=${BASH_REMATCH[1]}
mv $SPARK_WORKER_DIR/slaves $SPARK_WORKER_DIR/slaves.original
cat $SPARK_WORKER_DIR/slaves.original | sed -n "1,${ALCHEMISTNODECOUNT}p" > $SPARK_WORKER_DIR/hosts.alchemist
cat $SPARK_WORKER_DIR/slaves.original | sed -n "$((${ALCHEMISTNODECOUNT}+1)),\$p" > $SPARK_WORKER_DIR/slaves
mkdir -p $SPARK_WORKER_DIR/alchemistIOs

# start spark 
start-all.sh

# apparently has to be done *AFTER* spark launches, or spark workers won't launch?
# start alchemist (can't do this from spark driver as would on a laptop, b/c it needs to be run from the mom node)
# BE CAREFUL: SRUN NEEDS AN ABSOLUTE PATH TO THE EXECUTABLE
export OMP_NUM_THREADS=$2
srun -N ${ALCHEMISTNODECOUNT}\
     -n $((${ALCHEMISTNODECOUNT}*32/${OMP_NUM_THREADS}))\
     --cpus-per-task=${OMP_NUM_THREADS}\
     -w $SPARK_WORKER_DIR/hosts.alchemist\
     --output=$SPARK_WORKER_DIR/alchemistIOs/stdout_%t.log\
     --error=$SPARK_WORKER_DIR/alchemistIOs/stderr_%t.log\
     ./core/target/alchemist &

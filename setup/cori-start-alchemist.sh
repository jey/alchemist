# assumes there are an even number of nodes launched
# to be run once you have launched a batch/interactive job on Cori:
# loads the spark module and splits the nodes between Alchemist and Spark evenly,
# loads needed modules to run/compile Alchemist, and sets appropriate paths and variables
# to run/compile Alchemist

# need to update this variable depending on where you installed alchemist
export ALPREFIX=$SCRATCH/alchemistSHELL/bins

module load spark/2.1.0
[[ $SPARKURL =~ spark://(.*):(.*) ]]
export SPARK_MASTER_NODE=${BASH_REMATCH[1]}

pushd /tmp
HALFLINECOUNT=$((`wc -l $SPARK_WORKER_DIR/slaves | cut -f 1 -d' '`/2))
# slaves is automatrically generated when the spark module is loaded, and contains 
# the machines on which spark will run (so the machines allocated in this Cori batch job)
# note slaves has an odd number of lines (because it does not include the spark master node)
# one of these will be the alchemist master, and the remainder are split evenly over spark and alchemist as executors
# bash rounds down so 3/2=1
mv $SPARK_WORKER_DIR/slaves $SPARK_WORKER_DIR/slaves.original
cat $SPARK_WORKER_DIR/slaves.original | sed -n "$((${HALFLINECOUNT}+1)),\$p" > $SPARK_WORKER_DIR/hosts.alchemist
cat $SPARK_WORKER_DIR/slaves.original | sed -n "1,${HALFLINECOUNT}p" > $SPARK_WORKER_DIR/slaves
popd
# start spark on half the nodes
start-all.sh
# start alchemist on the other half (can't do this from spark driver b/c it needs to be run from the mom node)
NUMALPROCS=`wc -l $SPARK_WORKER_DIR/hosts.alchemist`
srun -N $NUMALPROCS -n $NUMALPROCS -w $SPARK_WORKER_DIR/hosts.alchemist core/target/alchemist &

module unload PrgEnv-intel
module load PrgEnv-gnu
module load gcc
module load java
module load python
module load boost
module load cmake
module load sbt

export PATH=$ALPREFIX/bin:$PATH
export CPATH=$ALPREFIX/include:$CPATH
export LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
export CC="cc"
export CXX="CC"
export FC="ftn"

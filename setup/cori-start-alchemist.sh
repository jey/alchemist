# assumes there are an even number of nodes launched
# to be run once you have launched a batch/interactive job on Cori:
# loads the spark module and splits the nodes between Alchemist and Spark evenly,
# loads needed modules to run/compile Alchemist, and sets appropriate paths and variables
# to run/compile Alchemist

module load spark/2.1.0
[[ $SPARKURL =~ spark://(.*):(.*) ]]
export SPARK_MASTER_NODE=${BASH_REMATCH[1]}

pushd /tmp
HALFLINECOUNT=$((`wc -l $SPARK_WORKER_DIR/slaves | cut -f 1 -d' '`/2))
# note slaves has an odd number of lines (does not include the spark master node)
# bash rounds down so 3/2=1
mv $SPARK_WORKER_DIR/slaves $SPARK_WORKER_DIR/slaves.original
cat $SPARK_WORKER_DIR/slaves.original | sed -n "$((${HALFLINECOUNT}+1)),\$p" > $SPARK_WORKER_DIR/hosts.alchemist
cat $SPARK_WORKER_DIR/slaves.original | sed -n "1,${HALFLINECOUNT}p" > $SPARK_WORKER_DIR/slaves
popd
# start spark on half the nodes
start-all.sh
# start alchemist on the other half (can't do this from spark driver b/c it needs to be run from the mom node)

module unload PrgEnv-intel
module load PrgEnv-gnu
module load gcc
module load java
module load python
module load boost
module load cmake
module load sbt

export ALPREFIX=$SCRATCH/alchemistSHELL/bins
export PATH=$ALPREFIX/bin:$PATH
export CPATH=$ALPREFIX/include:$CPATH
export LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$ALPREFIX/lib64:$ALPREFIX/lib:$LIBRARY_PATH
export CC="cc"
export CXX="CC"
export FC="ftn"

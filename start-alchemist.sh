module load spark/2.1.0
[[ $SPARKURL =~ spark://(.*):(.*) ]]
export SPARK_MASTER_NODE=${BASH_REMATCH[1]}

pushd /tmp
split -n 2 $SPARK_WORKER_DIR/slaves
mv xaa $SPARK_WORKER_DIR/slaves
mv xab $SPARK_WORKER_DIR/hosts.alchemist
popd
start-all.sh

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

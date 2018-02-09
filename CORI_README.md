# To run Alchemist on Cori
```
# launch in interactive mode w/ e.g.
# salloc -N 7 -t 30 -C haswell --qos=interactive -L SCRATCH
cd $SCRATCH/alchemistSHELL/alchemist # or wherever you installed alchemist as instructed below
# this will allocate 3 nodes for alchemist processes, and each alchemist process will be given 2 cores; 
# the remaining 4 nodes will run Spark; see the script for more details
source setup/cori-start-alchemist.sh 3 2
make check # will run the current test suite
```

# Installation instructions (for Cori: can run on a login node or inside a queue job)

## Clone the Alchemist repo and set the ALPREFIX environment variable to where supporting libraries are installed
```
bash # need to use bash
mkdir -p $SCRATCH/alchemistbase
cd $SCRATCH/alchemistbase
git clone https://github.com/alexgittens/alchemist.git
git checkout cori-version 
source ./alchemist/setup/cori-bootstrap.sh

module unload PrgEnv-intel
module load PrgEnv-gnu gcc java python boost sbt cray-hdf5 fftw
export ALPREFIX=$SCRATCH/alchemistbase/bins

cd alchemist
make build
```

# To test
Needs to be made less manual and more in line with standard practices, e.g., see the spark-perf project


# To run Alchemist on Cori
```
# launch in interactive mode w/ e.g.
# salloc -N 7 -t 30 -C haswell --qos=interactive -L SCRATCH
cd $SCRATCH/alchemistSHELL/alchemist # or wherever you installed alchemist as instructed below
# this will allocate 3 nodes for alchemist processes, and each alchemist process will be given 2 cores; 
# the remaining 4 nodes will run Spark; see the script for more details
source setup/cori-start-alchemist.sh 3 2
# now run one of the test scripts
```

# Installation instructions (for Cori: can run on a login node or inside a queue job)

## Clone the Alchemist repo and set the ALPREFIX environment variable to where supporting libraries are installed
```
bash # need to use bash
mkdir -p $SCRATCH/alchemistbase
cd $SCRATCH/alchemistbase
git clone https://github.com/alexgittens/alchemist.git
source ./alchemist/setup/cori-bootstrap.sh

module unload darshan
module unload PrgEnv-intel
module load PrgEnv-gnu gcc java python boost sbt hdf5-parallel fftw
export ALPREFIX=$SCRATCH/alchemistSHELL/bins

cd alchemist
make build

now edit the cori-start-alchemist.sh start script to ensure that ALPREFIX is set correctly
```

# To test
Look through the test src files as needed to see how to call them, or use one of the test scripts. Caveat the testing code is messy.
Needs to be made less manual and more in line with standard practices, e.g., see the spark-perf project


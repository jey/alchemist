# To run Alchemist on Cori
```
# launch in interactive mode w/ e.g.
# salloc -N 6 -t 30 -C haswell --qos=interactive -L SCRATCH
# to get 3 Spark nodes (1 driver, 2 exec) and 3 Alchemist nodes (1 driver, 2 exec)
cd $SCRATCH/alchemistSHELL/alchemist # or wherever you installed alchemist as instructed below
source setup/cori-start-alchemist.sh # will split nodes between Spark and Alchemist, and start Alchemist in background
make check # will run the current test suite
```

# Installation instructions (for Cori: can run on a login node or inside a queue job)

## Clone the Alchemist repo and set the ALPREFIX environment variable to where supporting libraries are installed
```
bash # need to use bash
mkdir -p $SCRATCH/alchemistSHELL
cd $SCRATCH/alchemistSHELL
git clone https://github.com/alexgittens/alchemist.git
git checkout cori-version 
source ./alchemist/setup/cori-bootstrap.sh

module unload PrgEnv-intel
module load PrgEnv-gnu gcc java cmake python boost sbt
export ALPREFIX=$SCRATCH/alchemistSHELL/bins

cd alchemist
make build
```

# To test
Needs to be made less manual and more in line with standard practices, e.g., see the spark-perf project


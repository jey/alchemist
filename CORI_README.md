# To run Alchemist on Cori
```
launch in interactive mode
cd $SCRATCH/alchemistBase # or wherever you installed it
export ALPREFIX=$SCRATCH/alchemistBase/bins # or whatever you used during install
make check # will both build and run the test suite
```

# Installation instructions (for Cori: can run on a login node)

## Clone the Alchemist repo and set the ALPREFIX environment variable where supporting libraries are installed
```
bash # need to use bash
mkdir -p $SCRATCH/alchemistBase
cd $SCRATCH/alchemistBase
git clone https://github.com/alexgittens/alchemist.git
source ./alchemist/setup/cori-bootstrap.sh

module unload PrgEnv-intel
module load PrgEnv-gnu gcc java cmake python boost sbt
export ALPREFIX=$SCRATCH/alchemistBase/bins

cd alchemist
make build
```

# To test
Needs to be made less manual and more in line with standard practices, e.g., see the spark-perf project


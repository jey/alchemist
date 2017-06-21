Alchemist is a framework for easily and efficiently calling MPI-based codes from Apache Spark.

Supporting libraries that Alchemist uses:
Elemental  --- used for distributing the matrices b/w Alchemist processes, distributed linear algebra
Eigen3 -- used for local matrix manipulations (more convenient interface than Elemental)
Arpack-ng -- for the computation of truncated SVDs
Arpackpp -- very convenient C++ interface to Arpack-ng

# To run Alchemist in a fresh terminal:
```
cd $HOME/Documents/alchemist # or whereever you installed it
export ALPREFIX=$HOME/Documents/alchemist/bins # or whatever you used during install
export PATH=$PATH:$HOME/local/spark-2.1.1/bin # or whereever spark-bin is located
make # will both build and run the test suite
```

# Installation instructions (for running locally on MacOS 10.12)

## Install some prereqs
Assuming that the XCode command line tools, Homebrew, and Spark have been installed:
```
brew install gcc
brew install cmake
brew install boost-mpi
brew install sbt
```

## Clone the Alchemist repo and set the ALPREFIX environment variable where supporting libraries will be installed
```
cd Documents # (if you want to install alchemist in $HOME/Documents/alchemist)
git clone https://github.com/alexgittens/alchemist.git
cd alchemist 
export ALPREFIX=$HOME/Documents/alchemist/bins
```

## Install Elemental into ALPREFIX
```
git clone https://github.com/elemental/Elemental.git
cd Elemental
git checkout 0.87
mkdir build
cd build
CC=gcc-7 CXX=g++-7 FC=gfortran-7 cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
nice make -j8
make install
cd ../..
rm -rf Elemental
```

## Install Eigen3 into ALPREFIX 
```
curl -L -O http://bitbucket.org/eigen/eigen/get/3.3.4.zip
unzip 3.3.4.zip
rm 3.3.4.zip
cd eigen-eigen-5a0156e40feb # or whatever the tag is
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
make install
cd ../..
rm -rf eigen-eigen-5a0156e40feb
```

## Install Arpack-ng into ALPREFIX 
```
git clone https://github.com/opencollab/arpack-ng.git
cd arpack-ng
mkdir build
cd build
CC=gcc-7 FC=gfortran-7 cmake -DMPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
nice make -j8
make install
cd ../..
rm -rf arpack-ng
```

## Install Arpackpp into ALPREFIX 
```
git clone https://github.com/m-reuter/arpackpp.git
cd arpackpp
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
make install
cd ../..
rm -rf arpackpp
```



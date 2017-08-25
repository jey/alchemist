#!/bin/bash
set -o verbose
set -o errexit

# How to use this script:
#
#   export ALPREFIX = $HOME/Documents/alchemistBase (or wherever you want to install it)
#   mkdir -p $ALPREFIX
#   cd $ALPREFIX
#   git clone https://github.com/alexgittens/alchemist.git
#   bash ./alchemist/setup/macos-bootstrap.sh
#
#   assumes the following prereqs have been installed
#

ALROOT=$PWD
ALPREFIX=$ALROOT/bins

WITH_PREREQS=0
WITH_EL=1
WITH_ARPACK=1
WITH_ARPACKPP=1
WITH_EIGEN=1
WITH_SPDLOG=1

# install prereqs if not already there

if [ $WITH_PREREQS = 1 ]; then
  brew install gcc
  brew install make --with-default-names
  brew install cmake
  brew install boost-mpi
  brew install sbt
  brew install gmp
fi

# Start download auxiliary packages

mkdir -p dl
cd dl

# Elemental
if [ "$WITH_EL" = 1 ]; then
  git clone https://github.com/elemental/Elemental
  cd Elemental
  git checkout v0.87.7
  mkdir build
  cd build
  CC=gcc-7 CXX=g++-7 FC=gfortran-7 cmake -DCMAKE_BUILD_TYPE=Release -DEL_IGNORE_OSX_GCC_ALIGNMENT_PROBLEM=ON -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  nice make -j8
  make install
  cd ../..
fi

# arpack-ng
if [ "$WITH_ARPACK" = 1 ]; then
  git clone https://github.com/opencollab/arpack-ng
  cd arpack-ng
  git checkout 3.5.0
  mkdir build
  cd build
  CC=gcc-7 FC=gfortran-7 cmake -DMPI=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  nice make -j8
  make install
  cd ../..
fi

# arpackpp
if [ "$WITH_ARPACKPP" = 1 ]; then
  git clone https://github.com/m-reuter/arpackpp
  cd arpackpp
  git checkout 88085d99c7cd64f71830dde3855d73673a5e872b
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  make install
  cd ../..
fi

# Eigen
if [ "$WITH_EIGEN" = 1 ]; then
  curl -L http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 | tar xvfj -
  cd eigen-eigen-5a0156e40feb
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  nice make -j8
  make install
  cd ../..
fi

# SPDLog
if [ "$WITH_SPDLOG" = 1 ]; then
  git clone https://github.com/gabime/spdlog.git
  cd spdlog
  git checkout 4fba14c79f356ae48d6141c561bf9fd7ba33fabd
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  make install -j8
  cd ../..
fi

#!/bin/bash
set -o verbose
set -o errexit

# How to use this script:
#
#   ssh cori
#   cd $SCRATCH
#   mkdir alchemist
#   cd alchemist
#   git clone https://github.com/jey/alchemist
#   ./alchemist/cori/bootstrap.sh
#

ALROOT=$PWD
ALPREFIX=$ALROOT/bins

WITH_GMP=1
WITH_EL=1
WITH_ARPACK=1
WITH_ARPACKPP=1
WITH_EIGEN=1

# Check that the cmake toolchain file is where we expect
[ -f $ALROOT/alchemist/cori/Cori-gnu.cmake ]

# Setup
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

mkdir -p $ALPREFIX
mkdir -p dl
cd dl

# GMP
if [ "$WITH_GMP" = 1 ]; then
  curl -L https://ftp.gnu.org/gnu/gmp/gmp-6.1.2.tar.bz2 > gmp-6.1.2.tar.bz2
  tar xfvj gmp-6.1.2.tar.bz2
  cd gmp-6.1.2
  ./configure --prefix=$ALPREFIX --enable-cxx --disable-static
  nice make -j16
  make install
  cd ..
fi

# Elemental
if [ "$WITH_EL" = 1 ]; then
  git clone https://github.com/elemental/Elemental
  cd Elemental
  git checkout v0.87.7
  mkdir build
  cd build
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$ALROOT/alchemist/cori/Cori-gnu.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j16
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
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$ALROOT/alchemist/cori/Cori-gnu.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j16
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
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$ALROOT/alchemist/cori/Cori-gnu.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j16
  make install
  cd ../..
fi

# Eigen
if [ "$WITH_EIGEN" = 1 ]; then
  curl -L http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 | tar xvfj -
  cd eigen-eigen-5a0156e40feb
  mkdir build
  cd build
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$ALROOT/alchemist/cori/Cori-gnu.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j16
  make install
  cd ../..
fi

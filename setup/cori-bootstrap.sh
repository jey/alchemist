#!/bin/bash
set -o verbose
#set -o errexit

# This script installs the prerequisites for building Alchemist on Cori
#
# How to use this script:
#
#   ssh cori
#   cd $SCRATCH
#   mkdir alchemistBase
#   cd alchemistBase
#   git clone https://github.com/alexgittens/alchemist.git
#   bash ./alchemist/setup/cori-bootstrap.sh
#

ALROOT=$PWD
ALPREFIX=$ALROOT/bins

MAKE_THREADS=16

# Set the following flags to indicate what needs to be installed

WITH_GMP=1
WITH_EL=1
WITH_RANDOM123=1
WITH_SKYLARK=1
WITH_ARPACK=1
WITH_ARPACKPP=1
WITH_EIGEN=1
WITH_SPDLOG=1

# Check that the cmake toolchain file is where we expect
TOOLCHAIN=$ALROOT/alchemist/setup/Cori-gnu.cmake
[ -f "$TOOLCHAIN" ]

# Setup
module unload PrgEnv-intel
module load PrgEnv-gnu
module load gcc
module load java
module load python
module load boost
module load cmake
module load sbt
module load fftw
module load cray-hdf5
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
  nice make -j"$MAKE_THREADS"
  make install
  cd ..
fi

# Elemental
if [ "$WITH_EL" = 1 ]; then
  git clone https://github.com/elemental/Elemental
  cd Elemental
  git checkout tags/v0.87.4
  mkdir build
  cd build
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j"$MAKE_THREADS"
  make install
  cd ../..
fi

# Random123
if [ "$WITH_RANDOM123" = 1 ]; then
  wget http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz
  tar xvfz Random123-1.08.tar.gz
  cp -r Random123-1.08/include/Random123 $ALPREFIX/include
fi

# Skylark
if [ "$WITH_SKYLARK" = 1 ]; then
  git clone https://github.com/xdata-skylark/libskylark.git
  cd libskylark
  mkdir build
  cd build
  export ELEMENTAL_ROOT="$ALPREFIX"
  export RANDOM123_ROOT="$ALPREFIX"
  # HDF5_ROOT is set when cray-hdf5 module is loaded
  CXXFLAGS="-dynamic -std=c++14 -fext-numeric-literals" cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DUSE_HYBRID=OFF \
    -DUSE_FFTW=ON \
    -DBUILD_PYTHON=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_EXAMPLES=ON ..
  nice make -j"$MAKE_THREADS"
  make install
  cd ../..

  # for some reason, this didn't seem to install, so manually copy it over
  # recheck in future and remove if unnecessary
  cp -r libskylark/utility/fft $ALPREFIX/include/skylark/utility
fi

# # Skylark
# # need to use development-v0.30 branch and patch it
# if [ "$WITH_SKYLARK" = 1 ]; then
#   git clone https://github.com/xdata-skylark/libskylark.git
#   cd libskylark
#   git checkout development-v0.30
#   git apply $ALROOT/alchemist/setup/crlsc.patch
#   mkdir build
#   cd build
#   export ELEMENTAL_ROOT="$ALPREFIX"
#   export RANDOM123_ROOT="$ALPREFIX"
#   CXXFLAGS="-dynamic -std=c++14 -fext-numeric-literals" cmake \
#     -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
#     -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
#     -DCMAKE_BUILD_TYPE=RELEASE \
#     -DUSE_HYBRID=OFF \
#     -DUSE_FFTW=ON \
#     -DBUILD_PYTHON=OFF \
#     -DBUILD_SHARED_LIBS=ON \
#     -DBUILD_EXAMPLES=ON ..
#   nice make -j"$MAKE_THREADS"
#   make install
#   cd ../..
# fi

# arpack-ng
if [ "$WITH_ARPACK" = 1 ]; then
  git clone https://github.com/opencollab/arpack-ng
  cd arpack-ng
  git checkout 3.5.0
  mkdir build
  cd build
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j"$MAKE_THREADS"
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
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j"$MAKE_THREADS"
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
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j"$MAKE_THREADS"
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
  cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-dynamic" \
    -DCMAKE_CXX_FLAGS="-dynamic" \
    -DCMAKE_Fortran_FLAGS="-dynamic" \
    ..
  nice make -j"$MAKE_THREADS"
  make install
  cd ../..
fi

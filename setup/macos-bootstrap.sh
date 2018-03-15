#!/bin/bash
set -o verbose
set -o errexit

# This script installs the prerequisites for building Alchemist on MacOS 
# (tested on Sierra)

# How to use this script:
#
#   go to the directory that will serve as the alchemist root directory
#   git clone https://github.com/alexgittens/alchemist.git
#   # MAKE SURE the correct WITH_* flags are set to 1
#   bash ./alchemist/setup/macos-bootstrap.sh

ALROOT=$PWD
ALPREFIX=$ALROOT/bins

MAKE_THREADS=8

# Set the following flags to indicate what needs to be installed
# NOTE: skylark doesn't link against the available version of COMBBLAS, so enabling
# COMBBLAS will build it, but skylark will not use it
WITH_BREW_PREREQS=0
WITH_EL=1
WITH_COMBBLAS=0
WITH_RANDOM123=1
WITH_HDF5=1
WITH_SKYLARK=1
WITH_ARPACK=1
WITH_ARPACKPP=1
WITH_EIGEN=1
WITH_SPDLOG=1

# install brewable prereqs if not already there
# TODO: really don't like installing brew packages w/ nonstandard compiler, but works for now
# try to replace those brew calls with explicit builds managed in the booststrap script 
if [ $WITH_BREW_PREREQS = 1 ]; then
  xcode-select --install
  brew install gcc
  brew install make --with-default-names
  brew install cmake
  brew install boost --cc=gcc-7
  brew install boost-mpi --cc=gcc-7
  # may need to do the following if get a dyld error about finding mpi libraries, see https://github.com/Homebrew/homebrew-science/issues/6425
  #  brew reinstall --build-from-source boost-mpi
  brew install sbt
  brew install gmp
  brew install fftw
  brew install zlib
  brew install szlib
fi

export CC="gcc-7"
export CXX="g++-7"
export FC="gfortran-7"

# Start download auxiliary packages
mkdir -p $ALPREFIX
mkdir -p dl
cd dl

# Elemental
if [ "$WITH_EL" = 1 ]; then
  if [ ! -d Elemental ]; then
    git clone https://github.com/elemental/Elemental
  fi
  cd Elemental
  git checkout v0.87.4
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  # had to set LDFLAGS using output of mpicc --showme:link
  # otherwise Parmetis build fails with a link error showing it can't find MPI
  LDFLAGS="-L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/3.0.0_2/lib -lmpi" cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DEL_IGNORE_OSX_GCC_ALIGNMENT_PROBLEM=ON \
    -DCMAKE_INSTALL_PREFIX=$ALPREFIX \
    ..
  nice make -j"$MAKE_THREADS" 
  make install
  cd ../..
fi

# Combinatorial BLAS
if [ "$WITH_COMBBLAS" = 1 ]; then
  if [ ! -d CombBLAS_beta_16_1 ]; then
    curl -L  http://eecs.berkeley.edu/~aydin/CombBLAS_FILES/CombBLAS_beta_16_1.tgz > CombBLAS_beta_16_1.tgz
    tar xvfz CombBLAS_beta_16_1.tgz
  fi
  cd CombBLAS_beta_16_1
  # the C and C++ compiler were explicitly set to mpicc and mpicxx, which call Clang. remove those settings
  sed -i.bak -e '1,2d' CMakeLists.txt
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  cmake \
    -DCMAKE_EXE_LINKER_FLAGS="-lmpi" ..
  nice make -j$MAKE_THREADS
  if [ ! -d ../lib ]; then
    mkdir ../lib
  else
    rm -rf ../lib/*
  fi
  cp *.a ../lib
  cp graph500-1.2/generator/*.a ../lib
  cp usort/*.a ../lib
  cd ../..
  cp -r CombBLAS_beta_16_1 $ALPREFIX
fi

# Random123
if [ "$WITH_RANDOM123" = 1 ]; then
  if [ ! -d Random123-1.08 ]; then
    curl -L http://www.thesalmons.org/john/random123/releases/1.08/Random123-1.08.tar.gz > Random123-1.08.tar.gz
    tar xvfz Random123-1.08.tar.gz
  fi
  cp -rf Random123-1.08/include/Random123 $ALPREFIX/include
fi

# HDF5
# TODO: figure out how to use CMAKE so it can detect zlib and szip
# for now, have to manually update the library paths in the install code if versions of packages change
# the include and lib for the szlib and zlib packages were obtained using brew ls 'package'
# ditto for the cflags and ldflags for open-mpi: I used mpicc --showme:compile / --showme:link
if [ "$WITH_HDF5" = 1 ]; then
	if [ ! -d hdf5-1.10.1 ]; then
		curl -L https://support.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.10.1.tar > hdf5-1.10.1.tar
		tar xzf hdf5-1.10.1.tar
	fi
	cd hdf5-1.10.1
  CFLAGS="-I/usr/local/Cellar/open-mpi/3.0.0_2/include" \
  FCFLAGS="-I/usr/local/Cellar/open-mpi/3.0.0_2/include" \
  LDFLAGS="-L/usr/local/opt/libevent/lib -L/usr/local/Cellar/open-mpi/3.0.0_2/lib -lmpi -lmpi_mpifh" \
  ./configure --prefix="$ALPREFIX" \
		--enable-fortran \
    --enable-parallel \
		--with-zlib=/usr/local/opt/zlib/include,/usr/local/opt/zlib/lib \
		--with-szlib=/usr/local/Cellar/szip/2.1.1/include/,/usr/local/Cellar/szip/2.1.1/lib/ 
	nice make -j"$MAKE_THREADS"
	make install
  cd ..
fi

# Skylark
# Don't support Skylark w/ HDF5 because it requires the C++ bindings, which are 
# crappy and conflict with using HDF5 w/ MPIO support
if [ "$WITH_SKYLARK" = 1 ]; then
  if [ ! -d libskylark ]; then
    git clone https://github.com/xdata-skylark/libskylark.git
  fi
  cd libskylark
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  export ELEMENTAL_ROOT="$ALPREFIX"
  export COMBBLAS_ROOT="$ALPREFIX/CombBLAS_beta_16_1"
  export RANDOM123_ROOT="$ALPREFIX"
  export HDF5_ROOT="."
  CXXFLAGS="-dynamic -std=c++14 -fext-numeric-literals" cmake \
    -DCMAKE_INSTALL_PREFIX="$ALPREFIX" \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DUSE_HYBRID=OFF \
    -DUSE_FFTW=ON \
    -DUSE_COMBBLAS=OFF \
    -DBUILD_PYTHON=OFF \
    -DBUILD_EXAMPLES=ON ..
  VERBOSE=1 nice make -j"$MAKE_THREADS"
  make install
  cd ../..

  # skylark's install messes with the id which gives dyld errors about
  # not being able to find the libcskylark.so file, so give the correct id
  # cf the cmake_install.cmake file in libskylark/build/capi
  install_name_tool -id $ALPREFIX/lib/libcskylark.so $ALPREFIX/lib/libcskylark.so

  # for some reason this directory is not installed, so manually copy it
  cp -r libskylark/utility/fft $ALPREFIX/include/skylark/utility
fi

# arpack-ng
if [ "$WITH_ARPACK" = 1 ]; then
  if [ ! -d arpack-ng ]; then
    git clone https://github.com/opencollab/arpack-ng
  fi
  cd arpack-ng
  git checkout 3.5.0
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  cmake -DMPI=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  nice make -j"$MAKE_THREADS"
  make install
  cd ../..
fi

# arpackpp
if [ "$WITH_ARPACKPP" = 1 ]; then
  if [ ! -d arpackpp ]; then
    git clone https://github.com/m-reuter/arpackpp
  fi
  cd arpackpp
  git checkout 88085d99c7cd64f71830dde3855d73673a5e872b
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  make install
  cd ../..
fi

# Eigen
if [ "$WITH_EIGEN" = 1 ]; then
  if [ ! -d eigen-eigen-5a0156e40feb ]; then
    curl -L http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2 | tar xvfj -
  fi
  cd eigen-eigen-5a0156e40feb
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  nice make -j"$MAKE_THREADS"
  make install
  cd ../..
fi

# SPDLog
if [ "$WITH_SPDLOG" = 1 ]; then
  if [ ! -d spdlog ]; then
    git clone https://github.com/gabime/spdlog.git
  fi
  cd spdlog
  git checkout 4fba14c79f356ae48d6141c561bf9fd7ba33fabd
  if [ ! -d build ]; then
    mkdir build
  else
    rm -rf build/*
  fi
  cd build
  cmake -DCMAKE_INSTALL_PREFIX=$ALPREFIX ..
  make install -j"$MAKE_THREADS"
  cd ../..
fi

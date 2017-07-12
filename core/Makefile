ifdef ALPREFIX
  EL_PATH ?= $(ALPREFIX)
  EIGEN_PATH ?= $(ALPREFIX)/include
  ARPACKPP_PATH ?= $(ALPREFIX)/include/arpackpp
endif
EL_PATH ?= $(CURDIR)/../../bins
EIGEN_PATH ?= $(CURDIR)/../../bins/include
ARPACKPP_PATH ?= $(CURDIR)/../../bins/include/arpackpp
include $(EL_PATH)/conf/ElVars

SRC_PATH = src/main

ifdef ALPREFIX
  LDFLAGS += -Wl,-rpath,$(ALPREFIX)/lib
endif

ifdef BOOST_DIR
  CXXFLAGS += "-I$(BOOST_DIR)/include"
  LDFLAGS += "-L$(BOOST_DIR)/lib" "-Wl,-rpath,$(BOOST_DIR)/lib"
endif

CXXFLAGS += $(EL_COMPILE_FLAGS) -Wall -Wno-unused -I $(EIGEN_PATH) -I $(ARPACKPP_PATH)
#CXXFLAGS += -O0
LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
LDFLAGS += -lboost_serialization -lboost_mpi
LDFLAGS += -larpack

CXX_OBJS = \
	$(SRC_PATH)/cpp/alchemist.o \
	$(SRC_PATH)/cpp/driver.o \
	$(SRC_PATH)/cpp/worker.o \
	#

.PHONY: default
default: target/alchemist

target/alchemist: target $(SRC_PATH)/cpp/alchemist.h $(CXX_OBJS)
	$(CXX) -dynamic $(CXXFLAGS) -o $@ $(CXX_OBJS) $(LDFLAGS)

target:
	mkdir -p target

.PHONY: clean
clean:
	rm -rf target $(CXX_OBJS)

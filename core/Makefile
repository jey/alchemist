ifdef ALPREFIX
  EL_PATH ?= $(ALPREFIX)
  EIGEN_PATH ?= $(ALPREFIX)/include
endif
EL_PATH ?= $(CURDIR)/../../bins
EIGEN_PATH ?= $(CURDIR)/../../bins/include
include $(EL_PATH)/conf/ElVars

SRC_PATH = src/main

CXXFLAGS += $(EL_COMPILE_FLAGS) -Wall -Wno-unused -I $(EIGEN_PATH)
#CXXFLAGS += -O0
LDFLAGS += $(EL_LINK_FLAGS) -Wl,-rpath,$(EL_LIB) $(EL_LIBS)
LDFLAGS += -lboost_serialization -lboost_mpi

ifdef ALPREFIX
  LDFLAGS += -Wl,-rpath,$(ALPREFIX)/lib
endif

CXX_SRCS = \
	$(SRC_PATH)/cpp/alchemist.cpp \
	$(SRC_PATH)/cpp/driver.cpp \
	$(SRC_PATH)/cpp/worker.cpp \
	#

.PHONY: default
default: target/alchemist

target/alchemist: target $(SRC_PATH)/cpp/alchemist.h $(CXX_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $(CXX_SRCS) $(LDFLAGS)

target:
	mkdir -p target

.PHONY: clean
clean:
	rm -rf target

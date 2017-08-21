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
TARGET_PATH = target

CXXFLAGS += $(EL_COMPILE_FLAGS) -Wall -Wno-unused -I $(EIGEN_PATH) -I $(ARPACKPP_PATH) -std=c++14 -O3 
LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
LDFLAGS += -lboost_serialization -lboost_mpi 
LDFLAGS += -larpack

ifeq ($(shell uname), "Darwin")
  LDFLAGS += -lmpi
endif

ifdef ALPREFIX
  LDFLAGS += -Wl,-rpath,$(ALPREFIX)/lib
endif

ifdef BOOST_DIR
  CXXFLAGS += "-I$(BOOST_DIR)/include"
  LDFLAGS += "-L$(BOOST_DIR)/lib" "-Wl,-rpath,$(BOOST_DIR)/lib"
endif

OBJ_FILES = \
	$(TARGET_PATH)/kmeans++.o \
	$(TARGET_PATH)/alchemist.o \
	$(TARGET_PATH)/driver.o \
	$(TARGET_PATH)/worker.o \
	#

$(TARGET_PATH)/%.o: $(SRC_PATH)/cpp/%.cpp $(SRC_PATH)/cpp/alchemist.h
	$(CXX) -c $(CXXFLAGS) $< -o $@

.PHONY: default
default: $(TARGET_PATH)/alchemist

$(TARGET_PATH)/alchemist: $(TARGET_PATH) $(OBJ_FILES)
	$(CXX) -dynamic $(CXXFLAGS) -o $@ $(OBJ_FILES) $(LDFLAGS)

$(TARGET_PATH):
	mkdir -p $(TARGET_PATH)

.PHONY: clean
clean:
	rm -rf $(TARGET_PATH)

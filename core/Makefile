ALPREFIX ?= $(CURDIR)/../../bins
EL_PATH ?= $(ALPREFIX)
include $(EL_PATH)/conf/ElVars

SRC_PATH = src/main
TARGET_PATH = target

# put libEl's CXXFLAGS in front so ours can override it
CXXFLAGS += $(EL_COMPILE_FLAGS) -fdiagnostics-color=always
#CXXFLAGS += -Wall
CXXFLAGS += -Wno-unused -Wno-reorder -std=c++14 -fext-numeric-literals
CXXFLAGS += "-I$(ALPREFIX)/include"
LDFLAGS += "-L$(EL_LIB)" "-Wl,-rpath,$(EL_LIB)" $(EL_LIBS)
ifeq ($(shell uname), Darwin)
LDFLAGS += -lboost_serialization-mt -lboost_mpi-mt -lboost_system-mt -larpack -lboost_thread-mt
else
LDFLAGS += -lboost_serialization -lboost_mpi -lboost_system -larpack -lboost_thread 
endif
LDFLAGS += -Wl,-rpath,$(ALPREFIX)/lib

# poor heuristic to check if on Cori 
ifeq ($(shell uname), Linux)
#fix a linkage issue that shouldn't be occurring
LDFLAGS += -Wl,-rpath,/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin/ -L/opt/intel/compilers_and_libraries_2016.3.210/linux/compiler/lib/intel64_lin/ -lirc
endif

ifeq ($(shell uname), Darwin)
HDF5_BASE = $(shell brew --prefix hdf5)
LDFLAGS += "-L$(HDF5_BASE)/lib" -lmpi -lgfortran -lhdf5 -lhdf5_hl -lhdf5_hl_cpp -lhdf5_cpp
CXXFLAGS += "-I$(HDF5_BASE)/include"
endif

# some environments (e.g. Cori) require this to find boost
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

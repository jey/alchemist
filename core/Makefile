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
#CXXFLAGS += -O0
LDFLAGS += $(EL_LINK_FLAGS) -Wl,-rpath,$(EL_LIB) $(EL_LIBS)
LDFLAGS += -lboost_serialization -lboost_mpi -lmpi
LDFLAGS += -larpack

ifdef ALPREFIX
  LDFLAGS += -Wl,-rpath,$(ALPREFIX)/lib
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
	$(CXX) -o $(TARGET_PATH)/alchemist $(OBJ_FILES) $(LDFLAGS)

$(TARGET_PATH):
	mkdir -p $(TARGET_PATH)

.PHONY: clean
clean:
	rm -rf $(TARGET_PATH)

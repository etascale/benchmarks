TARGET=streamcluster-serial
OBJS=streamcluster.o

ifdef version
	ifeq "$(version)" "pthreads"
	CXXFLAGS := $(CXXFLAGS) -DENABLE_THREADS -pthread
	OBJS += parsec_barrier.o
	TARGET = streamcluster-pthreads
endif
ifeq "$(version)" "ompss"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss --no-copy-deps
	OBJS = ompss_streamcluster.o
	TARGET = streamcluster-ompss
endif
ifeq "$(version)" "ompss_instr"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss --no-copy-deps --instrument
	OBJS = ompss_streamcluster.o
	TARGET = streamcluster-ompss_instr
endif
ifeq "$(version)" "ompss-2"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss-2 --no-copy-deps
	OBJS = ompss-2_streamcluster.o
	TARGET = streamcluster-ompss-2
endif
ifeq "$(version)" "ompss-2_instr"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss-2 --no-copy-deps --instrument
	OBJS = ompss-2_streamcluster.o
	TARGET = streamcluster-ompss-2_instr
endif
ifeq "$(version)" "ompss-2-cluster"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss-2 --no-copy-deps
	OBJS = ompss-2-cluster_streamcluster.o
	TARGET = streamcluster-ompss-2-cluster
endif
ifeq "$(version)" "ompss-2-cluster_instr"
	CXX := mcxx
	CXXFLAGS := $(CXXFLAGS) -DENABLE_OMPSS --ompss-2 --no-copy-deps --instrument
	OBJS = ompss-2-cluster_streamcluster.o
	TARGET = streamcluster-ompss-2-cluster_instr
endif
ifeq "$(version)" "omp2"
	CXX := g++
	CXXFLAGS := $(CXXFLAGS) -DSCHED_POLICY=static -fopenmp
	OBJS = omp2_streamcluster.o
	TARGET = streamcluster-omp2
endif

endif

all: $(MAKEOBJECTIVES) $(OBJS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS)

## Files *.o need *.cpp, compile them
%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f *.o $(TARGET)

install:
	mkdir -p ../bin
	cp -f $(TARGET) ../bin/$(TARGET)


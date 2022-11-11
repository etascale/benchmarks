TARGET := fluidanimate
OBJS := serial.o cellpool.o
CXXFLAGS := $(CXXFLAGS) -Wno-invalid-offsetof -D_GNU_SOURCE -D__XOPEN_SOURCE=600
ifdef version
	ifeq "$(version)" "serial"
		CXX := g++
		CC := gcc
		CXXFLAGS += $(CXXFLAGS)
		OBJS = serial.o cellpool.o
		TARGET = fluidanimate-serial
	endif
	ifeq "$(version)" "pthreads"
		CXX := g++
		CC := gcc
		OBJS = pthreads.o cellpool.o parsec_barrier.o
		CXXFLAGS += -pthread $(CXXFLAGS)
		TARGET = fluidanimate-pthreads
	endif
	ifeq "$(version)" "ompss"
		CXX := mcxx
		CC := mcc
		CXXFLAGS += --ompss --no-copy-deps $(CXXFLAGS)
		OBJS = ompss-multideps-nobar.o cellpool.o
		TARGET = fluidanimate-ompss
	endif
	ifeq "$(version)" "ompss_instr"
		CXX := mcxx
		CC := mcc
		CXXFLAGS += --ompss --no-copy-deps --instrument $(CXXFLAGS)
		OBJS = ompss-multideps-nobar.o cellpool.o
		TARGET = fluidanimate-ompss_instr
	endif
	ifeq "$(version)" "omp2"
		CXX := g++
		CC := gcc
		CXXFLAGS += -fopenmp -DSCHED_POLICY=static $(CXXFLAGS)
		OBJS = omp2.o cellpool.o
		TARGET = fluidanimate-omp2
	endif
	ifeq "$(version)" "omp4"
		CXX := g++
		CC := gcc
		CXXFLAGS += -fopenmp $(CXXFLAGS)
		OBJS = omp4.o cellpool.o
		TARGET = fluidanimate-omp4
	endif
endif


all: $(OBJS) fluidcmp
	$(CXX) $(CXXFLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<
	
fluidcmp: fluidcmp.cpp
	rm -f fluidcmp
	$(CXX) fluidcmp.cpp -o fluidcmp

clean:
	rm -rf *.o fluidcmp $(TARGET)

install:
	mkdir -p ../bin
	mv -f $(TARGET) ../bin
	mv -f fluidcmp ../bin

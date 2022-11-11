export ARGO_INSTALL_DIRECTORY=${HOME}/argodsm/install

MCXX = g++
EXEC = swaptions-serial

OBJS = CumNormalInv.o MaxFunction.o RanUnif.o nr_routines.o icdf.o  \
       HJM_SimPath_Forward_Blocking.o HJM.o HJM_Swaption_Blocking.o \
       HJM_Securities.o

CXXFLAGS += -Wno-deprecated -Wno-write-strings
LIBS     := $(LIBS) -lm

ifdef version
	ifeq "$(version)" "pthreads"
	DEF      := $(DEF) -DENABLE_THREADS -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) -pthread
	EXEC      = swaptions-pthreads
endif

ifeq "$(version)" "ompss"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) --ompss --no-copy-deps
	EXEC      = swaptions-ompss
endif

ifeq "$(version)" "ompss_instr"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) --ompss --no-copy-deps --instrument
	EXEC      = swaptions-ompss_instr
endif

ifeq "$(version)" "ompss-2"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS_2 -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) --ompss-2 --no-copy-deps
	EXEC      = swaptions-ompss-2
endif

ifeq "$(version)" "ompss-2_instr"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS_2 -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) --ompss-2 --no-copy-deps --instrument
	EXEC      = swaptions-ompss-2_instr
endif

ifeq "$(version)" "ompss-2-cluster"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS_2_CLUSTER -DENABLE_OUTPUT -DENABLE_WEAK
	CXXFLAGS := $(CXXFLAGS) --ompss-2 --no-copy-deps
	EXEC      = swaptions-ompss-2-cluster
endif

ifeq "$(version)" "ompss-2-cluster_instr"
	MCXX      = mcxx
	DEF      := $(DEF) -DENABLE_OMPSS_2_CLUSTER -DENABLE_OUTPUT #-DENABLE_WEAK
	CXXFLAGS := $(CXXFLAGS) --ompss-2 --no-copy-deps --instrument
	EXEC      = swaptions-ompss-2-cluster_instr
endif

ifeq "$(version)" "omp2"
	MCXX      = g++
	DEF      := $(DEF) -DENABLE_OMP2 -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) -fopenmp -DSCHED_POLICY=static
	EXEC      = swaptions-omp2
endif

ifeq "$(version)" "omp4"
	MCXX      = g++
	DEF      := $(DEF) -DENABLE_OMP4 -DENABLE_OUTPUT
	CXXFLAGS := $(CXXFLAGS) -fopenmp 
	EXEC      = swaptions-omp4
endif

ifeq "$(version)" "argo"
	MCXX      = mpic++
	DEF      += -DENABLE_ARGO -DENABLE_OUTPUT -DSCHED_POLICY=static
	CXXFLAGS += -fopenmp -std=c++11
	CXXFLAGS += -I${ARGO_INSTALL_DIRECTORY}/include/argo
	CXXFLAGS += -L${ARGO_INSTALL_DIRECTORY}/lib
	CXXFLAGS += -Wl,-rpath=${ARGO_INSTALL_DIRECTORY}/lib
	LIBS     += -largo -largobackend-mpi -lnuma -lrt
	EXEC      = swaptions-argo
endif

endif

all: $(EXEC)

$(EXEC): $(OBJS)
	$(MCXX) $(CXXFLAGS) $(LDFLAGS) $(DEF) $(OBJS) $(INCLUDE) $(LIBS) -o $(EXEC)

.cpp.o:
	$(MCXX) $(CXXFLAGS) $(DEF) -c $*.cpp -o $*.o

.c.o:
	$(MCXX) $(CXXFLAGS) $(DEF) -c $*.c -o $*.o

clean:
	rm -f $(OBJS) $(EXEC)

install: $(EXEC)
	mkdir -p ../bin
	cp -f $(EXEC) ../bin/$(EXEC)


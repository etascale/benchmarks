export ARGO_INSTALL_DIRECTORY=${HOME}/argodsm/install

CC	= g++
SRC     = blackscholes.c
CSRC    = blackscholes.c
TARGET  = blackscholes-serial 
M4_BASE = .
M4      = m4
MACROS  = c.m4.pthreads
LIBS    = -lm


ifdef version
	ifeq "$(version)" "pthreads"
	M4_SRC   = blackscholes.m4.cpp
	CSRC     = $(M4_SRC)
	MT       = -DENABLE_THREADS -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += -pthread
	TARGET   = blackscholes-pthreads
endif
ifeq "$(version)" "omp2"
	MT       = -DENABLE_OPENMP -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += -fopenmp -DSCHED_POLICY=static
	TARGET   = blackscholes-omp2
endif
ifeq "$(version)" "ompss"
	CC       = mcc
	CSRC     = blackscholes-ompss.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += --ompss --no-copy-deps
	TARGET   = blackscholes-ompss
endif
ifeq "$(version)" "ompss_instr"
	CC       = mcc
	CSRC     = blackscholes-ompss.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += --ompss --no-copy-deps --instrument
	TARGET   = blackscholes-ompss_instr
endif
ifeq "$(version)" "ompss-2"
	CC       = mcc
	CSRC     = blackscholes-ompss-2.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += --ompss-2 --no-copy-deps
	TARGET   = blackscholes-ompss-2
endif
ifeq "$(version)" "ompss-2_instr"
	CC       = mcc
	CSRC     = blackscholes-ompss-2.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += --ompss-2 --no-copy-deps --instrument
	TARGET   = blackscholes-ompss-2_instr
endif
ifeq "$(version)" "ompss-2-cluster-strong"
	CC       = mcc
	CSRC	 = blackscholes-ompss-2-cluster-strong.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT #-DERR_CHK
	CFLAGS	+= --ompss-2 --no-copy-deps
	TARGET   = blackscholes-ompss-2-cluster-strong
endif
ifeq "$(version)" "ompss-2-cluster-strong_instr"
	CC       = mcc
	CSRC	 = blackscholes-ompss-2-cluster-strong.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT #-DERR_CHK
	CFLAGS	+= --ompss-2 --no-copy-deps --instrument
	TARGET   = blackscholes-ompss-2-cluster-strong_instr
endif
ifeq "$(version)" "ompss-2-cluster-weak"
	CC       = mcc
	CSRC	 = blackscholes-ompss-2-cluster-weak.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT #-DERR_CHK
	CFLAGS	+= --ompss-2 --no-copy-deps
	TARGET   = blackscholes-ompss-2-cluster-weak
endif
ifeq "$(version)" "ompss-2-cluster-weak_instr"
	CC       = mcc
	CSRC	 = blackscholes-ompss-2-cluster-weak.c
	MT       = -DENABLE_OMPSS -DENABLE_OUTPUT #-DERR_CHK
	CFLAGS	+= --ompss-2 --no-copy-deps --instrument
	TARGET   = blackscholes-ompss-2-cluster-weak_instr
endif
ifeq "$(version)" "omp4"
	CC       = gcc
	CSRC     = blackscholes-omp.c
	MT       = -DENABLE_OPENMP -DENABLE_OUTPUT -DERR_CHK
	CFLAGS  += -fopenmp
	TARGET   = blackscholes-omp4
endif
ifeq "$(version)" "argo"
	CC       = mpic++
	CSRC     = blackscholes-argo.cpp
	MT       = -DSCHED_POLICY=static -DENABLE_OUTPUT #-DERR_CHK
	CFLAGS  += -fopenmp -std=c++11
	IDFLAGS += -I${ARGO_INSTALL_DIRECTORY}/include/argo
	LDFLAGS += -L${ARGO_INSTALL_DIRECTORY}/lib
	LDFLAGS += -Wl,-rpath=${ARGO_INSTALL_DIRECTORY}/lib
	LIBS    += -largo -largobackend-mpi -lnuma -lrt
	TARGET   = blackscholes-argo
endif

else
	MT = -DENABLE_OUTPUT -DERR_CHK
endif

# Default build single precision version
NCO = -DNCO=4

ifdef chk_err
	ERR = -DERR_CHK
endif

ifdef single
	NCO = -DNCO=4
endif

ifdef size
	SZ = -DN=$(size)
else
	SZ = -DN=960
endif

ifdef double
	NCO = -DNCO=2
endif

CFLAGS += $(MT) $(SZ) $(NCO) $(FUNC) $(ERR) $(CSRC)

all: $(TARGET)

$(TARGET): clean $(CSRC)
	$(CC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET)

$(M4_SRC): $(SRC)
	$(M4) $(M4_BASE)/$(MACROS) $< > $@

clean:
	rm -f $(TARGET) $(M4_SRC)

install:
	mkdir -p ../bin
	cp -f $(TARGET) ../bin/$(TARGET)

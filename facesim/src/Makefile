#DIRECTORY IN WHICH THE BENCHMARK DIRECTORY RESIDES
#Prefix->
BINARY		:= facesim
CURRENTDIR	:= $(CURDIR)
#LINUX VARIABLES
export MKDIR					=	mkdir -p
export LS						=	ls
export CP						=	cp -af
export MV						=	rm -rf
export DATE						=	date
export LOGDATE					=	date +%F_%T
export TEE						=	tee -a
export CAT						=	cat
export UNTAR					=	tar -xvf
export CMP						=	cmp
export SED						=	sed -f

#BUILDING ENVIRONMENT VARIABLES
export CC_HOME					=	/usr
export BINUTIL_HOME				=	/usr
export GNUTOOL_HOME				=	/usr
export BINARY_PREFIX			=

#GNU BINUTILS
export LD						=	$(BINUTIL_HOME)/bin/$(BINARY_PREFIX)ld
export AS						=	$(BINUTIL_HOME)/bin/$(BINARY_PREFIX)as
export AR						=	$(BINUTIL_HOME)/bin/$(BINARY_PREFIX)ar
export RANLIB					=	$(BINUTIL_HOME)/bin/$(BINARY_PREFIX)ranlib
export STRIP					=	$(BINUTIL_HOME)/bin/$(BINARY_PREFIX)strip

#GNU TOOLS
export M4						=	$(GNUTOOL_HOME)/bin/$(BINARY_PREFIX)m4
export MAKE						=	$(GNUTOOL_HOME)/bin/$(BINARY_PREFIX)make

#PHYSBAM DIRECTORY
export PHYSBAM					=	$(CURRENTDIR)

#OTHER FLAGS USED BY THE PHYSBAM LIBRARY
#export LINK_FLAGS				=
#export PTHREAD_LINK_FLAGS		=
export TYPE						= release

#XML2 FLAGS
export XML2_LINK_FLAGS          =   -L/usr/lib64 -lxml2

#FACESIM LIBS
export MAKEOBJECTIVES			=

#    @echo $(PHYSBAM)

#ifdef enablepapi
#	LINK_FLAGS			+= -L$(PAPI_HOME)/lib -lpapi
#	CXXFLAGS			+= -DENABLE_PAPI -I$(PAPI_HOME)/include
#endif

#FLAGS FOR INCREASED PORTABILITY OF BINARIES
#export PORTABILITY_FLAGS		=	-static-libgcc -Wl,--hash-style=both,--as-needed

#ifdef version
#	ifeq (, $(findstring ompss,$(version))) #si findstring no encuentra la cadena ompss en "$version", devuelve vacio
#		export PORTABILITY_FLAGS	=	-static-libgcc -Wl,--hash-style=both,--as-needed
#	else
#		export PORTABILITY_FLAGS	=	--Wn,-static-libgcc -Wl,--hash-style=both,--as-needed
#	endif
#endif


# FLAGS PARTICULAR TO FACESIM BENCHMARK
ifdef version
	ifeq ("$(version)", "serial")
		CXXFLAGS		+= -DNEW_SERIAL_IMPLEMENTATIOM -g -fexceptions
		TYPE			:=release
		MAKEOBJECTIVES		+=taskq
	endif
	ifeq ("$(version)", "pthreads")
		CXXFLAGS		+= -DENABLE_PTHREADS -g -fexceptions
		TYPE			:=release
		MAKEOBJECTIVES		+=taskq
	endif
	ifeq ("$(version)", "ompss")
		CC				:=  mcc
		CXX				:=  mcxx
		CXXFLAGS		+= 	-DENABLE_OMPSS -DUSE_TASKS -fexceptions --ompss --no-copy-deps
		export TYPE		:=noalamere
	endif
	ifeq ("$(version)", "ompss_instr")
		CC				:=  mcc
		CXX				:=  mcxx
		CXXFLAGS		+= 	-DENABLE_OMPSS -DUSE_TASKS -fexceptions --ompss --no-copy-deps --instrument
		export TYPE		:=noalamere
	endif
	ifeq ("$(version)", "ompss-hyb")
		CC				:=  mcc
		CXX				:=  mcxx
		CXXFLAGS		+= 	-DENABLE_OMPSS -DUSE_TASKS -DUSE_HYBRID -fexceptions --ompss --no-copy-deps
		export TYPE		:=noalamere
	endif
    ifeq ("$(version)", "omp4")
		CC				:=  g++
		CXX				:=  g++
		CXXFLAGS		+= 	-DENABLE_OPENMP -DUSE_TASKS -fexceptions -fopenmp
		export TYPE		:=noalamere
	endif
    ifeq ("$(version)", "omp4-hyb")
		CC				:=  g++
		CXX				:=  g++
		CXXFLAGS		+= 	-DENABLE_OPENMP -DUSE_TASKS -DUSE_HYBRID -fexceptions -fopenmp
		export TYPE		:=noalamere
	endif

endif

all: $(MAKEOBJECTIVES)
	$(MAKE) -C Benchmarks/facesim

taskq:
	$(MAKE) -C TaskQ/lib all

install:
	mkdir -p $(ROOT)/facesim/bin
	cp -f $(ROOT)/facesim/src/Benchmarks/facesim/$(BINARY) $(ROOT)/facesim/bin/$(BINARY)-$(version)

clean:
	$(MAKE) -C Benchmarks/facesim clean
	$(MAKE) -C Public_Library/ clean
	$(MAKE) -C TaskQ/lib clean

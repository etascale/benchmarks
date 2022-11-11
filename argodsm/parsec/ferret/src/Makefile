PREFIX = ..
TARGET = ferret-serial

LDLIBS += $(LIBS) -lrt -lm -lcass -lgsl -lgslcblas -ljpeg -lstdc++

.PHONY:	all clean install dirs lib tools image server benchmark 

ROOT = $(CURDIR)
SRCDIR = $(ROOT)/src
INCDIR = $(ROOT)/include

BINDIR = $(ROOT)/bin
LIBDIR = $(ROOT)/lib
OBJDIR = $(ROOT)/obj

CFLAGS += -I$(INCDIR) -I${PARSEC_LIBS}/gsl/include 
LDFLAGS += -L$(LIBDIR) -L${PARSEC_LIBS}/gsl/lib -L${PARSEC_LIBS}/gsl/lib64 -Wl,-rpath,${PARSEC_LIBS}/gsl/lib -Wl,-rpath,${PARSEC_LIBS}/gsl/lib64

VPATH = $(CURDIR)/src:$(CURDIR)/src/lsh:$(CURDIR)/tools:$(CURDIR)/image:$(CURDIR)/benchmark

export LIBDIR BINDIR OBJDIR CC CXX CFLAGS LDFLAGS LDLIBS

all:	dirs lib tools benchmark

dirs:
	mkdir -p $(BINDIR)
	mkdir -p $(LIBDIR)
	mkdir -p $(OBJDIR)

# build the cass library
lib:	$(LIBDIR)/libcass.a 

lib_src := $(wildcard src/*.c) $(wildcard src/lsh/*.c)
lib_src := $(notdir $(lib_src))
lib_obj := $(addprefix $(OBJDIR)/, $(lib_src:.c=.o))


$(LIBDIR)/libcass.a:	$(lib_obj)
	@echo "   A  '$@'"
	@$(AR) rcs $@ $^

# build the image support library
libimage_src := image.c extract.c edge.c srm.c
libimage_obj := $(addprefix $(OBJDIR)/, $(libimage_src:.c=.o))


$(LIBDIR)/libcassimage.a:	$(libimage_obj)
	@echo "   A  '$@'"
	@$(AR) rcs $@ $^

# build the tool programs
tools_src := $(wildcard tools/*.c)
tools_tgt := $(notdir $(tools_src))
tools_tgt := $(addprefix $(BINDIR)/, $(tools_tgt:.c=))

tools : $(tools_tgt)

server_src := $(wildcard server/*.c) 
server_tgt := $(notdir $(server_src))
server_tgt := $(addprefix $(BINDIR)/, $(server_tgt:.c=))

server : $(server_tgt)

# Default version to build
benchmark_base := ferret-serial

ifdef version
	TARGET=ferret-pthreads
  ifeq "$(version)" "pthreads"
    benchmark_base := ferret-pthreads
    LDLIBS += -lpthread
  endif
  ifeq "$(version)" "ompss"
	TARGET=ferret-ompss
    benchmark_base := ferret-ompss
    CC = mcc
    CXX = mcxx
    CFLAGS += --ompss --no-copy-deps
  endif  
  ifeq "$(version)" "ompss_instr"
	TARGET=ferret-ompss_instr
    benchmark_base := ferret-ompss
    CC = mcc
    CXX = mcxx
    CFLAGS += --ompss --no-copy-deps --instrument
  endif

  ifeq "$(version)" "omp4"
    TARGET=ferret-omp4
    benchmark_base := ferret-omp4
    CC = gcc
    CXX = g++
    CFLAGS += -fopenmp 
  endif
endif

benchmark_src := $(benchmark_base).c
ifeq "$(version)" "tbb"
  benchmark_src := $(benchmark_base).cpp
endif

benchmark_tgt := $(notdir $(benchmark_src))
benchmark_tgt := $(addprefix $(BINDIR)/, $(benchmark_tgt:.c=))

benchmark : $(benchmark_tgt)

image_tgt := cass_img_extract
image_tgt := $(addprefix $(BINDIR)/, $(image_tgt))

image: $(image_tgt)

clean:
	rm $(OBJDIR)/*

install:
	cp -r $(BINDIR) $(PREFIX)/
	cp -r $(LIBDIR) $(PREFIX)/
	mv $(BINDIR)/$(benchmark_base) $(PREFIX)/bin/$(TARGET)

$(OBJDIR)/%.o : %.c
	@echo "   CC '$<' ==> '$@'"
	@$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o : %.cpp
	@echo "   CXX '$<' ==> '$@'"
	@$(CXX) $(CFLAGS) $(CXXFLAGS) -c $< -o $@

$(tools_tgt): $(BINDIR)/% : $(OBJDIR)/%.o $(LIBDIR)/libcass.a
	@echo "   CXX '$<' ==> '$@'"
	@$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(benchmark_tgt): $(BINDIR)/% : $(OBJDIR)/%.o $(LIBDIR)/libcass.a $(LIBDIR)/libcassimage.a
	@echo "   CXX '$<' ==> '$@'"
	@$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(IMAGE_LDLIBS)

$(image_tgt): $(BINDIR)/% : $(OBJDIR)/%.o $(LIBDIR)/libcass.a $(LIBDIR)/libcassimage.a
	@echo "   CXX '$<' ==> '$@'"
	@$(CXX) $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS) $(IMAGE_LDLIBS)


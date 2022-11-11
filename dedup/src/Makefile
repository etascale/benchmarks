# Makefile for dedup kernel

TARGET=dedup-serial

CC=gcc

CFLAGS += -fno-strict-aliasing -D_XOPEN_SOURCE=600

ostype=$(findstring solaris, ${PARSECPLAT})

ifeq "$(ostype)" "solaris"
    CFLAGS += -std=gnu99
endif

LIBS += -lm

DEDUP_OBJ = hashtable.o util.o dedup.o rabin.o encoder.o decoder.o mbuffer.o sha.o

# Uncomment the following to enable gzip compression
CFLAGS += -DENABLE_GZIP_COMPRESSION
LIBS += -lz

# Uncomment the following to enable bzip2 compression
#CFLAGS += -DENABLE_BZIP2_COMPRESSION
#LIBS += -lbz2

ifdef version
  ifeq "$(version)" "pthreads"
    TARGET=dedup-pthreads
    CFLAGS += -DENABLE_PTHREADS -pthread
    DEDUP_OBJ += queue.o binheap.o tree.o
  endif
  
  ifeq "$(version)" "ompss"
    CC=mcc
    TARGET=dedup-ompss
    CFLAGS += -DENABLE_OMPSS -DENABLE_OMPSS_LOCKS --ompss --no-copy-deps
    DEDUP_OBJ += binheap.o tree.o
  endif

  ifeq "$(version)" "ompss_instr"
    CC=mcc
    TARGET=dedup-ompss_instr
    CFLAGS += -DENABLE_OMPSS -DENABLE_OMPSS_LOCKS --ompss --no-copy-deps --instrument
    DEDUP_OBJ += binheap.o tree.o
  endif

  ifeq "$(version)" "omp4"
    CC=gcc
    TARGET=dedup-omp4
    CFLAGS += -DENABLE_OMP -DENABLE_OMPSS_LOCKS -fopenmp 
    DEDUP_OBJ += binheap.o tree.o
  endif

endif


all: $(TARGET)

.c.o:
	$(CC) -c $(CFLAGS) $< -o $@

$(TARGET): $(DEDUP_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(TARGET) $(DEDUP_OBJ) $(LIBS)

clean:
	rm -f *~ *.o $(TARGET)

install:
	mkdir -p ../bin
	cp -f $(TARGET) ../bin/$(TARGET)


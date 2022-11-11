TARGET = freqmine-serial

CXXFLAGS += -Wno-deprecated

ifdef version
  ifeq "$(version)" "omp2"
    CXX=g++
    CXXFLAGS += -fopenmp -DSCHED_POLICY=static 
    TARGET=freqmine-omp2
    LDFLAGS += 
  endif
endif

ifdef version
  ifeq "$(version)" "ompss"
    CXX=mcxx
    CXXFLAGS += -D_OMPSS --ompss --no-copy-deps
    TARGET=freqmine-ompss
  endif
endif

ifdef version
  ifeq "$(version)" "ompss_instr"
    CXX=mcxx
    CXXFLAGS += -D_OMPSS --ompss --no-copy-deps --instrument
    TARGET=freqmine-ompss_instr
  endif
endif


ifdef version
  ifeq "$(version)" "omp4"
    CXX=g++
    CXXFLAGS += -D_OMP -fopenmp
    TARGET=freqmine-omp4
  endif
endif

all: $(TARGET)

$(TARGET): fpgrowth.o data.o fp_tree.o buffer.o fsout.o fp_node.o wtime.o
	$(CXX) $(CXXFLAGS) fpgrowth.o data.o fp_tree.o buffer.o fsout.o fp_node.o wtime.o $(LIBS) -o $(TARGET) ${LDFLAGS}

wtime.o: wtime.cpp wtime.h
	$(CXX) $(CXXFLAGS) -c wtime.cpp -o wtime.o

fpgrowth.o: fpmax.cpp data.h fp_tree.h common.h fsout.h
	$(CXX) $(CXXFLAGS) -c fpmax.cpp -o fpgrowth.o

data.o: data.cpp data.h
	$(CXX) $(CXXFLAGS) -c data.cpp

fp_tree.o: fp_tree.cpp data.h fp_tree.h common.h fsout.h
	$(CXX) $(CXXFLAGS) -c fp_tree.cpp

buffer.o: buffer.cpp data.h buffer.h
	$(CXX) $(CXXFLAGS) -c buffer.cpp

fsout.o: fsout.cpp fsout.h common.h
	$(CXX) $(CXXFLAGS) -c fsout.cpp

fp_node.o: fp_node.cpp fp_node.h fsout.h common.h
	$(CXX) $(CXXFLAGS) -c fp_node.cpp

clean:
	rm -f *.o $(TARGET)

install:
	mkdir -p ../bin
	cp -f $(TARGET) ../bin/$(TARGET)


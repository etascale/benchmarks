
all: taskQ.o

taskQ.o: 
	$(CXX) $(CXXFLAGS) $(INSTRUMENT_FLAGS) -DTASKQ_DIST_FIXED -c taskQDistCommon.c
	mv taskQDistCommon.o taskQ.o

clean:
	rm -f taskQ.o

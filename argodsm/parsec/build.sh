BENCHMARKS=$1
VERSIONS=$2
ADD_COMPILE_FLAGS=$3


PORTABILITY_FLAGS="-static-libgcc -Wl,--hash-style=both,--as-needed"
export CFLAGS=" -O3 -g -funroll-loops -fprefetch-loop-arrays ${PORTABILITY_FLAGS} ${ADD_COMPILE_FLAGS}"
export CXXFLAGS="-O3 -g -funroll-loops -fprefetch-loop-arrays -fpermissive -fno-exceptions ${PORTABILITY_FLAGS} ${ADD_COMPILE_FLAGS}"
#export CFLAGS="$CFLAGS ${ADD_COMPILE_FLAGS}"
#export CXXFLAGS="$CXXFLAGS ${ADD_COMPILE_FLAGS}"

# Check if parsec hooks should be used.
# Note that you will need to compile them if you want to use them (parseclibs/hooks)
if [[ $ADD_COMPILE_FLAGS == *"ENABLE_PARSEC_HOOKS"* ]]
then
export CFLAGS="$CFLAGS -I${ROOT}/parseclibs/hooks/include"
export CXXFLAGS="$CXXFLAGS -I${ROOT}/parseclibs/hooks/include"
export LDFLAGS="-L${ROOT}/parseclibs/hooks/lib"
export LIBS="-lhooks"
fi

for bench in ${BENCHMARKS}; do
	for version in ${VERSIONS}; do
		cd ${bench}
		./bench/build.sh ${version}
		cd ..
	done
done


#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4
#NDIVS=$4

if [ -z "$NDIVS" ]; then
    NDIVS=${NTHREADS}
fi

### STREAMCLUSTER ARGS###
### Define arguments for the PARSEC Benchmark and different input sizes ###
case $INPUT in
  "native") ARGS="10 20 128 1000000 200000 5000 none ${ROOT}/streamcluster/outputs/output.txt";;
  "simlarge") ARGS="10 20 128 16384 16384 1000 none ${ROOT}/streamcluster/outputs/output.txt";;
  "simmedium") ARGS="10 20 64 8192 8192 1000 none ${ROOT}/streamcluster/outputs/output.txt";;
  "simsmall") ARGS="10 20 32 4096 4096 1000 none ${ROOT}/streamcluster/outputs/output.txt";;
  "simdev") ARGS="3 10 3 16 16 10 none output.txt ${ROOT}/streamcluster/outputs/output.txt";;
  "test") ARGS="2 5 1 10 10 5 none ${ROOT}/streamcluster/outputs/output.txt";;
esac

mkdir -p ${ROOT}/streamcluster/outputs

#SPECIFY RUNTIME THREADS
case ${VERSION} in
    ompss*)
	export OMP_NUM_THREADS=${NTHREADS}
        export NX_ARGS="$EXTRA_ARGS ${NX_ARGS}"
        ;;
    omp* )
	    export OMP_NUM_THREADS=${NTHREADS}
        ;;
    pthreads*)
        ;;
	serial*)
		NTHREADS=1
		;;
    *)
        echo -e "\033[0;31mVERSION = $VERSION not correct, stopping $BENCHID run\033[0m"
        exit
        ;;
esac

${ROOT}/streamcluster/bin/streamcluster-${VERSION} ${ARGS} ${NTHREADS} ${NDIVS}

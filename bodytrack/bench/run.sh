#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

BENCHPATH=${ROOT}/bodytrack

case $INPUT in
  "native") ARGS="${BENCHPATH}/inputs/sequenceB_261 4 261 4000 5 0";;
  "simlarge") ARGS="${BENCHPATH}/inputs/sequenceB_4 4 4 4000 5 0";;
  "simmedium") ARGS="${BENCHPATH}/inputs/sequenceB_2 4 2 2000 5 0";;
  "simsmall") ARGS="${BENCHPATH}/inputs/sequenceB_1 4 1 1000 5 0";;
  "simdev") ARGS="${BENCHPATH}/inputs/sequenceB_1 4 1 100 3 0";;
  "test") ARGS="${BENCHPATH}/inputs/sequenceB_1 4 1 100 3 0";;
esac

if [ $VERSION = "omp4" ] || [ $VERSION = "omp2" ]; then

	export OMP_NUM_THREADS=${NTHREADS}

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="$EXTRA_ARGS"

fi

${BENCHPATH}/bin/bodytrack-${VERSION} ${ARGS} ${NTHREADS} 

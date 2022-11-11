#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

BENCHPATH=${ROOT}/freqmine

case $INPUT in
  "native") ARGS="${BENCHPATH}/inputs/webdocs_250k.dat 11000";;
  "simlarge") ARGS="${BENCHPATH}/inputs/kosarak_990k.dat 790";;
  "simmedium") ARGS="${BENCHPATH}/inputs/kosarak_500k.dat 410";;
  "simsmall") ARGS="${BENCHPATH}/inputs/webdocs_250k.dat 220";;
  "simdev") ARGS="${BENCHPATH}/inputs/T10I4D100K_1k.dat 3";;
  "test") ARGS="${BENCHPATH}/inputs/T10I4D100K_3.dat 1";;
esac

if [ $VERSION = "omp4" ] || [ $VERSION = "omp2" ]; then

	export OMP_NUM_THREADS=${NTHREADS}

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="$EXTRA_ARGS"

fi

${BENCHPATH}/bin/freqmine-${VERSION} ${ARGS}

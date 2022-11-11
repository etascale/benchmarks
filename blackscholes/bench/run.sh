#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

BENCHPATH=${ROOT}/blackscholes

case $INPUT in
  "native") ARGS="${BENCHPATH}/inputs/in_10M.txt ${BENCHPATH}/outputs/prices.txt";;
  "simlarge") ARGS="${BENCHPATH}/inputs/in_64K.txt ${BENCHPATH}/outputs/prices.txt";;
  "simmedium") ARGS="${BENCHPATH}/inputs/in_16K.txt ${BENCHPATH}/outputs/prices.txt";;
  "simsmall") ARGS="${BENCHPATH}/inputs/in_4K.txt ${BENCHPATH}/outputs/prices.txt";;
  "simdev") ARGS="${BENCHPATH}/inputs/in_1K.txt ${BENCHPATH}/outputs/prices.txt";;
  "test") ARGS="${BENCHPATH}/inputs/in_4.txt ${BENCHPATH}/outputs/prices.txt";;
esac

mkdir -p ${BENCHPATH}/outputs

if [ $VERSION = "omp4" ] || [ $VERSION = "omp2" ]; then

	export OMP_NUM_THREADS=${NTHREADS}

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="$EXTRA_ARGS"

fi

${BENCHPATH}/bin/blackscholes-${VERSION} ${NTHREADS} ${ARGS}

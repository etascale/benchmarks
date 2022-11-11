#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

BENCHPATH=${ROOT}/ferret

case $INPUT in
  "native") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 50 20";;
  "simlarge") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 10 20";;
  "simmedium") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 10 20";;
  "simsmall") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 10 20";;
  "simdev") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 5 5";;
  "test") ARGS="${BENCHPATH}/inputs/corel lsh ${BENCHPATH}/inputs/queries 5 5";;
esac

mkdir -p ${BENCHPATH}/outputs

if [ $VERSION = "omp4" ] || [ $VERSION = "omp2" ]; then

	export OMP_NUM_THREADS=${NTHREADS}

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="$EXTRA_ARGS --disable-ut"

fi

${BENCHPATH}/bin/ferret-${VERSION} ${ARGS} ${NTHREADS} ${BENCHPATH}/outputs/output.txt

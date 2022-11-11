#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

if [ -z "$NDIVS" ]; then
    NDIVS=${NTHREADS}
fi

BENCHPATH=${ROOT}/fluidanimate

case $INPUT in
  "native") ARGS="500 ${BENCHPATH}/inputs/in_500K.fluid ${BENCHPATH}/outputs/out_500K.fluid";;
  "simlarge") ARGS="5 ${BENCHPATH}/inputs/in_300K.fluid ${BENCHPATH}/outputs/out_300K.fluid";;
  "simmedium") ARGS="5 ${BENCHPATH}/inputs/in_100K.fluid ${BENCHPATH}/outputs/out_100K.fluid";;
  "simsmall") ARGS="5 ${BENCHPATH}/inputs/in_35K.fluid ${BENCHPATH}/outputs/out_35K.fluid";;
  "simdev") ARGS="3 ${BENCHPATH}/inputs/in_15K.fluid ${BENCHPATH}/outputs/out_15K.fluid";;
  "test") ARGS="1 ${BENCHPATH}/inputs/in_5K.fluid ${BENCHPATH}/outputs/out_5K.fluid";;
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

${BENCHPATH}/bin/fluidanimate-${VERSION} ${NTHREADS} ${ARGS} ${NDIVS} 

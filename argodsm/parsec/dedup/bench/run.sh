#!/bin/bash

BENCHMARK=dedup
VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

BENCHPATH=${ROOT}/${BENCHMARK}

case $INPUT in
	"native") ARGS="-c -p -v -i ${BENCHPATH}/inputs/FC-6-x86_64-disc1.iso -o ${BENCHPATH}/outputs/output.dat.ddp";;
	"simlarge") ARGS="-c -p -v -i ${BENCHPATH}/inputs/media.dat -o ${BENCHPATH}/outputs/output.dat.ddp";;
	"simmedium") ARGS="-c -p -v -i ${BENCHPATH}/inputs/media.dat -o ${BENCHPATH}/outputs/output.dat.ddp";;
	"simsmall") ARGS="-c -p -v -i ${BENCHPATH}/inputs/media.dat -o ${BENCHPATH}/outputs/output.dat.ddp";;
	"simdev") ARGS="-c -p -v -i ${BENCHPATH}/inputs/hamlet.dat -o ${BENCHPATH}/outputs/output.dat.ddp";;
	"test") ARGS="-c -p -v -i ${BENCHPATH}/inputs/test.dat -o ${BENCHPATH}/outputs/output.dat.ddp";;
esac

mkdir -p ${BENCHPATH}/outputs

if [ $VERSION = "omp4" ] || [ $VERSION = "omp2" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	NTHREADS=1

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then

	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="$EXTRA_ARGS"
	NTHREADS=1

fi

${BENCHPATH}/bin/${BENCHMARK}-${VERSION} $ARGS -t $NTHREADS

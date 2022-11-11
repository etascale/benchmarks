#!/bin/bash

BENCHMARK=$1
VERSION=$2
INPUT=$3
NTHREADS=$4
ITERS=$5
EXTRA_ARGS=$6
EXTRA_TAG=$7

BENCHNAME=${BENCHMARK}-${VERSION}-${INPUT}

mkdir -p ${ROOT}/dump

for (( iter=0; iter<${ITERS}; iter++ )) do
	${ROOT}/${BENCHMARK}/bench/run.sh ${VERSION} ${INPUT} ${NTHREADS} ${EXTRA_ARGS} > ${ROOT}/dump/${BENCHNAME}-${NTHREADS}${EXTRA_TAG}-${iter}.dmp
done

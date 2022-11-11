#!/bin/bash

BENCHMARK=$1
VERSION=$2
INPUT=$3
NTHREADS=$4
ITERS=$5
EXTRA_ARGS=$6
EXTRA_TAG=$7

BENCHNAME=${BENCHMARK}-${VERSION}-${INPUT}

#for (( nthrds=1; nthrds<=${THREADS}; nthrds*=2 ))
#do
echo -e "\033[32mGenerating and submitting ${BENCHNAME} with ${NTHREADS} threads\033[m"
#generate script and submit script
echo  "
#BSUB -n 1 
#BSUB -R\"affinity[core(${NTHREADS})]\"
#BSUB -oo ${ROOT}/dump/${BENCHNAME}-run.out
#BSUB -eo ${ROOT}/dump/${BENCHNAME}-run.err
#BSUB -J ${BENCHNAME}_run
#BSUB -q bsc_debug
#BSUB -W 03:00

mkdir -p ${ROOT}/dump 

for (( iter=0; iter<${ITERS}; iter++ )) do
	${ROOT}/${BENCHMARK}/bench/run.sh ${VERSION} ${INPUT} ${NTHREADS} ${EXTRA_ARGS} > ${ROOT}/dump/${BENCHNAME}-${NTHREADS}${EXTRA_TAG}-\${iter}.dmp
done" > temp.run
bsub < temp.run
rm temp.run
#done

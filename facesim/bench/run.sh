#!/bin/bash

VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4
#NDIVS=$4

if [ -z "$NDIVS" ]; then
    NDIVS=${NTHREADS}
fi

#DEFINE ARGS
case $INPUT in
  "native") ARGS="-timing -lastframe 100 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
  "simlarge") ARGS="-timing -lastframe 1 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
  "simmedium") ARGS="-timing -lastframe 1 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
  "simsmall") ARGS="-timing -lastframe 1 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
  "simdev") ARGS="-timing -lastframe 1 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
  "test") ARGS="-timing -lastframe 1 -inputdir ${ROOT}/facesim/inputs -outputdir ${ROOT}/facesim/output/Storytelling/output";;
esac

mkdir -p ${ROOT}/facesim/outputs
#ADD THREADS/PARTITIONING SCHEME
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


#RUN PROGRAM
${ROOT}/facesim/bin/facesim-${VERSION} ${ARGS} -threads ${NTHREADS} -ndivs ${NDIVS} 



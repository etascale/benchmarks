BENCHMARK=x264
VERSION=$1
INPUT=$2
NTHREADS=$3
EXTRA_ARGS=$4

if [ -z "$NDIVS" ]; then
    NDIVS=${NTHREADS}
fi

BENCHPATH=${ROOT}/${BENCHMARK}
case $INPUT in
  "native") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_1920x1080_512.y4m --threads ${NTHREADS}";;
  "simlarge") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_640x360_128.y4m --threads ${NTHREADS}";;
  "simmedium") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_640x360_32.y4m --threads ${NTHREADS}";;
  "simsmall") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_640x360_8.y4m --threads ${NTHREADS}";;
  "simdev") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_64x36_3.y4m --threads ${NTHREADS}";;
  "test") ARGS="--quiet --qp 20 --partitions b8x8,i4x4 --ref 5 --direct auto --b-pyramid --weightb --mixed-refs --no-fast-pskip --me umh --subme 7 --analyse b8x8,i4x4 -o eledream.264 ${BENCHPATH}/inputs/eledream_32x18_1.y4m --threads ${NTHREADS}";;
esac

if [ $VERSION = "omp4" ] || [ $VERSION = "omp3" ]; then

	export OMP_NUM_THREADS=${NTHREADS}

elif [ $VERSION = "serial" ]; then

	NTHREADS=1

elif [ $VERSION = "ompss" ] || [ $VERSION="ompss_instr" ]; then
	
	export OMP_NUM_THREADS=${NTHREADS}
	export NX_ARGS="${EXTRA_ARGS} --enable-yield --yields=${NTHREADS}" 

fi

${BENCHPATH}/bin/${BENCHMARK}-${VERSION} $ARGS 

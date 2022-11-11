#!/bin/bash
echo -e "\033[32mSetting up Compiler Flags\033[m"
PORTABILITY_FLAGS="-static-libgcc -Wl,--hash-style=both,--as-needed"
export CFLAGS=" -O3 -g -funroll-loops -fprefetch-loop-arrays ${PORTABILITY_FLAGS}"
export CXXFLAGS="-O3 -g -funroll-loops -fprefetch-loop-arrays -fpermissive -fno-exceptions ${PORTABILITY_FLAGS}"

if [ -z "${ROOT}" ]; then
    export ROOT=`pwd` #EDIT THIS IF YOU WANT TO SPECIFY ANOTHER ROOT DIR
fi
echo -e "\033[32mSetting root path ROOT=${ROOT}\033[m"

export PARSEC_LIBS=${ROOT}/parseclibs

echo -e "\033[32mEnviroment Ready!\033[m"

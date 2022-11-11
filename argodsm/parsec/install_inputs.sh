#!/bin/bash

INPUT_ARCHIVE=$1
APPS='blackscholes  bodytrack  facesim  ferret  fluidanimate  freqmine  x264'
KERNELS='canneal dedup'

echo -e "\033[32mUnpacking inputs files...\033[m"
tar xvf ${INPUT_ARCHIVE}

echo -e "\033[32mCopying input files...\033[m"
for app in $APPS; do

	cp -rf ./parsec-3.0/pkgs/apps/${app}/inputs ./${app}/

done
 
for kernel in $KERNELS; do

	cp -rf ./parsec-3.0/pkgs/kernels/${kernel}/inputs ./${kernel}/

done
 

echo -e "\033[32mDone\033[m"


rm -rf parsec-3.0

#!/bin/bash

originpath=${PWD}
for i in ${1}; 
do
    cd $ROOT/parseclibs/$i
    ./build.sh
    cd $originpath
done

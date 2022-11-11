
#!/bin/bash

cd src

make distclean

export LIBS="${LIBS} -lm"

./configure --prefix=${ROOT}/parseclibs/gsl --disable-shared

make

make install

make distclean

cd ..

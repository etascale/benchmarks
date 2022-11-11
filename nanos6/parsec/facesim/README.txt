Facesim Benchmark Readme
------------------------
This Benchmark does a realistic animation of a human face by simulating the 
underlying physics. It employs a Newton Raphson method for solving a system 
of partial differential equations.

TABLE OF CONTENTS
1. Compiling
2. Executing


1. Compiling
------------
It has different implementations available: serial, Pthreads, OmpSs, OmpSs using
the "for" (#pragma omp for) worksharing, OmpSs hybrid, OpenMP using tasks (#pragma 
omp task), OpenMP using the "for" worksharing and also an OpenMP hybrid version. 
The hybrid versions of both OmpSs and OpenMP use tasks except in the conjugate 
gradient of the Newton Raphson method, where the "for" worksharing is used.

To compile any of the versions available, the environment variable ${ROOT} has to 
be exported. ${ROOT} has to point to the directory where "facesim" benchmark 
directory resides. For example, if "facesim" directory is in ${HOME}/parsec-benchmarks 
directory, ${ROOT} has to be exported like follows, in a BASH shell:

export ROOT=${HOME}/parsec-benchmarks

With ${ROOT} already exported, the environment is ready for compilation. To 
compile any of the versions previously mentioned, enter the "src" directory and start
compilation with

make version={version string}

In the place of {version string} one of the following strings has to be used:
* serial    -> make version=serial
* pthreads  -> make version=pthreads
* ompss     -> make version=ompss
* ompss-ws  -> make version=ompss-ws
* ompss-hyb -> make version=ompss-hyb
* omp       -> make version=omp
* omp-ws    -> make version=omp-ws
* omp-hyb   -> make version=omp-hyb

To put the binary on the "bin" folder, run 

make version={version string} install

That will put a binary named "facesim-${version string}" on the "bin" folder.
If you do not run the "install", a binary named "facesim" will be left on 
"Benchmarks/facesim" folder, and it will be overwritten when you compile another 
version.

The runtime and compiler requirements for OmpSs are nanox 0.7.4 or greater and 
Mercurium 1.99.6 or greater. OmpSs has been used with GCC 4.3 and 4.9.1. 
To compile the OpenMP version you need GCC 4.9.1 or 
later as for tasks it uses the new dependency clauses added in OpenMP 4.0. 
For the worksharing version, compilers supporting OpenMP prior to 4.0 can be used.

If you want to build a different version, remember to first do a clean:

make version={version string} clean

If you want to use Intel C++ compiler as backend for Mercurium, add the flags 
"--cxx=icpc --Wn,-cxxlib,-Kc++" to CXXFLAGS and "--ld=icpc --cxx=icpc --Wl,-cxxlib --Wn,-cxxlib" 
to LINK_FLAGS. The latter preferably in src/Public_Library/Makefile.common, when checking for 
any of the ompss versions in OMPSSLIST and/or OMPSSILIST variables.


2. Executing
------------
To execute facesim, do as follows:

./facesim-{version string} -lastframe {number of frames} -timing -inputdir {input directory} -outputdir {output directory} -threads {number of partitions}

"-lastframe" tells Facesim how many frames has to simulate. There is a limit of frames
to simulate established by the input. Currently no more than 2013 frames can be simulated.
In the context of the PARSEC Benchmark Suite, "-lastframe" defines the input size:

  Native input:		100 frames (-lastframe 100)	
  Simlarge input:	1 frame (-lastframe 1)
  Simmedium input:	1 frame (-lastframe 1)
  Simsmall input:	1 frame (-lastframe 1)
  Simdev input:		1 frame (-lastframe 1)
  Test input:		1 frame (-lastframe 1)

"-timing" tells Facesim to print timings. Please note that in the case of OmpSs 
and OpenMP tasks versions the correct times are SIMULATION, FRAME and CGI. 
That is because the times are being logged by the task creation thread which 
does not stop creating tasks at time logging points, so times from one section 
might be accumulated in another section, depending on where the barriers have 
been put.

"-inputdir" specifies a directory where to find the directory containing the face data (Face_Data).

"-outputdir" specifies the directory to which facesim has to write the output.

"-threads" tells Facesim the partitioning scheme of the face model to use. In the case of Pthreads, this 
parameter also defines the number of threads to use. In the case of Serial no more than 1 partition 
can be used. In the case of OmpSs and OpenMP, the number of partitions can be any of the available. 
The number of threads is set by the environment variable OMP_NUM_THREADS.
The available quantities of partitions (there is a 3D model for each quantity) are 
1,2,3,4,6,8,16,32,64 and 128. In the case of Pthreads that means Facesim can only run with those 
number of threads (and partitions done to the face model). OpenMP and OmpSs can run with any 
number of OMP_NUM_THREADS with any available quantities of partitions.

Example:

./facesim-pthreads -lastframe 100 -timing -inputdir ../inputs -outputdir Storytelling/output -threads 8

The above example, asuming it has been executed inside the "bin" directory, tells Facesim to simulate 
100 frames, measure times and the "Face_Data" directory with the input is in "../inputs". The output files 
will be put into "Storytelling/output" directory.. If the output directory does not exist, Facesim 
creates it.
The version of facesim used is the pthreads one.

This benchmark solves the online clusterization problem

Table of Contents
-----------------

1. Compiling
2. Executing


1. Compiling
------------
There are four versions: serial, pthreads, ompss and Intel Thread Building Blocks.
To compile it manually, from the src directory run make version={version string}, 
where {version string} can be:
* serial	-> make version=serial
* pthreads	-> make version=pthreads
* ompss		-> make version=ompss
* tbb		-> make version=tbb

Then, run make version={version string} install 
to put the binary in the "bin" directory so successive compilations do not overwrite 
the binary generated in "src" directory.
To compile another version is recommended to do first a make clean.

Streamcluster has been tested with Nanox 0.7.4 and mcxx 1.99.6 in the case of OmpSs 
version, with GCC 4.9.1. GCC 4.9.1 has also been employed for Pthreads and Intel TBB 
versions.


2. Executing
------------
To execute any version of streamcluster you can use one of the following series of 
arguments:


  Native input:		10 20 128 1000000 200000 5000 none output.txt ${NDIVS}
  Simlarge input:	10 20 128 16384 16384 1000 none output.txt ${NDIVS}
  Simmedium input:	10 20 64 8192 8192 1000 none output.txt ${NDIVS}
  Simsmall input:	10 20 32 4096 4096 1000 none output.txt ${NDIVS}
  Simdev input:		3 10 3 16 16 10 none output.txt ${NDIVS}
  Test input:		2 5 1 10 10 5 none output.txt ${NDIVS}


NDIVS indicates how many partitons do to the block of points being processed.

Pthreads and Intel TBB versions use that paremeter to also establish the number of threads 
to use, doing a mapping of 1 partition to 1 thread.
OmpSs version has the number of threads independent of that parameter. Threads 
are set with OMP_NUM_THREADS environment variable as usual. Then NDIVS serves to indicate 
how many partitions you want to use. Having more partitions than threads can offer 
better load balance in executions with more than one NUMA node.

The output.txt is the output file.

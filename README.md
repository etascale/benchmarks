# The PARSEC Parallel Benchmark suite in C/C++

The original repository can be found [here](https://pm.bsc.es/gitlab/benchmarks/parsec-ompss).

## Implementations

**serial** - Sequential version of the code.

**pthreads** - Parallel Pthread implementation.

**omp2** - Parallel OpenMP 2.0 implementation.

**omp4** - Parallel OpenMP 3.0/4.0 implementation.

**ompss** - Parallel OmpSs implementation.

**ompss-2** - Parallel OmpSs-2 implementation.

**ompss-2-cluster-strong** - Parallel OmpSs-2@Cluster implementation (strong dependencies).

**ompss-2-cluster-weak** - Parallel OmpSs-2@Cluster implementation (weak dependencies).

**argo** - Parallel ArgoDSM/OpenMP implementation.

## Benchmarks

List of available benchmarks in the PARSEC suite:

```sh
blackscholes  - granularity: coarse, sharing: low , exchange: low
bodytrack     - granularity: medium, sharing: high, exchange: high
canneal       - granularity: fine  , sharing: high, exchange: high
dedup         - granularity: medium, sharing: high, exchange: high
facesim       - granularity: coarse, sharing: low , exchange: medium
ferret        - granularity: coarse, sharing: high, exchange: high
fluidanimate  - granularity: fine  , sharing: low , exchange: medium
freqmine      - granularity: medium, sharing: high, exchange: medium
streamcluster - granularity: medium, sharing: low , exchange: medium
swaptions     - granularity: coarse, sharing: low , exchange: low
x264          - granularity: coarse, sharing: high, exchange: high
```

## Software Requirements

**argo** - Assumes you have already installed [argodsm](https://github.com/etascale/argodsm).

**ompss-2(-cluster)** - Assumes you have already installed [mercurium](https://github.com/bsc-pm/mcxx) and [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster).

## Building 

The benchmarks can (also) be build manually. For most benchmarks, blackscholes, canneal, dedup, facesim, ferret, fluidanimate, freqmine, and swaptions, compilation and installation is done with the following commands:

To build:
```sh
$ make version=<serial | pthreads | omp2 | omp4 | ompss | ompss-2 | ompss-2-cluster-strong | ompss-2-cluster-weak | argo>
```

To install:
```sh
$ make version=<serial | pthreads | omp2 | omp4 | ompss | ompss-2 | ompss-2-cluster-strong | ompss-2-cluster-weak | argo> install
```

To clean:
```sh
$ make version=<serial | pthreads | omp2 | omp4 | ompss | ompss-2 | ompss-2-cluster-strong | ompss-2-cluster-weak | argo> clean
```

## Executing

To execute:
```sh
$ (mpirun $OMPIFLAGS) .(/bin)/blackscholes-ompss-2(-cluster-*)
```

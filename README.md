# Toy ArgoDSM@OmpSs benchmarks

Contains toy benchmarks that are used to evaluate the performance of ArgoDSM@OmpSs.

Current benchmarks in C include:
1. [daxpy](./c_bench/daxpy/)
2. [matvec](./c_bench/matvec/)
3. [fibonacci](./c_bench/fibonacci/)

Current benchmarks in C++ include:
1. [himeno](./cpp_bench/himeno/)
2. [matmul](./cpp_bench/matmul/)
3. [stream](./cpp_bench/stream/)

## Building

Build the applications using `cmake` (version `2.8`).

```shell
mkdir build
cd build
CC=mcc CXX=mcxx cmake .. -DCMAKE_INSTALL_PREFIX=<target_installation_dir> -DCMAKE_BUILD_TYPE=Release
make install
```

This assumes you have already installed [mercurium](https://github.com/bsc-pm/mcxx) and [nanos6-cluster](https://github.com/bsc-pm/nanos6-cluster).

## Benchmarks

### **daxpy**

Performs the computation: `y += a * x` where `x`, `y`, are two vectors of `double`, of size `N` and `a` is a scalar.

#### **[Usage]**

```sh
mpirun $OMPIFLAGS ./daxpy-(strong/weak) N TS ITER [CHECK]

where:

N       the size of the vectors x and y
TS      the number of vector elements each leaf task will compute
ITER    number of iterations for each to execute the computation
CHECK   optional parameter that enables checks to make sure the comptuation is correct
```

### **matvec**

Calculates the matrix-vector product: `y = A * x`, where `A` is a matrix of `M` rows and `N` columns.

#### **[Usage]**

```sh
mpirun $OMPIFLAGS ./matvec-(strong/weak) M N TS ITER [CHECK]

where:

M       the rows of matrix A
N       the columns of matrix A
TS      the number of rows of matrix A each leaf task wil compute
ITER    number of iterations for which to execute the computations
CHECK   optional parameter that enables checks to make sure the comptuation is correct
```

### **fibonacci**

Calculates the n<sup>th</sup> fibonacci number.

#### **[Usage]**

```sh
mpirun $OMPIFLAGS ./fibonacci N [CHECK]

where:

N       the fibonacci number to calculate
CHECK   optional parameter that enables checks to make sure the computation is correct
```

### **himeno**

Measures the speed of major loops for solving Poisson’s equation using the Jacobi iteration method.

#### **[Usage]**

```sh
(mpirun $OMPIFLAGS) ./himeno-ompss-2(-cluster-(strong/weak)) TS (PS) ([CHECK])

where:

TS      task granularity
(PS)    the problem size set through CMakeLists.txt
(CHECK) optional output to a file for verification set through CMakeLists.txt
```

### **matmul**

Calculates the product of two matrices.

#### **[Usage]**

```sh
(mpirun $OMPIFLAGS) ./matmul-ompss-2(-cluster-(strong/weak)) -b TS [-v]

where:

TS      task granularity
-v      optional parameter to enable verification
```

### **stream**

A set of multiple kernel operations, that is, sequential accesses over array data with simple arithmetic. \
Kernel operations include:

+ `Copy`&nbsp;&nbsp;&nbsp;&nbsp;-> a(i) = b(i)
+ `Scale`&nbsp;&nbsp;-> a(i) = q*b(i)
+ `Add`&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> a(i) = b(i) + c(i)
+ `Triad`&nbsp;&nbsp;-> a(i) = b(i) = q*c(i)

#### **[Usage]**

```sh
(mpirun $OMPIFLAGS) ./stream-ompss-2(-cluster-(strong/weak)) TS (STREAM_ARRAY_SIZE) ([TEAMINIT])

where:

TS                task granularity
STREAM_ARRAY_SIZE the size of the working arrays set through CMakeLists.txt
TEAMINIT          optional parameter to disable/enable team process initialization on the arrays set through CMakeLists.txt
```

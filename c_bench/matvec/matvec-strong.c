/**
 * @file
 * @brief An OmpSs-2@Cluster implementation of matrix-vector multiplication.
 * @note The implementation uses strong dependencies only.
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <nanos6/debug.h>

#include "memory.h"

void
mult_vector(size_t M, double *A, size_t N, double *x, double *y)
{
	for (size_t i = 0; i < M; ++i) {
		double res = 0.0;
		for (size_t j = 0; j < N; ++j) {
			res += A[i * N + j] * x[j];
		}
		y[i] += res;
	}
}

void
init_vector(size_t M, double *vec, double value)
{
	for (size_t i = 0; i < M; ++i) {
		vec[i] = value;
	}
}

void
check_result(size_t M, double *A, size_t N, double *x, double *y, size_t ITER)
{
	double *y_serial = lmalloc_double(M);
	init_vector(M, y_serial, 0);
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		mult_vector(M, A, N, x, y_serial);
	}
	
	for (size_t i = 0; i < M; ++i) {
		if (y_serial[i] != y[i]) {
			printf("FAILED\n");
			lfree_double(y_serial, M);
			return;
		}
	}
	
	printf("SUCCESS\n");
	lfree_double(y_serial, M);
}

void
usage()
{
	fprintf(stderr, "usage: matvec_strong M N TS ITER [CHECK]\n");
	return;
}

int
main(int argc, char *argv[])
{
	bool check = false;
	double *A, *x, *y;
	size_t M, N, TS, ITER;
	struct timespec tp_start, tp_end;

	if (argc != 5 && argc != 6) {
		usage();
		return -1;
	}
	
	M     = atoi(argv[1]);
	N     = atoi(argv[2]);
	TS    = atoi(argv[3]);
	ITER  = atoi(argv[4]);
	check = (argc == 6) ? atoi(argv[5]) : false;
	
	if (M % TS) {
		fprintf(stderr, "The task-size needs to divide the number of rows\n");
		return -1;
	}
	
	A = dmalloc_double(M * N, nanos6_equpart_distribution, 0, NULL);
	x = dmalloc_double(N    , nanos6_equpart_distribution, 0, NULL);
	y = dmalloc_double(M    , nanos6_equpart_distribution, 0, NULL);

	/* Don't offload to remote node the initialization of `y` */
	#pragma oss task out(y[0;M]) 				\
			 node(nanos6_cluster_no_offload)	\
			 label("master: initialize y")
	init_vector(M, y, 0);
	
	/* Don't offload to remote node the initialization of `x` */
	#pragma oss task out(x[0;N]) 				\
			 node(nanos6_cluster_no_offload)	\
			 label("master: initialize x")
	init_vector(N, x, 1);
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based row initialization of `A`
	 * ////////////////////////////////////////////////////// */
	/* Spawn sub-tasks and offload to remote */
	for (size_t i = 0; i < M; i += TS) {
		#pragma oss task out(A[i*N;TS*N]) \
				 label("remote: initialize chunk of rows in `A`")
		init_vector(TS*N, &A[i*N], 2);
	}
	#pragma oss taskwait
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	
	/* Timer: computation */
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based row multiplication y = A * x
	 * ////////////////////////////////////////////////////// */
	for (size_t iter = 0; iter < ITER; ++iter) {
		/* Spawn sub-tasks and offload to remote */
		for (size_t i = 0; i < M; i += TS) {
			#pragma oss task in(A[i*N;TS*N])	\
					 in(x[0;N]) 		\
					 inout(y[i;TS])		\
					 label("remote: calculate chunk of rows in `y`")
			mult_vector(TS, &A[i*N], N, x, &y[i]);
		}
	}
	#pragma oss taskwait
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	
	/* Timer: computation */
	clock_gettime(CLOCK_MONOTONIC, &tp_end);

	/* Don't offload to remote node the correctness check of `y` */
	if (check) {
		#pragma oss task in(A[0;M*N])				\
				 in(x[0;N]) 				\
				 in(y[0;M])				\
				 node(nanos6_cluster_no_offload)	\
				 label("master: correctness check")
		check_result(M, A, N, x, y, ITER);
		#pragma oss taskwait
	}
	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ ((double)(tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
	
	double mflops =
		ITER 			/* 'ITER' times of kernel FLOPS        */
		* 3 * M * N 		/* 3 operations for every element of A */
		/ (time_msec / 1000.0) 	/* time in seconds                     */
		/ 1e6; 			/* convert to Mega                     */

	printf("M:%zu N:%zu TS:%zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
		M, N, TS, ITER, nanos6_get_num_cluster_nodes(), nanos6_get_num_cpus(),
		time_msec, mflops);
	
	dfree_double(A, M * N);
	dfree_double(x, N);
	dfree_double(y, M);
	
	return 0;
}

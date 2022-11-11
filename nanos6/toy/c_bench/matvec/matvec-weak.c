/**
 * @file
 * @brief An OmpSs-2@Cluster implementation of matrix-vector multiplication.
 * @note The implementation uses weak and strong dependencies.
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <nanos6/debug.h>

#include "memory.h"

// undef: fetch tasks disabled
//   def: fetch tasks  enabled
// #define ENABLE_FETCH_TASKS

size_t
node_chunk(const size_t size, const size_t task_size)
{
	/* calc chunk */
	const size_t nodes = nanos6_get_num_cluster_nodes();
	const size_t chunk = size / nodes;
	/* make check */
	assert(task_size <= chunk);
	assert(chunk % task_size == 0);

	return chunk;
}

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
	
	y = dmalloc_double(M    , nanos6_equpart_distribution, 0, NULL);
	x = dmalloc_double(N    , nanos6_equpart_distribution, 0, NULL);
	A = dmalloc_double(M * N, nanos6_equpart_distribution, 0, NULL);
	
	/* Don't offload to remote node the initialization of `y` */
	// #pragma oss task out(y[0;M]) 			\
	// 		 node(nanos6_cluster_no_offload)	\
	// 		 label("master: initialize y")
	// init_vector(M, y, 0);

	/* ////////////////////////////////////////////////////////
	 * Chunk-based initialization of `y`
	 * ////////////////////////////////////////////////////// */
	/* Calculate chunk for each node */
	size_t chunk_per_node = node_chunk(M, TS);

	for (size_t i = 0; i < M; i += chunk_per_node) {
		/* Calculate the node to which the chunk belongs  */
		const int node_id = i / chunk_per_node;

		/* Spawn a task for the whole chunk and bind to `node` */
		#pragma oss task weakout(y[i;chunk_per_node]) 		\
				 firstprivate(i, chunk_per_node, TS)	\
				 node(node_id) 				\
				 label("remote: initialize chunk in `y`")
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task out(y[i;chunk_per_node])		\
					 node(nanos6_cluster_no_offload)	\
					 label("remote: fetch all necessary data at once")
			{
				// fetch all data in one go
			}
#endif

			/* Spawn sub-tasks and don't offload to remote */
			for (size_t j = i; j < i + chunk_per_node; j += TS) {
				#pragma oss task out(y[j;TS]) 			\
						 node(nanos6_cluster_no_offload)\
						 label("local: initialize chunk in `y`")
				init_vector(TS, &y[j], 0);
			}
		}
	}
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */

	/* Don't offload to remote node the initialization of `x` */
	// #pragma oss task out(x[0;N]) 			\
	// 		 node(nanos6_cluster_no_offload)	\
	// 		 label("master: initialize x")
	// init_vector(N, x, 1);

	/* ////////////////////////////////////////////////////////
	 * Chunk-based initialization of `x`
	 * ////////////////////////////////////////////////////// */
	/* Calculate chunk for each node */
	chunk_per_node = node_chunk(N, TS);

	for (size_t i = 0; i < N; i += chunk_per_node) {
		/* Calculate the node to which the chunk belongs  */
		const int node_id = i / chunk_per_node;

		/* Spawn a task for the whole chunk and bind to `node` */
		#pragma oss task weakout(x[i;chunk_per_node]) 		\
				 firstprivate(i, chunk_per_node, TS)	\
				 node(node_id) 				\
				 label("remote: initialize chunk in `x`")
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task out(x[i;chunk_per_node])		\
					 node(nanos6_cluster_no_offload)	\
					 label("remote: fetch all necessary data at once")
			{
				// fetch all data in one go
			}
#endif

			/* Spawn sub-tasks and don't offload to remote */
			for (size_t j = i; j < i + chunk_per_node; j += TS) {
				#pragma oss task out(x[j;TS]) 			\
						 node(nanos6_cluster_no_offload)\
						 label("local: initialize chunk in `x`")
				init_vector(TS, &x[j], 1);
			}
		}
	}
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based row initialization of `A`
	 * ////////////////////////////////////////////////////// */
	/* Calculate row chunk for each node */
	chunk_per_node = node_chunk(M, TS);

	for (size_t i = 0; i < M; i += chunk_per_node) {
		/* Calculate the node to which the chunk belongs  */
		const int node_id = i / chunk_per_node;

		/* Spawn a task for the whole chunk and bind to `node` */
		#pragma oss task weakout(A[i*N;chunk_per_node*N]) 	\
				 firstprivate(i, chunk_per_node, N, TS)	\
				 node(node_id) 				\
				 label("remote: initialize chunk of rows in `A`")
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task out(A[i*N;chunk_per_node*N])		\
					 node(nanos6_cluster_no_offload)	\
					 label("remote: fetch all necessary data at once")
			{
				// fetch all data in one go
			}
#endif

			/* Spawn sub-tasks and don't offload to remote */
			for (size_t j = i; j < i + chunk_per_node; j += TS) {
				#pragma oss task out(A[j*N;TS*N]) 			\
						 node(nanos6_cluster_no_offload)	\
						 label("local: initialize chunk of rows in `A`")
				init_vector(TS*N, &A[j*N], 2);
			}
		}
	}
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	#pragma oss taskwait
	
	/* Timer: computation */
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based row multiplication y = A * x
	 * ////////////////////////////////////////////////////// */
	for (size_t iter = 0; iter < ITER; ++iter) {
		for (size_t i = 0; i < M; i += chunk_per_node) {
			/* Calculate the node to which the chunk belongs  */
			const int node_id = i / chunk_per_node;

			/* Spawn a task for the whole chunk and bind to `node` */
			#pragma oss task weakin(A[i*N;chunk_per_node*N])	\
					 weakin(x[0;N])				\
					 weakinout(y[i;chunk_per_node]) 	\
					 firstprivate(i, chunk_per_node, N, TS)	\
					 node(node_id) 				\
					 label("remote: calculate chunk of rows in `y`")
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(A[i*N;chunk_per_node*N])		\
						 in(x[0;N])				\
						 inout(y[i;chunk_per_node])		\
						 node(nanos6_cluster_no_offload)	\
						 label("remote: fetch all necessary data at once")
				{
					// fetch all data in one go
				}
#endif

				/* Spawn sub-tasks and don't offload to remote */
				for (size_t j = i; j < i + chunk_per_node; j += TS) {
					#pragma oss task in(A[j*N;TS*N]) 			\
							 in(x[0;N]) 				\
							 inout(y[j;TS]) 			\
							 node(nanos6_cluster_no_offload)	\
							 label("local: calculate chunk of rows in `y`")
					mult_vector(TS, &A[j*N], N, x, &y[j]);
				}
			}
		}
	}
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	#pragma oss taskwait
	
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
	
	dfree_double(y, M);
	dfree_double(x, N);
	dfree_double(A, M * N);
	
	return 0;
}

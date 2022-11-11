/**
 * @file
 * @brief An OmpSs-2@Cluster implementation of daxpy BLAS operation.
 * @note The implementation uses strong dependencies only.
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
daxpy(size_t N, double *x, double alpha, double *y)
{
	for (size_t i = 0; i < N; ++i) {
		y[i] += alpha * x[i];
	}
}

void
init_vector(size_t N, double *vector, double value)
{
	for (size_t i = 0; i < N; ++i) {
		vector[i] = value;
	}
}

void
check_result(size_t N, double *x, double alpha, double *y, size_t ITER)
{
	double *y_serial = lmalloc_double(N);
	init_vector(N, y_serial, 0);
	
	for (size_t iter = 0; iter < ITER; ++iter) {
		daxpy(N, x, alpha, y_serial);
	}
	
	for (size_t i = 0; i < N; ++i) {
		if (y_serial[i] != y[i]) {
			printf("FAILED\n");
			lfree_double(y_serial, N);
			return;
		}
	}
	
	printf("SUCCESS\n");
	lfree_double(y_serial, N);
}

void
usage()
{
	fprintf(stderr, "usage: daxpy_strong N TS ITER [CHECK]\n");
	return;
}

int
main(int argc, char *argv[])
{
	bool check = false;
	double alpha, *x, *y;
	size_t N, TS, ITER;
	struct timespec tp_start, tp_end;
	
	if (argc != 4 && argc != 5) {
		usage();
		return -1;
	}
	
	N     = atol(argv[1]);
	TS    = atol(argv[2]);
	ITER  = atol(argv[3]);
	check = (argc == 5) ? atoi(argv[4]) : false;
	
	if (N % TS) {
		fprintf(stderr, "The task-size needs to divide the vector size\n");
		return -1;
	}
	
	x = dmalloc_double(N, nanos6_equpart_distribution, 0, NULL);
	y = dmalloc_double(N, nanos6_equpart_distribution, 0, NULL);
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based initialization of `y` & `x`
	 * ////////////////////////////////////////////////////// */
	/* Calculate row region for each node */
	size_t chunk_per_node = node_chunk(N, TS);

	for (size_t i = 0; i < N; i += chunk_per_node) {
		/* Calculate the node to which the chunk belongs  */
		const int node_id = i / chunk_per_node;

		/* Spawn a task for the whole chunk and bind to `node` */
		#pragma oss task weakout(y[i;chunk_per_node]) 		\
				 weakout(x[i;chunk_per_node])		\
				 firstprivate(i, chunk_per_node, TS)	\
				 node(node_id) 				\
				 label("remote: initialize row region in `y` & `x`")
		{
#ifdef ENABLE_FETCH_TASKS
			#pragma oss task out(y[i;chunk_per_node]) 		\
					 out(x[i;chunk_per_node])		\
					 node(nanos6_cluster_no_offload)	\
					 label("remote: fetch all necessary data at once")
			{
				// fetch all data in one go
			}
#endif

			/* Spawn sub-tasks and don't offload to remote */
			for (size_t j = i; j < i + chunk_per_node; j += TS) {
				#pragma oss task out(y[j;TS]) 				\
						 out(x[j;TS])				\
						 node(nanos6_cluster_no_offload)	\
						 label("local: initialize row region in `y` & `x`")
				{
					init_vector(TS, &y[j], 0);
					init_vector(TS, &x[j], 42);
				}
			}
		}
	}
	#pragma oss taskwait
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	
	/* Timer: computation */
	clock_gettime(CLOCK_MONOTONIC, &tp_start);
	
	/* ////////////////////////////////////////////////////////
	 * Chunk-based daxpy BLAS operation
	 * ////////////////////////////////////////////////////// */
	for (size_t iter = 0; iter < ITER; ++iter) {
		for (size_t i = 0; i < N; i += chunk_per_node) {
			/* Calculate the node to which the chunk belongs  */
			const int node_id = i / chunk_per_node;

			/* Spawn a task for the whole chunk and bind to `node` */
			#pragma oss task weakin(x[i;chunk_per_node]) 		\
					 weakinout(y[i;chunk_per_node])		\
					 firstprivate(i, chunk_per_node, TS)	\
					 node(node_id) 				\
					 label("remote: calculate row region in `y`")
			{
#ifdef ENABLE_FETCH_TASKS
				#pragma oss task in(x[i;chunk_per_node]) 		\
						 inout(y[i;chunk_per_node])		\
						 node(nanos6_cluster_no_offload)	\
						 label("remote: fetch all necessary data at once")
				{
					// fetch all data in one go
				}
#endif

				/* Spawn sub-tasks and don't offload to remote */
				for (size_t j = i; j < i + chunk_per_node; j += TS) {
					#pragma oss task in(x[j;TS]) 				\
							 inout(y[j;TS])				\
							 node(nanos6_cluster_no_offload)	\
							 label("remote: calculate row region in `y`")
					daxpy(TS, &x[j], alpha, &y[j]);
				}
			}			 
		}
	}
	#pragma oss taskwait
	/* ////////////////////////////////////////////////////////
	 * ////////////////////////////////////////////////////// */
	
	/* Timer: computation */
	clock_gettime(CLOCK_MONOTONIC, &tp_end);
	
	/* Don't offload to remote node the correctness check of `y` */
	if (check) {
		#pragma oss task in(x[0;N]) 				\
				 inout(y[0;N]) 				\
				 node(nanos6_cluster_no_offload)	\
				 label("master: correctness check")
		check_result(N, x, alpha, y, ITER);
		#pragma oss taskwait
	}
	
	double time_msec = (tp_end.tv_sec - tp_start.tv_sec) * 1e3
		+ ((tp_end.tv_nsec - tp_start.tv_nsec) * 1e-6);
	
	double mflops =
		ITER 			/* 'ITER' times of kernel FLOPS          */
		* 3 * N 		/* 3 operations for every vector element */
		/ (time_msec / 1000.0) 	/* time in seconds                       */
		/ 1e6;			/* convert to Mega                       */
	
	printf("N:%zu TS:%zu ITER:%zu NR_PROCS:%d CPUS:%d TIME_MSEC:%.2lf MFLOPS:%.2lf\n",
		N, TS, ITER, nanos6_get_num_cluster_nodes(), nanos6_get_num_cpus(),
		time_msec, mflops);
	
	dfree_double(x, N);
	dfree_double(y, N);

	return 0;
}

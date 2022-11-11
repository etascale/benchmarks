//HJM_Securities.cpp
//Routines to compute various security prices using HJM framework (via Simulation).
//Authors: Mark Broadie, Jatin Dewanwala
//Collaborator: Mikhail Smelyanskiy, Jike Chong, Intel
//OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis - Barcelona Supercomputing Center
//ArgoDSM/OpenMP version written by Ioannis Anevlavis - Eta Scale AB
//OmpSs-2 version written by Ioannis Anevlavis - Eta Scale AB
//OmpSs-2@Cluster version written by Ioannis Anevlavis - Eta Scale AB

#ifdef ENABLE_ARGO
#include "argo.hpp"
#endif

#include <omp.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "nr_routines.h"
#include "HJM.h"
#include "HJM_Securities.h"
#include "HJM_type.h"


#ifdef ENABLE_THREADS
#include <pthread.h>
#define MAX_THREAD 1024
#endif //ENABLE_THREADS

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Macro for only node0 to do stuff
#define MAIN_PROC(rank, inst) \
do { \
	if ((rank) == 0) { inst; } \
} while (0)

int BSIZE = BSIZE_UNIT;
int NUM_TRIALS = DEFAULT_NUM_TRIALS;
int BLOCK_SIZE = DEFAULT_BLOCK_SIZE;
int nThreads = 1;
int nSwaptions = 1;
int iN = 11; 
FTYPE dYears = 5.5; 
int iFactors = 3; 
parm *swaptions;

long seed = 1979; //arbitrary (but constant) default value (birth year of Christian Bienia)
long swaption_seed;

// =================================================
FTYPE *dSumSimSwaptionPrice_global_ptr;
FTYPE *dSumSquareSimSwaptionPrice_global_ptr;
int chunksize;

int numtasks;
int workrank;

static bool *gflag;
static long *gseed;

void distribute(int& beg, int& end, const int& loop_size,
		const int& beg_offset, const int& less_equal)
{
	int chunk = loop_size / numtasks;
	beg = workrank * chunk + ((workrank == 0) ? beg_offset : less_equal);
	end = (workrank != numtasks - 1) ? workrank * chunk + chunk : loop_size;
}

void task_chunk(int& beg, int& end, int& chunk,
		const int& size, const int& index, const int& bsize)
{
	chunk = (size - index > bsize) ? bsize : size - index;
	beg = index;
	end = index + chunk;
}

#if defined(ENABLE_OMPSS_2_CLUSTER)
void node_chunk(int& node_id, int& chunk,
		const int& size, const int& index, const int& bsize)
{
	static const int nodes = nanos6_get_num_cluster_nodes();
	chunk = (size - index > bsize) ? bsize : size - index;
	node_id = ((index / chunk) < nodes) ? index / chunk : nodes-1;
}
#endif

void write_to_file()
{
	int i, rv;
	FILE *file;
	char *outputFile = (char*)"out.swaptions";

	file = fopen(outputFile, "w");
	if(file == NULL) {
		printf("ERROR: Unable to open file `%s'.\n", outputFile);
		exit(1);
	}
	for (i = 0; i < nSwaptions; i++) {
		rv = fprintf(file,"Swaption%d: [SwaptionPrice: %.10lf StdError: %.10lf] \n", 
				i, swaptions[i].dSimSwaptionMeanPrice, swaptions[i].dSimSwaptionStdError);
		if(rv < 0) {
			printf("ERROR: Unable to write to file `%s'.\n", outputFile);
			fclose(file);
			exit(1);
		}
	}
	rv = fclose(file);
	if(rv != 0) {
		printf("ERROR: Unable to close file `%s'.\n", outputFile);
		exit(1);
	}
}

void * worker(void *arg)
#if defined(ENABLE_OMPSS) || defined(ENABLE_OMPSS_2) || defined(ENABLE_OMPSS_2_CLUSTER)
{
	int iSuccess;
	FTYPE pdSwaptionPrice[2];

#if defined(ENABLE_OMPSS)
	for(int i = 0; i < nSwaptions; i += BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, nSwaptions, i, BSIZE);

		#pragma omp task inout(swaptions[beg:end-1])		\
				 private(iSuccess, pdSwaptionPrice)	\
				 firstprivate(beg, end,			\
				 	      NUM_TRIALS, BLOCK_SIZE)
#elif defined(ENABLE_OMPSS_2)
	for(int i = 0; i < nSwaptions; i += BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, nSwaptions, i, BSIZE);

		#pragma oss task inout(swaptions[beg:end-1])		\
				 private(iSuccess, pdSwaptionPrice)	\
				 firstprivate(beg, end,			\
				 	      NUM_TRIALS, BLOCK_SIZE)
#elif defined(ENABLE_OMPSS_2_CLUSTER)
#ifdef ENABLE_WEAK
	int generic_chunk_nSwaptions = nSwaptions / nanos6_get_num_cluster_nodes();

	for(int z = 0; z < nSwaptions; z += generic_chunk_nSwaptions) {
		int node_id, chunk_per_node_nSwaptions;
		node_chunk(node_id, chunk_per_node_nSwaptions, nSwaptions, z, generic_chunk_nSwaptions);

		#pragma oss task weakinout(swaptions[z;chunk_per_node_nSwaptions])	\
				 private(iSuccess, pdSwaptionPrice)			\
				 firstprivate(z, chunk_per_node_nSwaptions,		\
				 	      BSIZE, NUM_TRIALS, BLOCK_SIZE)		\
				 node(node_id)
		{

			#pragma oss task inout(swaptions[z;chunk_per_node_nSwaptions])	\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}

			for(int i = z; i < z+chunk_per_node_nSwaptions; i += BSIZE) {
				int beg, end, chunk;
				task_chunk(beg, end, chunk, z+chunk_per_node_nSwaptions, i, BSIZE);

				#pragma oss task inout(swaptions[beg:end-1])		\
						 private(iSuccess, pdSwaptionPrice)	\
						 firstprivate(beg, end,			\
						 	      NUM_TRIALS, BLOCK_SIZE)	\
						 node(nanos6_cluster_no_offload)
#else
	for(int i = 0; i < nSwaptions; i += BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, nSwaptions, i, BSIZE);

		#pragma oss task inout(swaptions[beg:end-1])		\
				 private(iSuccess, pdSwaptionPrice)	\
				 firstprivate(beg, end,			\
				 	      NUM_TRIALS, BLOCK_SIZE)
#endif
#endif
		{
			for (int i = beg; i < end; i++) {
				iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
						swaptions[i].dCompounding, swaptions[i].dMaturity, 
						swaptions[i].dTenor, swaptions[i].dPaymentInterval,
						swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
						swaptions[i].pdYield, swaptions[i].ppdFactors,
						100, NUM_TRIALS, BLOCK_SIZE, 0);
				assert(iSuccess == 1);
				swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
				swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
			}
		}
	}
#if defined(ENABLE_OMPSS)
	#pragma omp taskwait
#elif defined(ENABLE_OMPSS_2)
	#pragma oss taskwait
#elif defined(ENABLE_OMPSS_2_CLUSTER)
#ifdef ENABLE_WEAK
		}
	}
#endif
	#pragma oss taskwait
#endif

	return NULL;
}
#elif defined(ENABLE_OMP4)
{
	#pragma omp parallel 
	{
		#pragma omp single 
		{
			int iSuccess;
			FTYPE pdSwaptionPrice[2];
			for(int i=0; i < nSwaptions; i++) {
				#pragma omp task firstprivate(i) private(iSuccess, pdSwaptionPrice) //depend(inout: swaptions[i])
				#pragma omp task firstprivate(i) private(iSuccess, pdSwaptionPrice) depend(inout: swaptions[i])
				{
					iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
							swaptions[i].dCompounding, swaptions[i].dMaturity, 
							swaptions[i].dTenor, swaptions[i].dPaymentInterval,
							swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
							swaptions[i].pdYield, swaptions[i].ppdFactors,
							100, NUM_TRIALS, BLOCK_SIZE, 0);
					assert(iSuccess == 1);
					swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
					swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
				}
			}
			#pragma omp taskwait
		} //end of single  
	} //end of parallel region
	return NULL;    
}
#elif defined(ENABLE_OMP2)
{
	#pragma omp parallel 
	{
		int iSuccess;
		FTYPE pdSwaptionPrice[2];
		#pragma omp for schedule(SCHED_POLICY)
		for(int i=0; i < nSwaptions; i++) {
			iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
					swaptions[i].dCompounding, swaptions[i].dMaturity, 
					swaptions[i].dTenor, swaptions[i].dPaymentInterval,
					swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
					swaptions[i].pdYield, swaptions[i].ppdFactors,
					100, NUM_TRIALS, BLOCK_SIZE, 0);
			assert(iSuccess == 1);
			swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
			swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
		}
		#pragma omp taskwait
	} //end of parallel region
	return NULL;    
}
#elif defined(ENABLE_ARGO)
{
	int beg, end;
	distribute(beg, end, nSwaptions, 0, 0);

	#pragma omp parallel 
	{
		int iSuccess;
		FTYPE pdSwaptionPrice[2];
		#pragma omp for schedule(SCHED_POLICY)
		for(int i=beg; i < end; i++) {
			iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
					swaptions[i].dCompounding, swaptions[i].dMaturity, 
					swaptions[i].dTenor, swaptions[i].dPaymentInterval,
					swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
					swaptions[i].pdYield, swaptions[i].ppdFactors,
					100, NUM_TRIALS, BLOCK_SIZE, 0);
			assert(iSuccess == 1);
			swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
			swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
		}
	} //end of parallel region
	argo::barrier();

	return NULL;    
}
#else
{
	int tid = *((int *)arg);
	FTYPE pdSwaptionPrice[2];

	int beg, end, chunksize;
	if (tid < (nSwaptions % nThreads)) {
		chunksize = nSwaptions/nThreads + 1;
		beg = tid * chunksize;
		end = (tid+1)*chunksize;
	} else {
		chunksize = nSwaptions/nThreads;
		int offsetThread = nSwaptions % nThreads;
		int offset = offsetThread * (chunksize + 1);
		beg = offset + (tid - offsetThread) * chunksize;
		end = offset + (tid - offsetThread + 1) * chunksize;
	}

	if(tid == nThreads -1 )
		end = nSwaptions;

	for(int i=beg; i < end; i++) {
		int iSuccess = HJM_Swaption_Blocking(pdSwaptionPrice,  swaptions[i].dStrike, 
				swaptions[i].dCompounding, swaptions[i].dMaturity, 
				swaptions[i].dTenor, swaptions[i].dPaymentInterval,
				swaptions[i].iN, swaptions[i].iFactors, swaptions[i].dYears, 
				swaptions[i].pdYield, swaptions[i].ppdFactors,
				swaption_seed+i, NUM_TRIALS, BLOCK_SIZE, 0);
		assert(iSuccess == 1);
		swaptions[i].dSimSwaptionMeanPrice = pdSwaptionPrice[0];
		swaptions[i].dSimSwaptionStdError = pdSwaptionPrice[1];
	}

	return NULL;
}
#endif // ENABLE_OMPSS || ENABLE_OMPSS_2 || ENABLE_OMPSS_2_CLUSTER

//print a little help message explaining how to use this program
void print_usage(char *name) {
	fprintf(stderr,"Usage: %s OPTION [OPTIONS]...\n", name);
	fprintf(stderr,"Options:\n");
	fprintf(stderr,"\t-ns [number of swaptions (should be > number of threads]\n");
	fprintf(stderr,"\t-sm [number of simulations]\n");
	fprintf(stderr,"\t-nt [number of threads]\n");
	fprintf(stderr,"\t-sd [random number seed]\n");
	fprintf(stderr,"\t-bs [task block size]\n");
}

//Please note: Whenever we type-cast to (int), we add 0.5 to ensure that the value is rounded to the correct number. 
//For instance, if X/Y = 0.999 then (int) (X/Y) will equal 0 and not 1 (as (int) rounds down).
//Adding 0.5 ensures that this does not happen. Therefore we use (int) (X/Y + 0.5); instead of (int) (X/Y);

int main(int argc, char *argv[])
{
#ifdef ENABLE_ARGO
	argo::init(10*1024*1024*1024UL);
	
	workrank = argo::node_id();
	numtasks = argo::number_of_nodes();
#else
	workrank = 0;
#endif

	int i,j;
	int beg, end;
	int iSuccess = 0;

	FTYPE **factors=NULL;

	//****bench begins****//

	MAIN_PROC(workrank, printf("PARSEC Benchmark Suite\n"));
	MAIN_PROC(workrank, fflush(NULL));


#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_swaptions);
#endif

	if(argc == 1)
	{
		MAIN_PROC(workrank, print_usage(argv[0]));
		exit(1);
	}

#if defined(ENABLE_OMPSS) || defined(ENABLE_OMPSS_2) || defined(ENABLE_OMPSS_2_CLUSTER) || defined(ENABLE_OMP2) || defined(ENABLE_OMP4) || defined(ENABLE_ARGO)
	MAIN_PROC(workrank, printf("Warning! Argumetn -nt is ignored, use NX_ARGS for OMPSs or OMP_NUM_THREADS for OpenMP 4.0\n"));
#endif

	for (int j=1; j<argc; j++) {
		if (!strcmp("-sm", argv[j])) {NUM_TRIALS = atoi(argv[++j]);}
		else if (!strcmp("-bs", argv[j])) {BSIZE = atoi(argv[++j]);}
		else if (!strcmp("-nt", argv[j])) {nThreads = atoi(argv[++j]);} 
		else if (!strcmp("-ns", argv[j])) {nSwaptions = atoi(argv[++j]);}
		else if (!strcmp("-sd", argv[j])) {seed = atoi(argv[++j]);}
		else {
			MAIN_PROC(workrank, fprintf(stderr,"Error: Unknown option: %s\n", argv[j]));
			MAIN_PROC(workrank, print_usage(argv[0]));
			exit(1);
		}
	}

	if(nSwaptions < nThreads) {
		MAIN_PROC(workrank, fprintf(stderr,"Error: Fewer swaptions than threads.\n"));
		MAIN_PROC(workrank, print_usage(argv[0]));
		exit(1);
	}

	MAIN_PROC(workrank, printf("Number of Simulations: %d, Number of threads: %d, Number of swaptions: %d, Task block size: %d\n", NUM_TRIALS, nThreads, nSwaptions, BSIZE));
	swaption_seed = (long)(2147483647L * RanUnif(&seed));

#if defined(ENABLE_THREADS)
	pthread_t      *threads;
	pthread_attr_t  pthread_custom_attr;

	if ((nThreads < 1) || (nThreads > MAX_THREAD))
	{
		fprintf(stderr,"Number of threads must be between 1 and %d.\n", MAX_THREAD);
		exit(1);
	}
	threads = (pthread_t *) malloc(nThreads * sizeof(pthread_t));
	pthread_attr_init(&pthread_custom_attr);
#elif defined(ENABLE_OMPSS) || defined(ENABLE_OMPSS_2) || defined(ENABLE_OMPSS_2_CLUSTER) || defined(ENABLE_OMP2) || defined(ENABLE_OMP4) || defined(ENABLE_ARGO)
	//ignore number of threads
	nThreads = nSwaptions;
#else
	if (nThreads != 1)
	{
		fprintf(stderr,"Number of threads must be 1 (serial version)\n");
		exit(1);
	}
#endif //ENABLE_THREADS

	// initialize input dataset
	factors = dmatrix(0, iFactors-1, 0, iN-2);
	//the three rows store vol data for the three factors
	factors[0][0]= .01;
	factors[0][1]= .01;
	factors[0][2]= .01;
	factors[0][3]= .01;
	factors[0][4]= .01;
	factors[0][5]= .01;
	factors[0][6]= .01;
	factors[0][7]= .01;
	factors[0][8]= .01;
	factors[0][9]= .01;

	factors[1][0]= .009048;
	factors[1][1]= .008187;
	factors[1][2]= .007408;
	factors[1][3]= .006703;
	factors[1][4]= .006065;
	factors[1][5]= .005488;
	factors[1][6]= .004966;
	factors[1][7]= .004493;
	factors[1][8]= .004066;
	factors[1][9]= .003679;

	factors[2][0]= .001000;
	factors[2][1]= .000750;
	factors[2][2]= .000500;
	factors[2][3]= .000250;
	factors[2][4]= .000000;
	factors[2][5]= -.000250;
	factors[2][6]= -.000500;
	factors[2][7]= -.000750;
	factors[2][8]= -.001000;
	factors[2][9]= -.001250;

	// setting up multiple swaptions
	swaptions = 
#if defined(ENABLE_ARGO)
		argo::conew_array<parm>(nSwaptions);

		gflag = argo::conew_array<bool>(numtasks);
		gseed = argo::conew_<long>(seed);
#elif defined(ENABLE_OMPSS_2_CLUSTER)
		(parm *)nanos6_dmalloc(sizeof(parm)*nSwaptions, nanos6_equpart_distribution, 0, NULL);

		gseed = (long*)nanos6_dmalloc(sizeof(long), nanos6_equpart_distribution, 0, NULL);
		#pragma oss task out(*gseed) firstprivate(seed)
			*gseed = seed;
			
#else
		(parm *)malloc(sizeof(parm)*nSwaptions);
#endif

	int k;
#if defined(ENABLE_ARGO)
	distribute(beg, end, nSwaptions, 0, 0);

	// done to avoid remote accesses for dYears, dStrike (*X)
	if (workrank != 0) {
		while(!gflag[workrank-1]) {
			argo::backend::selective_acquire(&gflag[workrank-1], sizeof(bool));
		}
	}

	// (*X)
	for (i = beg; i < end; i++) {
		swaptions[i].dYears = 5.0 + ((int)(60*RanUnif(gseed)))*0.25; //5 to 20 years in 3 month intervals
		swaptions[i].dStrike =  0.1 + ((int)(49*RanUnif(gseed)))*0.1; //strikes ranging from 0.1 to 5.0 in steps of 0.1 
	}

	// (*X)
	gflag[workrank] = 1;
	argo::backend::selective_release(&gflag[workrank], sizeof(bool));

	// let master process do the initialization for dYears and dStrike till issue is investigated
	// if (workrank == 0) {
	// 	for (i = 0; i < nSwaptions; i++) {
	// 		swaptions[i].dYears = 5.0 + ((int)(60*RanUnif(&seed)))*0.25; //5 to 20 years in 3 month intervals
	// 		swaptions[i].dStrike =  0.1 + ((int)(49*RanUnif(&seed)))*0.1; //strikes ranging from 0.1 to 5.0 in steps of 0.1 
	// 	}
	// }
	// argo::barrier();

	//#pragma omp parallel for private(i, k, j) schedule(SCHED_POLICY)
	for (i = beg; i < end; i++) {
#elif defined(ENABLE_OMPSS_2_CLUSTER)
	int generic_chunk_nSwaptions = nSwaptions / nanos6_get_num_cluster_nodes();

	for(int z = 0; z < nSwaptions; z += generic_chunk_nSwaptions) {
		int node_id, chunk_per_node_nSwaptions;
		node_chunk(node_id, chunk_per_node_nSwaptions, nSwaptions, z, generic_chunk_nSwaptions);

		#pragma oss task out(swaptions[z;chunk_per_node_nSwaptions])	\
				 inout(*gseed)					\
				 private(i)					\
				 firstprivate(z, chunk_per_node_nSwaptions)	\
				 node(node_id)
		for (i = z; i < z+chunk_per_node_nSwaptions; i++) {
			swaptions[i].dYears = 5.0 + ((int)(60*RanUnif(gseed)))*0.25; //5 to 20 years in 3 month intervals
			swaptions[i].dStrike =  0.1 + ((int)(49*RanUnif(gseed)))*0.1; //strikes ranging from 0.1 to 5.0 in steps of 0.1 
		}
		#pragma oss taskwait
	}
#ifdef ENABLE_WEAK
	for(int z = 0; z < nSwaptions; z += generic_chunk_nSwaptions) {
		int node_id, chunk_per_node_nSwaptions;
		node_chunk(node_id, chunk_per_node_nSwaptions, nSwaptions, z, generic_chunk_nSwaptions);

		#pragma oss task weakinout(swaptions[z;chunk_per_node_nSwaptions])		\
				 private(i, j, k)						\
				 firstprivate(z, chunk_per_node_nSwaptions, iN, iFactors)	\
				 node(node_id)
		{
			#pragma oss task inout(swaptions[z;chunk_per_node_nSwaptions])	\
					 node(nanos6_cluster_no_offload)
			{
				// fetch all data in one go
			}

			for(i = z; i < z+chunk_per_node_nSwaptions; i += BSIZE) {
				int beg, end, chunk;
				task_chunk(beg, end, chunk, z+chunk_per_node_nSwaptions, i, BSIZE);

				#pragma oss task inout(swaptions[beg:end-1])		\
						 private(i, j, k)			\
						 firstprivate(beg, end, iN, iFactors)	\
						 node(nanos6_cluster_no_offload)
				for (i = beg; i < end; i++) {
#else
	for (i = 0; i < nSwaptions; i += BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, nSwaptions, i, BSIZE);
	
		#pragma oss task inout(swaptions[beg:end-1])	\
				 private(i, j, k)		\
				 firstprivate(beg, end, iN, iFactors)
		for (i = beg; i < end; i++) {
#endif
#else
	for (i = 0; i < nSwaptions; i++) {
#endif
		swaptions[i].Id = i;
		swaptions[i].iN = iN;
		swaptions[i].iFactors = iFactors;
#ifndef ENABLE_ARGO
#ifndef ENABLE_OMPSS_2_CLUSTER
		swaptions[i].dYears = 5.0 + ((int)(60*RanUnif(&seed)))*0.25; //5 to 20 years in 3 month intervals
		swaptions[i].dStrike =  0.1 + ((int)(49*RanUnif(&seed)))*0.1; //strikes ranging from 0.1 to 5.0 in steps of 0.1 
#endif
#endif
		swaptions[i].dCompounding =  0;
		swaptions[i].dMaturity =  1.0;
		swaptions[i].dTenor =  2.0;
		swaptions[i].dPaymentInterval =  1.0;

#ifndef ENABLE_OMPSS_2_CLUSTER
		swaptions[i].pdYield = dvector(0,iN-1);
		swaptions[i].pdYield[0] = .1;
		for(j=1;j<=swaptions[i].iN-1;++j)
			swaptions[i].pdYield[j] = swaptions[i].pdYield[j-1]+.005;
#endif

#ifndef ENABLE_OMPSS_2_CLUSTER
		swaptions[i].ppdFactors = dmatrix(0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
		for(k=0;k<=swaptions[i].iFactors-1;++k)
			for(j=0;j<=swaptions[i].iN-2;++j)
				swaptions[i].ppdFactors[k][j] = factors[k][j];
#endif
	}
#if defined(ENABLE_OMPSS_2_CLUSTER)
#ifdef ENABLE_WEAK
			}
		}
	}
#else
	}
#endif
	#pragma oss taskwait
#elif defined(ENABLE_ARGO)
	argo::barrier();
#endif

	// **********Calling the Swaption Pricing Routine*****************

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

	//****ROI begins****//
	int startt = time(NULL);
#ifdef ENABLE_THREADS

	int threadIDs[nThreads];
	for (i = 0; i < nThreads; i++) {
		threadIDs[i] = i;
		pthread_create(&threads[i], &pthread_custom_attr, worker, &threadIDs[i]);
	}
	for (i = 0; i < nThreads; i++) {
		pthread_join(threads[i], NULL);
	}

	free(threads);

#else
	int threadID=0;
	worker(&threadID);
#endif //ENABLE_THREADS
	MAIN_PROC(workrank, std::cout << "Critical code execution time: " << time(NULL) - startt << std::endl);
	//***ROI ends***//

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

#ifdef ENABLE_OUTPUT
#ifdef ENABLE_OMPSS_2_CLUSTER
	#pragma oss task in(swaptions[0;nSwaptions])	\
			 firstprivate(workrank)		\
			 node(nanos6_cluster_no_offload)
#endif
	MAIN_PROC(workrank, write_to_file());
#ifdef ENABLE_OMPSS_2_CLUSTER
	#pragma oss taskwait
#endif
#endif

#ifdef ENABLE_ARGO
	for (i = beg; i < end; i++) {
#else
	for (i = 0; i < nSwaptions; i++) {
#endif
#ifndef ENABLE_OMPSS_2_CLUSTER
		free_dvector(swaptions[i].pdYield, 0, swaptions[i].iN-1);
		free_dmatrix(swaptions[i].ppdFactors, 0, swaptions[i].iFactors-1, 0, swaptions[i].iN-2);
#endif
	}

#ifdef ENABLE_ARGO
	argo::codelete_array(swaptions);
#elif defined(ENABLE_OMPSS_2_CLUSTER)
	nanos6_dfree(swaptions, sizeof(parm)*nSwaptions);
#else
	free(swaptions);
#endif

	//***********************************************************


	//****bench ends***//
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

#ifdef ENABLE_ARGO
	argo::finalize();
#endif

	return iSuccess;
}

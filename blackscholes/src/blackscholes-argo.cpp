// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,
//
// ArgoDSM/OpenMP version written by Ioannis Anevlavis - Eta Scale AB

#include "argo.hpp"

#include <omp.h>
#include <cmath>
#include <mutex>
#include <vector>
#include <iostream>
#include <sys/time.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

// Macro for only node0 to do stuff
#define MAIN_PROC(rank, inst) \
do { \
	if ((rank) == 0) { inst; } \
} while (0)

typedef struct OptionData_ {
	fptype s;          // spot price
	fptype strike;     // strike price
	fptype r;          // risk-free interest rate
	fptype divq;       // dividend rate
	fptype v;          // volatility
	fptype t;          // time to maturity or option expiration in years 
			   // (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
	char OptionType;   // Option type.  "P"=PUT, "C"=CALL
	fptype divs;       // dividend vals (not used in this test)
	fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

OptionData *data;
fptype *prices;
size_t numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;

int workrank;
int numtasks;
int nthreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Initialization Progress
void init_progress()
{
	static const size_t step = 10;
	static const size_t node_chunk = numOptions/numtasks;
	static const size_t step_chunk = node_chunk/step;
	
	static std::vector<bool> elem;
	static std::mutex progr_mutex;
	static size_t stigma = step_chunk;
	
	std::unique_lock<std::mutex> lock(progr_mutex, std::defer_lock);
	lock.lock();
	elem.push_back(1);
	if (elem.size() >= stigma) {
		std::cout << "init progress -- "
		          << "node: "   << workrank
		          << ", done: " << ((double)stigma/node_chunk)*100 << "%"
		          << std::endl;
		stigma += step_chunk;
	}
	lock.unlock();
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

fptype CNDF ( fptype InputX ) 
{
	int sign;

	fptype OutputX;
	fptype xInput;
	fptype xNPrimeofX;
	fptype expValues;
	fptype xK2;
	fptype xK2_2, xK2_3;
	fptype xK2_4, xK2_5;
	fptype xLocal, xLocal_1;
	fptype xLocal_2, xLocal_3;

	// Check for negative value of InputX
	if (InputX < 0.0) {
		InputX = -InputX;
		sign = 1;
	} else 
		sign = 0;

	xInput = InputX;

	// Compute NPrimeX term common to both four & six decimal accuracy calcs
	expValues = exp(-0.5f * InputX * InputX);
	xNPrimeofX = expValues;
	xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

	xK2 = 0.2316419 * xInput;
	xK2 = 1.0 + xK2;
	xK2 = 1.0 / xK2;
	xK2_2 = xK2 * xK2;
	xK2_3 = xK2_2 * xK2;
	xK2_4 = xK2_3 * xK2;
	xK2_5 = xK2_4 * xK2;

	xLocal_1 = xK2 * 0.319381530;
	xLocal_2 = xK2_2 * (-0.356563782);
	xLocal_3 = xK2_3 * 1.781477937;
	xLocal_2 = xLocal_2 + xLocal_3;
	xLocal_3 = xK2_4 * (-1.821255978);
	xLocal_2 = xLocal_2 + xLocal_3;
	xLocal_3 = xK2_5 * 1.330274429;
	xLocal_2 = xLocal_2 + xLocal_3;

	xLocal_1 = xLocal_2 + xLocal_1;
	xLocal   = xLocal_1 * xNPrimeofX;
	xLocal   = 1.0 - xLocal;

	OutputX  = xLocal;

	if (sign) {
		OutputX = 1.0 - OutputX;
	}

	return OutputX;
} 

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
		fptype strike, fptype rate, fptype volatility,
		fptype time, int otype, float timet )
{
	fptype OptionPrice;

	// local private working variables for the calculation
	fptype xStockPrice;
	fptype xStrikePrice;
	fptype xRiskFreeRate;
	fptype xVolatility;
	fptype xTime;
	fptype xSqrtTime;

	fptype logValues;
	fptype xLogTerm;
	fptype xD1; 
	fptype xD2;
	fptype xPowerTerm;
	fptype xDen;
	fptype d1;
	fptype d2;
	fptype FutureValueX;
	fptype NofXd1;
	fptype NofXd2;
	fptype NegNofXd1;
	fptype NegNofXd2;    

	xStockPrice = sptprice;
	xStrikePrice = strike;
	xRiskFreeRate = rate;
	xVolatility = volatility;

	xTime = time;
	xSqrtTime = sqrt(xTime);

	logValues = log( sptprice / strike );

	xLogTerm = logValues;


	xPowerTerm = xVolatility * xVolatility;
	xPowerTerm = xPowerTerm * 0.5;

	xD1 = xRiskFreeRate + xPowerTerm;
	xD1 = xD1 * xTime;
	xD1 = xD1 + xLogTerm;

	xDen = xVolatility * xSqrtTime;
	xD1 = xD1 / xDen;
	xD2 = xD1 -  xDen;

	d1 = xD1;
	d2 = xD2;

	NofXd1 = CNDF( d1 );
	NofXd2 = CNDF( d2 );

	FutureValueX = strike * ( exp( -(rate)*(time) ) );        
	if (otype == 0) {            
		OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);
	} else { 
		NegNofXd1 = (1.0 - NofXd1);
		NegNofXd2 = (1.0 - NofXd2);
		OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
	}

	return OptionPrice;
}

void distribute( int& beg, int& end, const int& loop_size,
		const int& beg_offset, const int& less_equal )
{
	int chunk = loop_size / numtasks;
	beg = workrank * chunk + ((workrank == 0) ? beg_offset : less_equal);
	end = (workrank != numtasks - 1) ? workrank * chunk + chunk : loop_size;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

int bs_thread() {
	int i, j;
	int beg, end;
	fptype price;
	fptype priceDelta;

	distribute(beg, end, numOptions, 0, 0);

	for (j=0; j<NUM_RUNS; j++) {
		#pragma omp parallel for private(i, price, priceDelta) schedule(SCHED_POLICY)
		for (i=beg; i<end; i++) {
			/* Calling main function to calculate option value based on 
			 * Black & Scholes's equation.
			 */
			price = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
					rate[i], volatility[i], otime[i], 
					otype[i], 0);
			prices[i] = price;

		// We put a barrier here to avoid overlapping the execution of
		// tasks in different runs
		// argo::barrier();
#ifdef ERR_CHK
			priceDelta = data[i].DGrefval - price;
			if( fabs(priceDelta) >= 1e-4 ){
				printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
						i, price, data[i].DGrefval, priceDelta);
				numError ++;
			}
#endif
		}
	}
	argo::barrier();

	return 0;
}

int main (int argc, char **argv)
{
	argo::init(0.5*1024*1024*1024UL);

	workrank = argo::node_id();
	numtasks = argo::number_of_nodes();

	#pragma omp parallel
	{
		#if defined(_OPENMP)
			#pragma omp master
			nthreads = omp_get_num_threads();
		#endif /* _OPENMP */
	}

	FILE *file;
	int i;
	int loopnum;
	int beg, end;
	fptype * buffer;
	int * buffer2;
	int rv;
	struct timeval start;
	struct timeval stop;
	unsigned long elapsed;

	MAIN_PROC(workrank, printf("PARSEC Benchmark Suite\n"));
	MAIN_PROC(workrank, fflush(NULL));

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_blackscholes);
#endif

	if (argc < 3)
	{
		MAIN_PROC(workrank, printf("Usage:\n\t%s <inputFile> <outputFile>\n", argv[0]));
		exit(1);
	}
	char *inputFile = argv[1];
	char *outputFile = argv[2];

	// Read input data from file
	file = fopen(inputFile, "r");
	if(file == NULL) {
		MAIN_PROC(workrank, printf("ERROR: Unable to open file `%s'.\n", inputFile));
		exit(1);
	}
	rv = fscanf(file, "%lu", &numOptions);
	if(rv != 1) {
		MAIN_PROC(workrank, printf("ERROR: Unable to read from file `%s'.\n", inputFile));
		fclose(file);
		exit(1);
	}
	if(nthreads > numOptions) {
		MAIN_PROC(workrank, printf("WARNING: Not enough work, reducing number of threads to match number of options.\n"));
		nthreads = numOptions;
	}

	// Alloc spaces for the option data
	data = argo::conew_array<OptionData>(numOptions);
	prices = argo::conew_array<fptype>(numOptions);
	
	if (workrank == 0) {
		for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
		{
			rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
			if(rv != 9) {
				printf("ERROR: Unable to read from file `%s'.\n", inputFile);
				fclose(file);
				exit(1);
			}
		}
	}
	argo::barrier();

	rv = fclose(file);
	if(rv != 0) {
		MAIN_PROC(workrank, printf("ERROR: Unable to close file `%s'.\n", inputFile));
		exit(1);
	}

	MAIN_PROC(workrank, printf("Num of Options: %lu\n", numOptions));
	MAIN_PROC(workrank, printf("Num of Runs: %d\n", NUM_RUNS));

#define PAD 256
#define LINESIZE 64

	buffer = argo::conew_array<fptype>(5 * numOptions + PAD/sizeof(fptype));
	sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
	strike = sptprice + numOptions;
	rate = strike + numOptions;
	volatility = rate + numOptions;
	otime = volatility + numOptions;

	buffer2 = (int *) argo::conew_array<fptype>(numOptions + PAD/sizeof(fptype));
	otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

	distribute(beg, end, numOptions, 0, 0);

	#pragma omp parallel for private(i) schedule(SCHED_POLICY)
	for (i=beg; i<end; i++) {
		otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
		sptprice[i]   = data[i].s;
		strike[i]     = data[i].strike;
		rate[i]       = data[i].r;
		volatility[i] = data[i].v;
		otime[i]      = data[i].t;

		// init progress
		init_progress();
	}
	argo::barrier();

	MAIN_PROC(workrank, printf("Size of data: %lu\n", numOptions * (sizeof(OptionData) + sizeof(int))));

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

	gettimeofday(&start,NULL);
	bs_thread();
	gettimeofday(&stop,NULL);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

#ifdef ENABLE_OUTPUT
	// Write prices to output file
	if (workrank == 0) {
		file = fopen(outputFile, "w");
		if(file == NULL) {
			printf("ERROR: Unable to open file `%s'.\n", outputFile);
			exit(1);
		}
		rv = fprintf(file, "%lu\n", numOptions);
		if(rv < 0) {
			printf("ERROR: Unable to write to file `%s'.\n", outputFile);
			fclose(file);
			exit(1);
		}
		for(i=0; i<numOptions; i++) {
			rv = fprintf(file, "%.18f\n", prices[i]);
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
#endif

#ifdef ERR_CHK
	printf("Workrank: %d, Num Errors: %d\n", workrank, numError);
#endif

	argo::codelete_array(data);
	argo::codelete_array(prices);
	argo::codelete_array(buffer);
	argo::codelete_array(buffer2);

	elapsed = 1000000 * (stop.tv_sec - start.tv_sec);
	elapsed += stop.tv_usec - start.tv_usec;
	MAIN_PROC(workrank, printf("par_sec_time_us:%lu\n",elapsed));

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	argo::finalize();

	return 0;
}

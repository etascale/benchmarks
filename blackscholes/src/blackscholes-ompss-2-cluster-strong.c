// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,
//
// OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis and Iulian Brumar - Barcelona Supercomputing Center
//
// OmpSs-2@Cluster version written by Ioannis Anevlavis - Eta Scale AB

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Multi-threaded OpenMP header
// #include <omp.h>

#define BSIZE_UNIT 1024
int BSIZE;

// Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

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
size_t numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
static void task_chunk(int* beg,
		int* end,
		int* chunk,
		const int size,
		const int index,
		const int bsize)
{
	*chunk = (size - index > bsize) ? bsize : size - index;
	*beg = index;
	*end = index + *chunk;
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

// For debugging
void print_xmm(fptype in, char* s) {
	printf("%s: %f\n", s, in);
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
// #pragma oss task input([bsize]sptprice,[bsize]strike,[bsize]rate,[bsize]volatility,[bsize]time,[bsize]otype,timet) output([bsize]OptionPrice) label("BlkSchlsEqEuroNoDiv")
void BlkSchlsEqEuroNoDiv( fptype sptprice[BSIZE],
		fptype strike[BSIZE], fptype rate[BSIZE], fptype volatility[BSIZE],
		fptype time[BSIZE], int otype[BSIZE], float timet, fptype OptionPrice[BSIZE], int bsize)
{
	int i;

	for (i=0; i<bsize; i++) {
		// Local private working variables for the calculation
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

		xStockPrice = sptprice[i];
		xStrikePrice = strike[i];
		xRiskFreeRate = rate[i];
		xVolatility = volatility[i];

		xTime = time[i];
		xSqrtTime = sqrt(xTime);

		logValues = log( sptprice[i] / strike[i] );

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

		FutureValueX = strike[i] * ( exp( -(rate[i])*(time[i]) ) );        
		if (otype[i] == 0) {            
			OptionPrice[i] = (sptprice[i] * NofXd1) - (FutureValueX * NofXd2);
		} else { 
			NegNofXd1 = (1.0 - NofXd1);
			NegNofXd2 = (1.0 - NofXd2);
			OptionPrice[i] = (FutureValueX * NegNofXd2) - (sptprice[i] * NegNofXd1);
		}
	} 
}
void BlkSchlsEqEuroNoDiv_inline( fptype *sptprice,
		fptype *strike, fptype *rate, fptype *volatility,
		fptype *time, int *otype, float timet, fptype *OptionPrice, int size)
{
	int i;

	for (i=0; i<size; i++) {
		// Local private working variables for the calculation
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

		xStockPrice = sptprice[i];
		xStrikePrice = strike[i];
		xRiskFreeRate = rate[i];
		xVolatility = volatility[i];

		xTime = time[i];
		xSqrtTime = sqrt(xTime);

		logValues = log( sptprice[i] / strike[i] );

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

		FutureValueX = strike[i] * ( exp( -(rate[i])*(time[i]) ) );        
		if (otype[i] == 0) {            
			OptionPrice[i] = (sptprice[i] * NofXd1) - (FutureValueX * NofXd2);
		} else { 
			NegNofXd1 = (1.0 - NofXd1);
			NegNofXd2 = (1.0 - NofXd2);
			OptionPrice[i] = (FutureValueX * NegNofXd2) - (sptprice[i] * NegNofXd1);
		}
	} 
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
void bs_thread(fptype *prices) {
	for (int j = 0; j < NUM_RUNS; j++) {
		for (int i = 0; i < numOptions; i += BSIZE) {
			int beg, end, chunk;
			task_chunk(&beg, &end, &chunk, numOptions, i, BSIZE);

			/* Calling main function to calculate option value based on 
			 * Black & Sholes's equation.
			 */
			#pragma oss task in( sptprice  [beg:end-1],	\
					     strike    [beg:end-1],	\
					     rate      [beg:end-1],	\
					     volatility[beg:end-1],	\
					     otime     [beg:end-1],	\
					     otype     [beg:end-1])	\
					 out(prices    [beg:end-1])	\
					 firstprivate(beg, end)		\
					 label("local: BlkSchlsEqEuroNoDiv")
			BlkSchlsEqEuroNoDiv( &sptprice[beg], &strike[beg],
					&rate[beg], &volatility[beg], &otime[beg],
					&otype[beg], 0, &prices[beg], chunk);
		}

		// We put a barrier here to avoid overlapping the execution of
		// tasks in different runs
		// #pragma oss taskwait
#ifdef ERR_CHK
		for (int i=0; i<numOptions; i++) {
			fptype priceDelta = data[i].DGrefval - prices[i];
			if( fabs(priceDelta) >= 1e-4 ){
				printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
						i, prices[i], data[i].DGrefval, priceDelta);
				numError ++;
			}
		}
#endif
	}
	#pragma oss taskwait
}

int main (int argc, char **argv)
{
	int rv;
	FILE *file;
	int *buffer2;
	fptype *buffer;
	fptype *prices;
	struct timeval start;
	struct timeval stop;
	unsigned long elapsed;

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_blackscholes);
#endif

	if (argc < 3)
	{
		printf("Usage:\n\t%s <inputFile> <outputFile> [blocksize]\n", argv[0]);
		printf("Warning: nthreads is ignored! Use NX_ARGS=\"--threads=<nthreads>\" instead\n");
		exit(1);
	}

	char *inputFile = argv[1];
	char *outputFile = argv[2];

	if(argc > 3 ) {
		BSIZE = atoi(argv[3]);
	}
	else {
		BSIZE = BSIZE_UNIT;
	}

	// Read input data from file
	file = fopen(inputFile, "r");
	if(file == NULL) {
		printf("ERROR: Unable to open file `%s'.\n", inputFile);
		exit(1);
	}
	rv = fscanf(file, "%i", &numOptions);
	if(rv != 1) {
		printf("ERROR: Unable to read from file `%s'.\n", inputFile);
		fclose(file);
		exit(1);
	}
	if(BSIZE > numOptions) {
		printf("ERROR: Block size larger than number of options. Please reduce the block size, or use larger data size.\n");
		exit(1);
	}

	// Alloc spaces for the option data
	data = (OptionData*)nanos6_lmalloc(numOptions*sizeof(OptionData));
	prices = (fptype*)nanos6_dmalloc(numOptions*sizeof(fptype), nanos6_equpart_distribution, 0, NULL);
	
	for (int loopnum = 0; loopnum < numOptions; ++loopnum)
	{
		rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r, &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t, &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
		if(rv != 9) {
			printf("ERROR: Unable to read from file `%s'.\n", inputFile);
			fclose(file);
			exit(1);
		}
	}
	rv = fclose(file);
	if(rv != 0) {
		printf("ERROR: Unable to close file `%s'.\n", inputFile);
		exit(1);
	}

	printf("Num of Options: %d\n", numOptions);
	printf("Num of Runs: %d\n", NUM_RUNS);

#define PAD 256
#define LINESIZE 64

	buffer = (fptype *) nanos6_dmalloc(5 * numOptions * sizeof(fptype) + PAD, nanos6_equpart_distribution, 0, NULL);
	sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
	strike = sptprice + numOptions;
	rate = strike + numOptions;
	volatility = rate + numOptions;
	otime = volatility + numOptions;

	buffer2 = (int *) nanos6_dmalloc(numOptions * sizeof(fptype) + PAD, nanos6_equpart_distribution, 0, NULL);
	otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

	for (int i = 0; i < numOptions; i += BSIZE) {
		int beg, end, chunk;
		task_chunk(&beg, &end, &chunk, numOptions, i, BSIZE);
		
		#pragma oss task in( data      [beg:end-1])	\
				 out(otype     [beg:end-1],	\
				     sptprice  [beg:end-1],	\
				     strike    [beg:end-1],	\
				     rate      [beg:end-1],	\
				     volatility[beg:end-1], 	\
				     otime     [beg:end-1])	\
				 firstprivate(beg, end)		\
			 	 label("remote: initialize data")
		for (int k = beg; k < end; k++) {
			otype[k]      = (data[k].OptionType == 'P') ? 1 : 0;
			sptprice[k]   = data[k].s;
			strike[k]     = data[k].strike;
			rate[k]       = data[k].r;
			volatility[k] = data[k].v;
			otime[k]      = data[k].t;
		}
	}
	#pragma oss taskwait

	printf("Size of data: %lu\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

	gettimeofday(&start,NULL);
	bs_thread(prices);
	gettimeofday(&stop,NULL);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif

#ifdef ENABLE_OUTPUT
	// Bring price data locally
	fptype* lprices = (fptype*)nanos6_lmalloc(numOptions*sizeof(fptype));
	#pragma oss task in(prices[0;numOptions])	\
			 out(lprices[0;numOptions])	\
			 firstprivate(numOptions)	\
			 node(nanos6_cluster_no_offload)
	for (int i = 0; i < numOptions; ++i)
		lprices[i] = prices[i];
	#pragma oss taskwait

	// Write prices to output file
	file = fopen(outputFile, "w");
	if(file == NULL) {
		printf("ERROR: Unable to open file `%s'.\n", outputFile);
		exit(1);
	}
	rv = fprintf(file, "%i\n", numOptions);
	if(rv < 0) {
		printf("ERROR: Unable to write to file `%s'.\n", outputFile);
		fclose(file);
		exit(1);
	}
	for(int i=0; i<numOptions; i++) {
		rv = fprintf(file, "%.18f\n", lprices[i]);
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
#endif

#ifdef ERR_CHK
	printf("Num Errors: %d\n", numError);
#endif
	// Deallocate locally allocated memory
	nanos6_lfree(lprices, numOptions*sizeof(fptype));
	nanos6_lfree(data, numOptions*sizeof(OptionData));
	
	// Deallocate globally allocated memory
	nanos6_dfree(prices, numOptions*sizeof(fptype));
	nanos6_dfree(buffer, 5 * numOptions * sizeof(fptype) + PAD);
	nanos6_dfree(buffer2, numOptions * sizeof(fptype) + PAD);

	elapsed = 1000000 * (stop.tv_sec - start.tv_sec);
	elapsed += stop.tv_usec - start.tv_usec;
	printf("par_sec_time_us:%lu\n",elapsed);

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	return 0;
}

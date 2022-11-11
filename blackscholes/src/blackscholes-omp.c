// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,
//
// OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis and Iulian Brumar - Barcelona Supercomputing Center

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

// Multi-threaded OpenMP header
//#include <omp.h>

#define BSIZE_UNIT 1024;
int BSIZE;
//#define BSIZE 1024

//Precision to use for calculations
#define fptype float

#define NUM_RUNS 100

typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years 
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)  
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
//int nThreads;

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
//#pragma omp task input([bsize]sptprice,[bsize]strike,[bsize]rate,[bsize]volatility,[bsize]time,[bsize]otype,timet) output([bsize]OptionPrice) label(BlkSchlsEqEuroNoDiv)
void BlkSchlsEqEuroNoDiv( fptype sptprice[BSIZE],
                            fptype strike[BSIZE], fptype rate[BSIZE], fptype volatility[BSIZE],
                            fptype time[BSIZE], int otype[BSIZE], float timet, fptype OptionPrice[BSIZE], int bsize)
{
    int i;

    for (i=0; i<bsize; i++) {
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
void bs_thread(void *tid_ptr,fptype *prices) {

		#pragma omp parallel
		{
			#pragma omp single
			{
				int i, j;
				fptype priceDelta;
				int tid = *(int *)tid_ptr;

				for (j=0; j<NUM_RUNS; j++) {
				    for (i=0; i<=(numOptions-BSIZE); i+=BSIZE) {
				        /* Calling main function to calculate option value based on 
				         * Black & Sholes's equation.
				         */
				   			//#pragma omp task depend(in: sptprice[i],strike[i],rate[i],volatility[i],otime[i],otype[i]) depend(out: prices[i])
					 		#pragma omp task	
							BlkSchlsEqEuroNoDiv( &sptprice[i], &strike[i],
				                             &rate[i], &volatility[i], &otime[i],
				                             &otype[i], 0, &prices[i], BSIZE);
				    }
				    BlkSchlsEqEuroNoDiv( &sptprice[i], &strike[i],
				                                &rate[i], &volatility[i], &otime[i],
				                                &otype[i], 0, &prices[i], numOptions-i); //el thread creador tambe executa les iteracions sobrants d'anar de BS en BS

				    //We put a barrier here to avoid overlapping the execution of
				    // tasks in different runs
				    #pragma omp taskwait
		#ifdef ERR_CHK
				    for (i=0; i<numOptions; i++) {
				        priceDelta = data[i].DGrefval - prices[i];
				        if( fabs(priceDelta) >= 1e-4 ){
				            printf("Error on %d. Computed=%.5f, Ref=%.5f, Delta=%.5f\n",
				                   i, prices[i], data[i].DGrefval, priceDelta);
				            numError ++;
				        }
				    }
		#endif
				}
			} // end of single
		} //end of parallel region
}

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    fptype * buffer;
    fptype *prices;
    int * buffer2;
    int rv;
    struct timeval start;
    struct timeval stop;
    unsigned long elapsed;

#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_blackscholes);
#endif

   if (argc < 4)
        {
                printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile> [blocksize]\n", argv[0]);
                printf("Warning: nthreads is ignored! Use OMP_NUM_THREADS=<nthreads> instead\n");
                exit(1);
        }
        
    char *inputFile = argv[2];
    char *outputFile = argv[3];
	int nthreads = atoi(argv[1]);
	
	if(argc > 4 ) {
		BSIZE = atoi(argv[4]);
	}
	else {
		BSIZE = BSIZE_UNIT;
	}



    //Read input data from file
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
      //printf("WARNING: Not enough work, reducing number of threads to match number of options.\n");
      //nThreads = numOptions;
    }

    // alloc spaces for the option data
    data = (OptionData*)malloc(numOptions*sizeof(OptionData));
    prices = (fptype*)malloc(numOptions*sizeof(fptype));
    for ( loopnum = 0; loopnum < numOptions; ++ loopnum )
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

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i=0; i<numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s;
        strike[i]     = data[i].strike;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;
        otime[i]      = data[i].t;
    }

    printf("Size of data: %lu\n", numOptions * (sizeof(OptionData) + sizeof(int)));

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

    //do work
    int tid=0;
    //omp_set_num_threads(nThreads);
    gettimeofday(&start,NULL);
    bs_thread(&tid,prices);
    gettimeofday(&stop,NULL);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif

#ifdef ENABLE_OUTPUT
    //Write prices to output file
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
#endif

#ifdef ERR_CHK
    printf("Num Errors: %d\n", numError);
#endif
    free(data);
    free(prices);

    elapsed = 1000000 * (stop.tv_sec - start.tv_sec);
    elapsed += stop.tv_usec - start.tv_usec;

    printf("par_sec_time_us:%lu\n",elapsed);

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

    return 0;
}


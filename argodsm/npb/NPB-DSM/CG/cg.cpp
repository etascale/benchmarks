/**
 * NASA Advanced Supercomputing Parallel Benchmarks C++
 * 
 * based on NPB 3.3.1
 *
 * original version and technical report:
 * http://www.nas.nasa.gov/Software/NPB/
 *
 * Authors:
 *     M. Yarrow
 *     C. Kuszmaul
 *
 * C++ version:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Alves de Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 *
 * OpenMP version:
 *      Júnior Löff <loffjh@gmail.com>
 */

#include <vector>
#include <utility>

#include "argo.hpp"
#include "omp.h"
#include "../common/npb-CPP.hpp"
#include "npbparams.hpp"

/*
 * ---------------------------------------------------------------------
 * note: please observe that in the routine conj_grad three 
 * implementations of the sparse matrix-vector multiply have
 * been supplied. the default matrix-vector multiply is not
 * loop unrolled. the alternate implementations are unrolled
 * to a depth of 2 and unrolled to a depth of 8. please
 * experiment with these to find the fastest for your particular
 * architecture. if reporting timing results, any of these three may
 * be used without penalty.
 * ---------------------------------------------------------------------
 * class specific parameters: 
 * it appears here for reference only.
 * these are their values, however, this info is imported in the npbparams.h
 * include file, which is written by the sys/setparams.c program.
 * ---------------------------------------------------------------------
 */
#define NZ 		(NA*(NONZER+1)*(NONZER+1))
#define NAZ 		(NA*(NONZER+1))
#define T_INIT 		0
#define T_BENCH 	1
#define T_CONJ_GRAD 	2
#define T_LAST 		3

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static int colidx[NZ];
static int rowstr[NA+1];
static int iv[NA];
static int arow[NA];
static int acol[NAZ];
static double aelt[NAZ];
static double a[NZ];
#else
static int (*colidx)=(int*)malloc(sizeof(int)*(NZ));
static int (*rowstr)=(int*)malloc(sizeof(int)*(NA+1));
static int (*iv)=(int*)malloc(sizeof(int)*(NA));
static int (*arow)=(int*)malloc(sizeof(int)*(NA));
static int (*acol)=(int*)malloc(sizeof(int)*(NAZ));
static double (*aelt)=(double*)malloc(sizeof(double)*(NAZ));
static double (*a)=(double*)malloc(sizeof(double)*(NZ));
#endif
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
static double amult;
static double tran;
static boolean timeron;

static double (*x);
static double (*p);
static double (*q);
static double (*r);
static double (*z);
static double (*gnorms);

static int workrank;
static int numtasks;
static int nthreads;

inline uint32_t flp2(uint32_t x)
{
	   x = x | (x >>  1);
	   x = x | (x >>  2);
	   x = x | (x >>  4);
	   x = x | (x >>  8);
	   x = x | (x >> 16);
	return x - (x >>  1);
}

#define ALIGN_UP(size, align) (((size) + (align) - 1) & ~((align) - 1))
#define ALIGN_DOWN(size, align) ((size) & ~((align) - 1))

#define unaligned_chunk ((NA+2) / (numtasks))
#define aligned_down_chunk ALIGN_DOWN((unaligned_chunk), 512)
#define power_of_two_chunk flp2(unaligned_chunk)

#define aligned_chunk ((numtasks*512 <= NA+1) ? (aligned_down_chunk) : (power_of_two_chunk))
#define remainder_chunk ((NA+1) - (numtasks)*(aligned_chunk))

/* function prototypes */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm,
		int dist_table[]);
static int icnvrt(double x,
		int ipwr2);
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]);
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift);
static void sprnvc(int n,
		int nz,
		int nn1,
		double v[],
		int iv[]);
static void vecset(int n,
		double v[],
		int iv[],
		int* nzv,
		int i,
		double val);
static void distribute(int& beg,
		int& end,
		const int& loop_size,
		const int& beg_offset,
		const int& less_equal,
		int dist_table[]);
static void gnorms_acc(std::vector<std::pair<double*, double&>> pv);

/* cg */
int main(int argc, char **argv){
	/*
	 * -------------------------------------------------------------------------
	 * initialize argodsm
	 * -------------------------------------------------------------------------
	 */
	argo::init(256*1024*1024UL,
	           256*1024*1024UL);
	/*
	 * -------------------------------------------------------------------------
	 * fetch workrank, number of nodes, and number of threads
	 * -------------------------------------------------------------------------
	 */ 
	workrank = argo::node_id();
	numtasks = argo::number_of_nodes();

	#pragma omp parallel
	{
		#if defined(_OPENMP)
			#pragma omp master
			nthreads = omp_get_num_threads();
		#endif /* _OPENMP */
	}
	/*
	 * -------------------------------------------------------------------------
	 * move global arrays allocation here, since this is a collective operation
	 * -------------------------------------------------------------------------
	 */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#endif

	p = argo::conew_array<double>(NA+2);
	q = argo::conew_array<double>(NA+2);
	r = argo::conew_array<double>(NA+2);
	x = argo::conew_array<double>(NA+2);
	z = argo::conew_array<double>(NA+2);
	gnorms = argo::conew_array<double>(numtasks*512);

	/*
	 * ---------------------------------------------------------------------
	 * create distribution table
	 * ---------------------------------------------------------------------
	 */
	int dist_table[numtasks];
	int min_chunk_size = 512;
	int remainder = remainder_chunk;
	
	dist_table[0] = 0;
	for (int i = 1; i < numtasks; ++i) {
		if (remainder >= min_chunk_size) {
			dist_table[i] = dist_table[i-1] + aligned_chunk + min_chunk_size;
			remainder -= min_chunk_size;
		} else
			dist_table[i] = dist_table[i-1] + aligned_chunk;
	}

	/*
	 * -------------------------------------------------------------------------
	 * continue with the local allocations
	 * -------------------------------------------------------------------------
	 */
	int    i, j, k, it;
	double zeta;
	double rnorm;
	double norm_temp1, norm_temp2;
	double t, mflops, tmax;
	char class_npb;
	boolean verified;
	double zeta_verify_value, epsilon, err;

	char *t_names[T_LAST];

	for(i=0; i<T_LAST; i++){
		timer_clear(i);
	}

	FILE* fp;
	if((fp = fopen("timer.flag", "r")) != NULL){
		timeron = TRUE;
		t_names[T_INIT] = (char*)"init";
		t_names[T_BENCH] = (char*)"benchmk";
		t_names[T_CONJ_GRAD] = (char*)"conjgd";
		fclose(fp);
	}else{
		timeron = FALSE;
	}

	timer_start(T_INIT);

	firstrow = 0;
	lastrow  = NA-1;
	firstcol = 0;
	lastcol  = NA-1;

	if(NA == 1400 && NONZER == 7 && NITER == 15 && SHIFT == 10.0){
		class_npb = 'S';
		zeta_verify_value = 8.5971775078648;
	}else if(NA == 7000 && NONZER == 8 && NITER == 15 && SHIFT == 12.0){
		class_npb = 'W';
		zeta_verify_value = 10.362595087124;
	}else if(NA == 14000 && NONZER == 11 && NITER == 15 && SHIFT == 20.0){
		class_npb = 'A';
		zeta_verify_value = 17.130235054029;
	}else if(NA == 75000 && NONZER == 13 && NITER == 75 && SHIFT == 60.0){
		class_npb = 'B';
		zeta_verify_value = 22.712745482631;
	}else if(NA == 150000 && NONZER == 15 && NITER == 75 && SHIFT == 110.0){
		class_npb = 'C';
		zeta_verify_value = 28.973605592845;
	}else if(NA == 1500000 && NONZER == 21 && NITER == 100 && SHIFT == 500.0){
		class_npb = 'D';
		zeta_verify_value = 52.514532105794;
	}else if(NA == 9000000 && NONZER == 26 && NITER == 100 && SHIFT == 1500.0){
		class_npb = 'E';
		zeta_verify_value = 77.522164599383;
	}else{
		class_npb = 'U';
	}

	if (workrank == 0) {
		printf("\n\n NAS Parallel Benchmarks 4.1 Parallel C++ version with OpenMP - CG Benchmark\n\n");
		printf(" Size: %11d\n", NA);
		printf(" Iterations: %5d\n", NITER);
	}

	naa = NA;
	nzz = NZ;

	/* initialize random number generator */
	tran    = 314159265.0;
	amult   = 1220703125.0;
	zeta    = randlc( &tran, amult );

	makea(naa, 
			nzz, 
			a, 
			colidx, 
			rowstr, 
			firstrow, 
			lastrow, 
			firstcol, 
			lastcol, 
			arow, 
			(int(*)[NONZER+1])(void*)acol, 
			(double(*)[NONZER+1])(void*)aelt,
			iv);

	/*
	 * ---------------------------------------------------------------------
	 * note: as a result of the above call to makea:
	 * values of j used in indexing rowstr go from 0 --> lastrow-firstrow
	 * values of colidx which are col indexes go from firstcol --> lastcol
	 * so:
	 * shift the col index vals from actual (firstcol --> lastcol) 
	 * to local, i.e., (0 --> lastcol-firstcol)
	 * ---------------------------------------------------------------------
	 */
	#pragma omp parallel private(it,i,j,k)	
	{
		#pragma omp for nowait
		for(j = 0; j < lastrow-firstrow+1; j++){
			for(k = rowstr[j]; k < rowstr[j+1]; k++){
				colidx[k] = colidx[k]-firstcol;
			}
		}

		int beg_col,  end_col;
		int beg_naa1, end_naa1;
		
		distribute(beg_naa1, end_naa1, naa+1             , 0, 0, dist_table);
		distribute(beg_col , end_col , lastcol-firstcol+1, 0, 0, dist_table);

		/*
		 * -------------------------------------------------------------------
		 * ---->
		 * do one iteration untimed to init all code and data page tables
		 * ----> (then reinit, start timing, to niter its)
		 * -------------------------------------------------------------------*/
		for(it = 1; it <= 1; it++){
			/* the call to the conjugate gradient routine */
			conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm, dist_table);

			#pragma omp single
			{
				norm_temp1 = 0.0;
				norm_temp2 = 0.0;
			}

			/*
			 * --------------------------------------------------------------------
			 * zeta = shift + 1/(x.z)
			 * so, first: (x.z)
			 * also, find norm of z
			 * so, first: (z.z)
			 * --------------------------------------------------------------------
			 */
			#pragma omp for reduction(+:norm_temp1,norm_temp2)
			for(j = beg_col; j < end_col; j++){
				norm_temp1 += x[j]*z[j];
				norm_temp2 += z[j]*z[j];
			}
			gnorms_acc({{&gnorms[0], norm_temp1},
			            {&gnorms[1], norm_temp2}});

			#pragma omp single
			{
				norm_temp2 = 1.0 / sqrt(norm_temp2);
				zeta = SHIFT + 1.0 / norm_temp1;
			}
			
			/* normalize z to obtain x */
			#pragma omp for 
			for(j = beg_col; j < end_col; j++){
				x[j] = norm_temp2*z[j];
			}
		} /* end of do one iteration untimed */

		/* set starting vector to (1, 1, .... 1) */	
		#pragma omp for
		for(i = beg_naa1; i < end_naa1; i++){
			x[i] = 1.0;
		}

		#pragma omp single
			zeta = 0.0;
		argo::barrier(nthreads);

		#pragma omp master
		{
			timer_stop(T_INIT);
			if (workrank == 0)
				printf(" Initialization time = %15.3f seconds\n", timer_read(T_INIT));
			timer_start(T_BENCH);
		}

		/*
		 * --------------------------------------------------------------------
		 * ---->
		 * main iteration for inverse power method
		 * ---->
		 * --------------------------------------------------------------------
		 */
		for(it = 1; it <= NITER; it++){
			/* the call to the conjugate gradient routine */
			#pragma omp master
			if(timeron){timer_start(T_CONJ_GRAD);}
			conj_grad(colidx, rowstr, x, z, a, p, q, r, &rnorm, dist_table);
			#pragma omp master
			if(timeron){timer_stop(T_CONJ_GRAD);}

			#pragma omp single
			{
				norm_temp1 = 0.0;
				norm_temp2 = 0.0;
			}

			/*
			 * --------------------------------------------------------------------
			 * zeta = shift + 1/(x.z)
			 * so, first: (x.z)
			 * also, find norm of z
			 * so, first: (z.z)
			 * --------------------------------------------------------------------
			 */
			#pragma omp for reduction(+:norm_temp1,norm_temp2)
			for(j = beg_col; j < end_col; j++){
				norm_temp1 += x[j]*z[j];
				norm_temp2 += z[j]*z[j];
			}
			gnorms_acc({{&gnorms[0], norm_temp1},
			            {&gnorms[1], norm_temp2}});

			#pragma omp single
			{
				norm_temp2 = 1.0 / sqrt(norm_temp2);
				zeta = SHIFT + 1.0 / norm_temp1;
			}

			#pragma omp master
			{
				if (workrank == 0) {
					if(it==1){printf("\n   iteration           ||r||                 zeta\n");}
					printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);
				}
			}
			
			/* normalize z to obtain x */
			#pragma omp for 
			for(j = beg_col; j < end_col; j++){
				x[j] = norm_temp2*z[j];
			}
		} /* end of main iter inv pow meth */
	} /* end parallel */
	timer_stop(T_BENCH);

	/*
	 * --------------------------------------------------------------------
	 * end of timed section
	 * --------------------------------------------------------------------
	 */

	t = timer_read(T_BENCH);

if (workrank == 0) {
	printf(" Benchmark completed\n");

	epsilon = 1.0e-10;
	if(class_npb != 'U'){
		err = fabs(zeta - zeta_verify_value) / zeta_verify_value;
		if(err <= epsilon){
			verified = TRUE;
			printf(" VERIFICATION SUCCESSFUL\n");
			printf(" Zeta is    %20.13e\n", zeta);
			printf(" Error is   %20.13e\n", err);
		}else{
			verified = FALSE;
			printf(" VERIFICATION FAILED\n");
			printf(" Zeta                %20.13e\n", zeta);
			printf(" The correct zeta is %20.13e\n", zeta_verify_value);
		}
	}else{
		verified = FALSE;
		printf(" Problem size unknown\n");
		printf(" NO VERIFICATION PERFORMED\n");
	}
	if(t != 0.0){
		mflops = (double)(2.0*NITER*NA)
			* (3.0+(double)(NONZER*(NONZER+1))
					+ 25.0
					* (5.0+(double)(NONZER*(NONZER+1)))+3.0)
			/ t / 1000000.0;
	}else{
		mflops = 0.0;
	}
	c_print_results((char*)"CG",
			class_npb,
			NA,
			0,
			0,
			NITER,
			t,
			mflops,
			(char*)"          floating point",
			verified,
			(char*)NPBVERSION,
			(char*)COMPILETIME,
			(char*)COMPILERVERSION,
			(char*)LIBVERSION,
			std::getenv("OMP_NUM_THREADS"),
			(char*)CS1,
			(char*)CS2,
			(char*)CS3,
			(char*)CS4,
			(char*)CS5,
			(char*)CS6,
			(char*)CS7);

	/*
	 * ---------------------------------------------------------------------
	 * more timers
	 * ---------------------------------------------------------------------
	 */
	if(timeron){
		tmax = timer_read(T_BENCH);
		if(tmax == 0.0){tmax = 1.0;}
		printf("  SECTION   Time (secs)\n");
		for(i = 0; i < T_LAST; i++){
			t = timer_read(i);
			if(i == T_INIT){
				printf("  %8s:%9.3f\n", t_names[i], t);
			}else{
				printf("  %8s:%9.3f  (%6.2f%%)\n", t_names[i], t, t*100.0/tmax);
				if(i == T_CONJ_GRAD){
					t = tmax - t;
					printf("    --> %8s:%9.3f  (%6.2f%%)\n", "rest", t, t*100.0/tmax);
				}
			}
		}
	}
}
	/*
	 * -------------------------------------------------------------------------
	 * deallocate data structures
	 * -------------------------------------------------------------------------
	 */
	argo::codelete_array(p);
	argo::codelete_array(q);
	argo::codelete_array(r);
	argo::codelete_array(x);
	argo::codelete_array(z);

	argo::codelete_array(gnorms);
	/*
	 * -------------------------------------------------------------------------
	 * finalize argodsm
	 * -------------------------------------------------------------------------
	 */
	argo::finalize();

	return 0;
}

/*
 * ---------------------------------------------------------------------
 * floating point arrays here are named as in NPB1 spec discussion of 
 * CG algorithm
 * ---------------------------------------------------------------------
 */
static void conj_grad(int colidx[],
		int rowstr[],
		double x[],
		double z[],
		double a[],
		double p[],
		double q[],
		double r[],
		double* rnorm,
		int dist_table[]){
	int j, k;
	int cgit, cgitmax;
	double alpha, beta, suml;
	static double d, sum, rho, rho0;

	int beg_row,  end_row;
	int beg_col,  end_col;
	int beg_naa,  end_naa;
	int beg_naa1, end_naa1;

	distribute(beg_naa , end_naa , naa  , 0, 0, dist_table);
	distribute(beg_naa1, end_naa1, naa+1, 0, 0, dist_table);

	distribute(beg_row, end_row, lastrow-firstrow+1, 0, 0, dist_table);
	distribute(beg_col, end_col, lastcol-firstcol+1, 0, 0, dist_table);

	cgitmax = 25;
	#pragma omp single nowait
	{
		rho = 0.0;
		sum = 0.0;
	}
	
	/* initialize the CG algorithm */
	#pragma omp for
	for(j = beg_naa1; j < end_naa1; j++){
		q[j] = 0.0;
		z[j] = 0.0;
		r[j] = x[j];
		p[j] = r[j];
	}

	/*
	 * --------------------------------------------------------------------
	 * rho = r.r
	 * now, obtain the norm of r: First, sum squares of r elements locally...
	 * --------------------------------------------------------------------
	 */
	#pragma omp for reduction(+:rho)
	for(j = beg_col; j < end_col; j++){
		rho += r[j]*r[j];
	}
	gnorms_acc({{&gnorms[0], rho}});

	/* the conj grad iteration loop */
	for(cgit = 1; cgit <= cgitmax; cgit++){
		/*
		 * ---------------------------------------------------------------------
		 * q = A.p
		 * the partition submatrix-vector multiply: use workspace w
		 * ---------------------------------------------------------------------
		 * 
		 * note: this version of the multiply is actually (slightly: maybe %5) 
		 * faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		 * below. on the Cray t3d, the reverse is TRUE, i.e., the 
		 * unrolled-by-two version is some 10% faster.  
		 * the unrolled-by-8 version below is significantly faster
		 * on the Cray t3d - overall speed of code is 1.5 times faster.
		 */

		#pragma omp single nowait
		{
			d = 0.0;
			/*
			 * --------------------------------------------------------------------
			 * save a temporary of rho
			 * --------------------------------------------------------------------
			 */
			rho0 = rho;
			rho = 0.0;
		}

		#pragma omp for nowait
		for(j = beg_row; j < end_row; j++){
			suml = 0.0;
			for(k = rowstr[j]; k < rowstr[j+1]; k++){
				suml += a[k]*p[colidx[k]];
			}
			q[j] = suml;
		}

		/*
		 * --------------------------------------------------------------------
		 * obtain p.q
		 * --------------------------------------------------------------------
		 */

		#pragma omp for reduction(+:d)
		for (j = beg_col; j < end_col; j++) {
			d += p[j]*q[j];
		}
		gnorms_acc({{&gnorms[1], d}});

		/*
		 * --------------------------------------------------------------------
		 * obtain alpha = rho / (p.q)
		 * -------------------------------------------------------------------
		 */
		alpha = rho0 / d;
			
		/*
		 * ---------------------------------------------------------------------
		 * obtain z = z + alpha*p
		 * and    r = r - alpha*q
		 * ---------------------------------------------------------------------
		 */
		#pragma omp for reduction(+:rho)
		for(j = beg_col; j < end_col; j++){
			z[j] += alpha*p[j];
			r[j] -= alpha*q[j];
			
			/*
			 * ---------------------------------------------------------------------
			 * rho = r.r
			 * now, obtain the norm of r: first, sum squares of r elements locally...
			 * ---------------------------------------------------------------------
			 */
			rho += r[j]*r[j];
		}
		gnorms_acc({{&gnorms[0], rho}});

		/*
		 * ---------------------------------------------------------------------
		 * obtain beta
		 * ---------------------------------------------------------------------
		 */	
		beta = rho / rho0;

		/*
		 * ---------------------------------------------------------------------
		 * p = r + beta*p
		 * ---------------------------------------------------------------------
		 */
		#pragma omp for
		for(j = beg_col; j < end_col; j++){
			p[j] = r[j] + beta*p[j];
		}
		argo::barrier(nthreads);
	} /* end of do cgit=1, cgitmax */

	/*
	 * ---------------------------------------------------------------------
	 * compute residual norm explicitly: ||r|| = ||x - A.z||
	 * first, form A.z
	 * the partition submatrix-vector multiply
	 * ---------------------------------------------------------------------
	 */
	#pragma omp for nowait
	for(j = beg_row; j < end_row; j++){
		suml = 0.0;
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			suml += a[k]*z[colidx[k]];
		}
		r[j] = suml;
	}

	/*
	 * ---------------------------------------------------------------------
	 * at this point, r contains A.z
	 * ---------------------------------------------------------------------
	 */
	#pragma omp for reduction(+:sum)
	for(j = beg_col; j < end_col; j++){
		suml = x[j]-r[j];
		sum += suml*suml;
	}
	gnorms_acc({{&gnorms[1], sum}});

	#pragma omp single
	{
		*rnorm = sqrt(sum);
	}
}

/*
 * ---------------------------------------------------------------------
 * scale a double precision number x in (0,1) by a power of 2 and chop it
 * ---------------------------------------------------------------------
 */
static int icnvrt(double x, int ipwr2){
	return (int)(ipwr2 * x);
}

/*
 * ---------------------------------------------------------------------
 * generate the test problem for benchmark 6
 * makea generates a sparse matrix with a
 * prescribed sparsity distribution
 *
 * parameter    type        usage
 *
 * input
 *
 * n            i           number of cols/rows of matrix
 * nz           i           nonzeros as declared array size
 * rcond        r*8         condition number
 * shift        r*8         main diagonal shift
 *
 * output
 *
 * a            r*8         array for nonzeros
 * colidx       i           col indices
 * rowstr       i           row pointers
 *
 * workspace
 *
 * iv, arow, acol i
 * aelt           r*8
 * ---------------------------------------------------------------------
 */
static void makea(int n,
		int nz,
		double a[],
		int colidx[],
		int rowstr[],
		int firstrow,
		int lastrow,
		int firstcol,
		int lastcol,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int iv[]){
	int iouter, ivelt, nzv, nn1;
	int ivc[NONZER+1];
	double vc[NONZER+1];

	/*
	 * --------------------------------------------------------------------
	 * nonzer is approximately  (int(sqrt(nnza /n)));
	 * --------------------------------------------------------------------
	 * nn1 is the smallest power of two not less than n
	 * --------------------------------------------------------------------
	 */
	nn1 = 1;
	do{
		nn1 = 2 * nn1;
	}while(nn1 < n);

	/*
	 * -------------------------------------------------------------------
	 * generate nonzero positions and save for the use in sparse
	 * -------------------------------------------------------------------
	 */
	for(iouter = 0; iouter < n; iouter++){
		nzv = NONZER;
		sprnvc(n, nzv, nn1, vc, ivc);
		vecset(n, vc, ivc, &nzv, iouter+1, 0.5);
		arow[iouter] = nzv;
		for(ivelt = 0; ivelt < nzv; ivelt++){
			acol[iouter][ivelt] = ivc[ivelt] - 1;
			aelt[iouter][ivelt] = vc[ivelt];
		}
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... make the sparse matrix from list of elements with duplicates
	 * (iv is used as  workspace)
	 * ---------------------------------------------------------------------
	 */
	sparse(a,
			colidx,
			rowstr,
			n,
			nz,
			NONZER,
			arow,
			acol,
			aelt,
			firstrow,
			lastrow,
			iv,
			RCOND,
			SHIFT);
}

/*
 * ---------------------------------------------------------------------
 * rows range from firstrow to lastrow
 * the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
 * ---------------------------------------------------------------------
 */
static void sparse(double a[],
		int colidx[],
		int rowstr[],
		int n,
		int nz,
		int nozer,
		int arow[],
		int acol[][NONZER+1],
		double aelt[][NONZER+1],
		int firstrow,
		int lastrow,
		int nzloc[],
		double rcond,
		double shift){	
	int nrows;

	/*
	 * ---------------------------------------------------
	 * generate a sparse matrix from a list of
	 * [col, row, element] tri
	 * ---------------------------------------------------
	 */
	int i, j, j1, j2, nza, k, kk, nzrow, jcol;
	double size, scale, ratio, va;
	boolean goto_40;

	/*
	 * --------------------------------------------------------------------
	 * how many rows of result
	 * --------------------------------------------------------------------
	 */
	nrows = lastrow - firstrow + 1;

	/*
	 * --------------------------------------------------------------------
	 * ...count the number of triples in each row
	 * --------------------------------------------------------------------
	 */
	for(j = 0; j < nrows+1; j++){
		rowstr[j] = 0;
	}
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza] + 1;
			rowstr[j] = rowstr[j] + arow[i];
		}
	}
	rowstr[0] = 0;
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] + rowstr[j-1];
	}
	nza = rowstr[nrows] - 1;

	/*
	 * ---------------------------------------------------------------------
	 * ... rowstr(j) now is the location of the first nonzero
	 * of row j of a
	 * ---------------------------------------------------------------------
	 */
	if(nza > nz){
		printf("Space for matrix elements exceeded in sparse\n");
		printf("nza, nzmax = %d, %d\n", nza, nz);
		exit(EXIT_FAILURE);
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... preload data pages
	 * ---------------------------------------------------------------------
	 */
	for(j = 0; j < nrows; j++){
		for(k = rowstr[j]; k < rowstr[j+1]; k++){
			a[k] = 0.0;
			colidx[k] = -1;
		}
		nzloc[j] = 0;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... generate actual values by summing duplicates
	 * ---------------------------------------------------------------------
	 */
	size = 1.0;
	ratio = pow(rcond, (1.0 / (double)(n)));
	for(i = 0; i < n; i++){
		for(nza = 0; nza < arow[i]; nza++){
			j = acol[i][nza];

			scale = size * aelt[i][nza];
			for(nzrow = 0; nzrow < arow[i]; nzrow++){
				jcol = acol[i][nzrow];
				va = aelt[i][nzrow] * scale;

				/*
				 * --------------------------------------------------------------------
				 * ... add the identity * rcond to the generated matrix to bound
				 * the smallest eigenvalue from below by rcond
				 * --------------------------------------------------------------------
				 */
				if(jcol == j && j == i){
					va = va + rcond - shift;
				}

				goto_40 = FALSE;
				for(k = rowstr[j]; k < rowstr[j+1]; k++){
					if(colidx[k] > jcol){
						/*
						 * ----------------------------------------------------------------
						 * ... insert colidx here orderly
						 * ----------------------------------------------------------------
						 */
						for(kk = rowstr[j+1]-2; kk >= k; kk--){
							if(colidx[kk] > -1){
								a[kk+1] = a[kk];
								colidx[kk+1] = colidx[kk];
							}
						}
						colidx[k] = jcol;
						a[k]  = 0.0;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == -1){
						colidx[k] = jcol;
						goto_40 = TRUE;
						break;
					}else if(colidx[k] == jcol){
						/*
						 * --------------------------------------------------------------
						 * ... mark the duplicated entry
						 * -------------------------------------------------------------
						 */
						nzloc[j] = nzloc[j] + 1;
						goto_40 = TRUE;
						break;
					}
				}
				if(goto_40 == FALSE){
					printf("internal error in sparse: i=%d\n", i);
					exit(EXIT_FAILURE);
				}
				a[k] = a[k] + va;
			}
		}
		size = size * ratio;
	}

	/*
	 * ---------------------------------------------------------------------
	 * ... remove empty entries and generate final results
	 * ---------------------------------------------------------------------
	 */
	for(j = 1; j < nrows; j++){
		nzloc[j] = nzloc[j] + nzloc[j-1];
	}

	for(j = 0; j < nrows; j++){
		if(j > 0){
			j1 = rowstr[j] - nzloc[j-1];
		}else{
			j1 = 0;
		}
		j2 = rowstr[j+1] - nzloc[j];
		nza = rowstr[j];
		for(k = j1; k < j2; k++){
			a[k] = a[nza];
			colidx[k] = colidx[nza];
			nza = nza + 1;
		}
	}
	for(j = 1; j < nrows+1; j++){
		rowstr[j] = rowstr[j] - nzloc[j-1];
	}
	nza = rowstr[nrows] - 1;
}

/*
 * ---------------------------------------------------------------------
 * generate a sparse n-vector (v, iv)
 * having nzv nonzeros
 *
 * mark(i) is set to 1 if position i is nonzero.
 * mark is all zero on entry and is reset to all zero before exit
 * this corrects a performance bug found by John G. Lewis, caused by
 * reinitialization of mark on every one of the n calls to sprnvc
 * ---------------------------------------------------------------------
 */
static void sprnvc(int n, int nz, int nn1, double v[], int iv[]){
	int nzv, ii, i;
	double vecelt, vecloc;

	nzv = 0;

	while(nzv < nz){
		vecelt = randlc(&tran, amult);

		/*
		 * --------------------------------------------------------------------
		 * generate an integer between 1 and n in a portable manner
		 * --------------------------------------------------------------------
		 */
		vecloc = randlc(&tran, amult);
		i = icnvrt(vecloc, nn1) + 1;
		if(i>n){continue;}

		/*
		 * --------------------------------------------------------------------
		 * was this integer generated already?
		 * --------------------------------------------------------------------
		 */
		boolean was_gen = FALSE;
		for(ii = 0; ii < nzv; ii++){
			if(iv[ii] == i){
				was_gen = TRUE;
				break;
			}
		}
		if(was_gen){continue;}
		v[nzv] = vecelt;
		iv[nzv] = i;
		nzv = nzv + 1;
	}
}

/*
 * --------------------------------------------------------------------
 * set ith element of sparse vector (v, iv) with
 * nzv nonzeros to val
 * --------------------------------------------------------------------
 */
static void vecset(int n, double v[], int iv[], int* nzv, int i, double val){
	int k;
	boolean set;

	set = FALSE;
	for(k = 0; k < *nzv; k++){
		if(iv[k] == i){
			v[k] = val;
			set  = TRUE;
		}
	}
	if(set == FALSE){
		v[*nzv]  = val;
		iv[*nzv] = i;
		*nzv     = *nzv + 1;
	}
}

static void distribute(int& beg,
		int& end,
		const int& loop_size,
		const int& beg_offset,
		const int& less_equal,
		int dist_table[]){
	int chunk = aligned_chunk;
	beg = dist_table[workrank] + ((workrank == 0) ? beg_offset : less_equal);
	end = (workrank != numtasks - 1) ? dist_table[workrank+1]  : loop_size;
}

static void gnorms_acc(std::vector<std::pair<double*, double&>> pv)
{
	#pragma omp single
	{
		for (const auto iv : pv)
			*(iv.first + workrank*512) = iv.second;
		argo::barrier();

		for (const auto iv : pv)
			for (int i = 0; i < numtasks; ++i)
				if (i != workrank)
					iv.second += *(iv.first + i*512);
	}
}


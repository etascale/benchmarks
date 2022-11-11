/********************************************************************
  An OmpSs-2@Cluster implementation of Himeno.
  NOTE: This implementation uses strong dependencies.

  This benchmark test program is measuring a cpu performance
  of floating point operation by a Poisson equation solver.

  If you have any question, please ask me via email.
  written by Ryutaro HIMENO, November 26, 2001.
  Version 3.0
  ----------------------------------------------
  Ryutaro Himeno, Dr. of Eng.
  Head of Computer Information Division,
  RIKEN (The Institute of Pysical and Chemical Research)
  Email : himeno@postman.riken.go.jp
  ---------------------------------------------------------------
  You can adjust the size of this benchmark code to fit your target
  computer. In that case, please chose following sets of
  [mimax][mjmax][mkmax]:
  small : 33,33,65
  small : 65,65,129
  midium: 129,129,257
  large : 257,257,513
  ext.large: 513,513,1025
  This program is to measure a computer performance in MFLOPS
  by using a kernel which appears in a linear solver of pressure
  Poisson eq. which appears in an incompressible Navier-Stokes solver.
  A point-Jacobi method is employed in this solver as this method can 
  be easily vectorized and be parallelized.
  ------------------
  Finite-difference method, curvilinear coodinate system
  Vectorizable and parallelizable on each grid point
  No. of grid points : imax x jmax x kmax including boundaries
  ------------------
  A,B,C:coefficient matrix, wrk1: source term of Poisson equation
  wrk2 : working area, OMEGA : relaxation parameter
  BND:control variable for boundaries and objects ( = 0 or 1)
  P: pressure
  ------------------
  OmpSs-2@Cluster version written by Ioannis Anevlavis - Eta Scale AB
********************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include "memory.hpp"

#define BSIZE_UNIT 64

/** @brief: problem size options */
#define XS 0
#define S  1
#define M  2
#define L  3
#define XL 4

/** @brief: set matrix dimensions */
	#define MNUMS 4
#if   PROBLEM_SIZE == XS
	#define MROWS 32
	#define MCOLS 32
	#define MDEPS 64
#elif PROBLEM_SIZE == S
	#define MROWS 64
	#define MCOLS 64
	#define MDEPS 128
#elif PROBLEM_SIZE == M
	#define MROWS 128
	#define MCOLS 128
	#define MDEPS 256
#elif PROBLEM_SIZE == L
	#define MROWS 256
	#define MCOLS 256
	#define MDEPS 512
#elif PROBLEM_SIZE == XL
	#define MROWS 512
	#define MCOLS 512
	#define MDEPS 1024
#else
	#define MROWS 64
	#define MCOLS 64
	#define MDEPS 128
#endif

struct Mat {
	float* m;
	int mnums;
	int mrows;
	int mcols;
	int mdeps;
};

/* prototypes */
typedef struct Mat Matrix;

void clearMat(Matrix* Mat);
void set_param(int i[],
		char *size);
void mat_set(Matrix* Mat,
		int l,
		float z);
void mat_set_init(Matrix* Mat);
void write_to_file(Matrix* Mat);
void task_chunk(int& beg,
		int& end,
		int& chunk,
		const int& size,
		const int& index,
		const int& bsize);
int newMat(Matrix* Mat,
		int mnums,
		int mrows,
		int mcols,
		int mdeps);
float jacobi(int n,
		Matrix* M1,
		Matrix* M2,
		Matrix* M3,
		Matrix* M4,
		Matrix* M5,
		Matrix* M6,
		Matrix* M7);
double second();

int    BSIZE;
float  *ggosa;
float  omega=0.8;
Matrix a,b,c,p,bnd,wrk1,wrk2;

int
main(int argc, char *argv[])
{
	int    imax,jmax,kmax,mimax,mjmax,mkmax,msize[3];
	float  gosa,target;
	double cpu0,cpu1,cpu,xmflops2,score,flop;
	char   size[10];
	
	/* global scalar allocation */
	ggosa = dmalloc<float>(1);

	if(argc > 1){
		BSIZE = atoi(argv[1]);
	} else {
		BSIZE = BSIZE_UNIT;
	}

	mimax= MROWS;
	mjmax= MCOLS;
	mkmax= MDEPS;
	imax= mimax-1;
	jmax= mjmax-1;
	kmax= mkmax-1;

	target = 60.0;

	printf("mimax = %d mjmax = %d mkmax = %d\n",mimax,mjmax,mkmax);
	printf("imax = %d jmax = %d kmax =%d\n",imax,jmax,kmax);

	/*
	 *    Initializing matrixes
	 */
	newMat(&p,1,mimax,mjmax,mkmax);
	newMat(&bnd,1,mimax,mjmax,mkmax);
	newMat(&wrk1,1,mimax,mjmax,mkmax);
	newMat(&wrk2,1,mimax,mjmax,mkmax);
	newMat(&a,4,mimax,mjmax,mkmax);
	newMat(&b,3,mimax,mjmax,mkmax);
	newMat(&c,3,mimax,mjmax,mkmax);

	mat_set_init(&p);
	mat_set(&bnd,0,1.0);
	mat_set(&wrk1,0,0.0);
	mat_set(&wrk2,0,0.0);
	mat_set(&a,0,1.0);
	mat_set(&a,1,1.0);
	mat_set(&a,2,1.0);
	mat_set(&a,3,1.0/6.0);
	mat_set(&b,0,0.0);
	mat_set(&b,1,0.0);
	mat_set(&b,2,0.0);
	mat_set(&c,0,1.0);
	mat_set(&c,1,1.0);
	mat_set(&c,2,1.0);

	/*
	 * Start measuring
	 */
	int nn = 3;
	printf(" Start rehearsal measurement process.\n");
	printf(" Measure the performance in %d times.\n\n",nn);

	cpu0= second();
	gosa= jacobi(nn,&a,&b,&c,&p,&bnd,&wrk1,&wrk2);
	cpu1= second();
	cpu= cpu1 - cpu0;
	flop = (double)(kmax-1)*(double)(jmax-1)*(double)(imax-1)*34.0;
	
	if(cpu != 0.0)
		xmflops2= flop/cpu*1.e-6*(nn);
	printf(" MFLOPS: %f time(s): %f %e\n\n",xmflops2,cpu,gosa);

	/** @note: for verification, set this to a specific number */
	nn = (int)(target/(cpu/3.0));
	printf(" Now, start the actual measurement process.\n");
	printf(" The loop will be excuted in %d times\n",nn);
	printf(" This will take about one minute.\n");
	printf(" Wait for a while\n\n");

	cpu0 = second();
	gosa = jacobi(nn,&a,&b,&c,&p,&bnd,&wrk1,&wrk2);
	cpu1 = second();
	cpu = cpu1 - cpu0;
	
	if(cpu != 0.0)
		xmflops2 = (double)flop/cpu*1.0e-6*(nn);

	printf("cpu : %f sec.\n", cpu);
	printf("Loop executed for %d times\n",nn);
	printf("Gosa : %e \n",gosa);
	printf("MFLOPS measured : %f\n",xmflops2);
	score = xmflops2/82.84;
	printf("Score based on Pentium III 600MHz using Fortran 77: %f\n",score);

	/*
	 * Generate output file for verification
	 */
#ifdef ENABLE_OUTPUT
	#pragma oss task in(p.m[0;mimax*mjmax*mkmax])		\
			 node(nanos6_cluster_no_offload)	\
			 label("master: correctness check")
	write_to_file(&p);
	#pragma oss taskwait
#endif

	/* global scalar deallocation */
	dfree<float>(ggosa, 1);

	/*
	 *   Matrix free
	 */ 
	clearMat(&p);
	clearMat(&bnd);
	clearMat(&wrk1);
	clearMat(&wrk2);
	clearMat(&a);
	clearMat(&b);
	clearMat(&c);

	return 0;
}

void
set_param(int is[],char *size)
{
	if(!strcmp(size,"XS") || !strcmp(size,"xs")){
		is[0]= 32;
		is[1]= 32;
		is[2]= 64;
		return;
	}
	if(!strcmp(size,"S") || !strcmp(size,"s")){
		is[0]= 64;
		is[1]= 64;
		is[2]= 128;
		return;
	}
	if(!strcmp(size,"M") || !strcmp(size,"m")){
		is[0]= 128;
		is[1]= 128;
		is[2]= 256;
		return;
	}
	if(!strcmp(size,"L") || !strcmp(size,"l")){
		is[0]= 256;
		is[1]= 256;
		is[2]= 512;
		return;
	}
	if(!strcmp(size,"XL") || !strcmp(size,"xl")){
		is[0]= 512;
		is[1]= 512;
		is[2]= 1024;
		return;
	}
}

int
newMat(Matrix* Mat, int mnums,int mrows, int mcols, int mdeps)
{
	Mat->mnums = mnums;
	Mat->mrows = mrows;
	Mat->mcols = mcols;
	Mat->mdeps = mdeps;
	Mat->m     = NULL;
	Mat->m     = dmalloc<float>(mnums * mrows * mcols * mdeps);

	return(Mat->m != NULL) ? 1:0;
}

void
clearMat(Matrix* Mat)
{
	if(Mat->m)
		dfree<float>(Mat->m, Mat->mnums * Mat->mcols * Mat->mrows * Mat->mdeps);
	Mat->m     = NULL;
	Mat->mnums = 0;
	Mat->mcols = 0;
	Mat->mrows = 0;
	Mat->mdeps = 0;
}

void
mat_set(Matrix* Mat, int l, float val)
{
	/** @note: set preprocessor dimensions to bypass mcxx compilation error */
	float (*mat)[MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])Mat->m;

	int    mnums = Mat->mnums;
	int    mrows = Mat->mrows;
	int    mcols = Mat->mcols;
	int    mdeps = Mat->mdeps;

	for(int i=0; i<mrows; i+=BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, mrows, i, BSIZE);

		#pragma oss task out(mat[l][beg:end-1][0;mcols][0;mdeps]) 	\
				 firstprivate(beg, end, l, val, mcols, mdeps)	\
				 label("remote: initialize row chunk in `mat`")
		for(int i=beg; i<end; i++)
			for(int j=0; j<mcols; j++)
				for(int k=0; k<mdeps; k++)
					mat[l][i][j][k] = val;
	}
}

void
mat_set_init(Matrix* Mat)
{
	/** @note: set preprocessor dimensions to bypass mcxx compilation error */
	float (*mat)[MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])Mat->m;

	int    mnums = Mat->mnums;
	int    mrows = Mat->mrows;
	int    mcols = Mat->mcols;
	int    mdeps = Mat->mdeps;

	for(int i=0; i<mrows; i+=BSIZE) {
		int beg, end, chunk;
		task_chunk(beg, end, chunk, mrows, i, BSIZE);

		#pragma oss task out(mat[0][beg:end-1][0;mcols][0;mdeps]) 	\
				 firstprivate(beg, end, mrows, mcols, mdeps)	\
				 label("remote: initialize row chunk in `mat`")
		for(int i=beg; i<end; i++)
			for(int j=0; j<mcols; j++)
				for(int k=0; k<mdeps; k++)
					mat[0][i][j][k] = (float)(i*i)
						/(float)((mrows - 1)*(mrows - 1));
	}
}

float
jacobi(int nn, Matrix* a,Matrix* b,Matrix* c,
		Matrix* p,Matrix* bnd,Matrix* wrk1,Matrix* wrk2)
{
	/** @note: set preprocessor dimensions to bypass mcxx compilation error */
	float (*mat_a)   [MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])a->m;
	float (*mat_b)   [MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])b->m;
	float (*mat_c)   [MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])c->m;
	float (*mat_p)   [MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])p->m;
	float (*mat_bnd) [MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])bnd->m;
	float (*mat_wrk1)[MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])wrk1->m;
	float (*mat_wrk2)[MROWS][MCOLS][MDEPS] = (float(*)[MROWS][MCOLS][MDEPS])wrk2->m;

	int    imax = p->mrows-1;
	int    jmax = p->mcols-1;
	int    kmax = p->mdeps-1;

	for(int n=0 ; n<nn ; n++){
		#pragma oss task out(*ggosa)
			*ggosa = 0.0;
		
		for(int i=1; i<imax; i+=BSIZE) {
			int beg, end, chunk;
			task_chunk(beg, end, chunk, imax, i, BSIZE);

			#pragma oss task in( mat_a [0;4][beg:end-1][1;jmax][1;kmax],	\
					     mat_b [0;3][beg:end-1][1;jmax][1;kmax],	\
					     mat_c [0;3][beg:end-1][1;jmax][1;kmax],	\
					     mat_p   [0][beg-1:end][0:jmax][0:kmax],	\
					     mat_wrk1[0][beg:end-1][1;jmax][1;kmax],	\
					     mat_bnd [0][beg:end-1][1;jmax][1;kmax])	\
					 out(mat_wrk2[0][beg:end-1][1;jmax][1;kmax])	\
					 inout(*ggosa)					\
					 firstprivate(beg, end, jmax, kmax, omega)	\
					 label("remote: calculate himeno chunk")
			for(int i=beg; i<end; i++)
				for(int j=1; j<jmax; j++)
					for(int k=1; k<kmax; k++){
						float s0 =mat_a[0][i][j][k] * mat_p[0][i+1][j][k]
							+ mat_a[1][i][j][k] * mat_p[0][i][j+1][k]
							+ mat_a[2][i][j][k] * mat_p[0][i][j][k+1]
							+ mat_b[0][i][j][k]
							*( mat_p[0][i+1][j+1][k] - mat_p[0][i+1][j-1][k]
							 - mat_p[0][i-1][j+1][k] + mat_p[0][i-1][j-1][k] )
							+ mat_b[1][i][j][k]
							*( mat_p[0][i][j+1][k+1] - mat_p[0][i][j-1][k+1]
							 - mat_p[0][i][j+1][k-1] + mat_p[0][i][j-1][k-1] )
							+ mat_b[2][i][j][k]
							*( mat_p[0][i+1][j][k+1] - mat_p[0][i-1][j][k+1]
							 - mat_p[0][i+1][j][k-1] + mat_p[0][i-1][j][k-1] )
							+ mat_c[0][i][j][k] * mat_p[0][i-1][j][k]
							+ mat_c[1][i][j][k] * mat_p[0][i][j-1][k]
							+ mat_c[2][i][j][k] * mat_p[0][i][j][k-1]
							+ mat_wrk1[0][i][j][k];

						float ss = (s0*mat_a[3][i][j][k] - mat_p[0][i][j][k]) * mat_bnd[0][i][j][k];
						*ggosa += ss*ss;
						mat_wrk2[0][i][j][k] = mat_p[0][i][j][k] + omega*ss;
					}
		}
		
		for(int i=1; i<imax; i+=BSIZE) {
			int beg, end, chunk;
			task_chunk(beg, end, chunk, imax, i, BSIZE);

			#pragma oss task in( mat_wrk2[0][beg:end-1][1;jmax][1;kmax])	\
					 out(mat_p   [0][beg:end-1][1;jmax][1;kmax])	\
					 firstprivate(beg, end, jmax, kmax)		\
					 label("remote: calculate himeno chunk")
			for(int i=beg; i<end; i++)
				for(int j=1; j<jmax; j++)
					for(int k=1; k<kmax; k++)
						mat_p[0][i][j][k] = mat_wrk2[0][i][j][k];
		}
	} /* end n loop */
	#pragma oss taskwait

	return *ggosa;
}

double
second()
{
	struct timeval tm;
	double t;

	static int base_sec = 0, base_usec = 0;

	gettimeofday(&tm, NULL);

	if(base_sec == 0 && base_usec == 0)
	{
		base_sec = tm.tv_sec;
		base_usec = tm.tv_usec;
		t = 0.0;
	} else {
		t = (double) (tm.tv_sec-base_sec) + 
			((double) (tm.tv_usec-base_usec))/1.0e6;
	}

	return t;
}

void
write_to_file(Matrix* Mat)
{
	int i, rv;
	FILE *file;
	char *outputFile = (char*)"out.himeno";

	file = fopen(outputFile, "w");
	if(file == NULL) {
		printf("ERROR: Unable to open file `%s'.\n", outputFile);
		exit(1);
	}
	for (i = 0; i < MROWS*MCOLS*MDEPS; i++) {
		rv = fprintf(file,"p.m[%d]: %f\n", i, Mat->m[i]);
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

void
task_chunk(int& beg, int& end, int& chunk,
		const int& size, const int& index, const int& bsize)
{
	chunk = (size - index > bsize) ? bsize : size - index;
	beg = index;
	end = index + chunk;
}

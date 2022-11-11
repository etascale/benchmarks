/*
 * Copyright (C) 2008 Princeton University
 * All rights reserved.
 * Authors: Jia Deng, Gilberto Contreras
 * OmpSs/OpenMP 4.0 versions written by Raul Vidal and Dimitrios Chasapis - Barcelona Supercomputing Center
 * OmpSs-2 version written by Ioannis Anevlavis - Eta Scale AB
 *
 * streamcluster - Online clustering algorithm
 *
 * Current ompss code is using the points array as input/output, but should also add the Points struct or 
 * declare it as shared and put the corresponding copy_deps clause, or change variable names to avoid 
 * deferring the Points struct.
 * 
 */
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <sys/resource.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

using namespace std;

#define DEFAULT_NDIVS 8
#define MAXNAMESIZE 1024 // max filename length
#define SEED 1
/* increase this to reduce probability of random error */
/* increasing it also ups running time of "speedy" part of the code */
/* SP = 1 seems to be fine */
#define SP 1 // number of repetitions of speedy must be >=1

/* higher ITER --> more likely to get correct # of centers */
/* higher ITER also scales the running time almost linearly */
#define ITER 3 // iterate ITER* k log k times; ITER >= 1

#define CACHE_LINE 32 // cache line in byte

int NUMANODES; /* Indicates the number of NUMA nodes (sockets)*/

/* this structure represents a point */
/* these will be passed around to avoid copying coordinates */
typedef struct {
	float weight;
	float *coord;
	long assign;  /* number of point where this one is assigned */
	float cost;  /* cost of that assignment, weight*distance */
} Point;

/* this is the array of points */
typedef struct {
	long num; /* number of points; may not be N if this is a sample */
	int dim;  /* dimensionality */
	Point *p; /* the array itself */
} Points;

static bool* switch_membership; //whether to switch membership in pgain
static bool* is_center; //whether a point is a center
static int* center_table; //index table of centers
static double* gl_lower;
static int nproc; //# of threads

double getusec_()
{
	struct timeval time;
	gettimeofday(&time, 0);
	return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
}



float dist(Point p1, Point p2, int dim);


/********************************************/



int isIdentical(float *i, float *j, int D)
	// tells whether two points of D dimensions are identical
{
	int a = 0;
	int equal = 1;

	while (equal && a < D) {
		if (i[a] != j[a]) equal = 0;
		else a++;
	}
	if (equal) return 1;
	else return 0;

}

/* comparator for floating point numbers */
static int floatcomp(const void *i, const void *j)
{
	float a, b;
	a = *(float *)(i);
	b = *(float *)(j);
	if (a > b) return (1);
	if (a < b) return (-1);
	return(0);
}

/* shuffle points into random order */
void shuffle(Points *points)
{
	long i, j;
	Point temp;
	for (i=0;i<points->num-1;i++) {
		j=(lrand48()%(points->num - i)) + i;
		temp = points->p[i];
		points->p[i] = points->p[j];
		points->p[j] = temp;
	}
}

/* shuffle an array of integers */
void intshuffle(int *intarray, int length)
{
	long i, j;
	int temp;
	for (i=0;i<length;i++) {
		j=(lrand48()%(length - i))+i;
		temp = intarray[i];
		intarray[i]=intarray[j];
		intarray[j]=temp;
	}
}

/* compute Euclidean distance squared between two points */
float dist(Point p1, Point p2, int dim)
{
	int i;
	float result=0.0;
	for (i=0;i<dim;i++)
		result += (p1.coord[i] - p2.coord[i])*(p1.coord[i] - p2.coord[i]);
	return(result);
}


float pspeedy(Points *points, float z, long *kcenter)
{
	//my block
	long bsize = points->num/nproc;
	long k1;
	long k2;

	static double totalcost;
	static int i;

	Point* po = points->p;
	/* create center at first point, send it to itself */
	for (int p = 0; p < nproc; p++) {
		k1 = bsize*p; k2 = k1 + bsize; if (p == nproc-1) k2 = points->num;
		// nanos_current_socket(p%NUMANODES);
#pragma oss task inout(po[k1:k2]) firstprivate(k1,k2) label("PSPEEDY-FIRST-CENTER")
		{
			for( int k = k1; k < k2; k++ )    {
				float distance = dist(points->p[k],points->p[0],points->dim);
				points->p[k].cost = distance * points->p[k].weight;
				points->p[k].assign=0;
			}
		}
	}
#pragma oss taskwait
	*kcenter = 1;

	// I am the master thread. I decide whether to open a center and notify others if so.

	for(i = 1; i < points->num; i++ )  {
		bool to_open = ((float)lrand48()/(float)INT_MAX)<(points->p[i].cost/z);
		if( to_open )  {
			(*kcenter)++;
			//open = true;
			for (int p = 0; p < nproc; p++) {
				k1 = p*bsize; k2 = k1 + bsize; if (p == nproc-1) k2 = points->num;
				// nanos_current_socket(p%NUMANODES);
#pragma oss task inout(po[k1:k2]) firstprivate(k1,k2) label("PSPEEDY-OPEN-CENTERS")
				{
					for( int k = k1; k < k2; k++ )  {
						float distance = dist(points->p[i],points->p[k],points->dim);
						if( distance*points->p[k].weight < points->p[k].cost )  {
							points->p[k].cost = distance * points->p[k].weight;
							points->p[k].assign=i;
						}
					}
				}
			}
#pragma oss taskwait
		}
	}
	//reduction of the previous loop
	double mytotal = 0;
	totalcost=z*(*kcenter);
	for (int p = 0; p < nproc; p++) {
		k1 = p*bsize; k2 = k1 + bsize; if (p == nproc-1) k2 = points->num;
		//this task changes native output. The atomic might be the culprit
		//#pragma oss task inout(po[k1:k2]) firstprivate(k1,k2,mytotal,p) label("PSPEEDY-COST-REDUCTION")
		//{ 
		mytotal = 0;
		for( int k = k1; k < k2; k++ )  {
			mytotal += points->p[k].cost;
		}
		//#pragma oss atomic
		totalcost += mytotal;
		//}
	}

	return(totalcost);
}



/* For a given point x, find the cost of the following operation:
 * -- open a facility at x if there isn't already one there,
 * -- for points y such that the assignment distance of y exceeds dist(y, x),
 *    make y a member of x,
 * -- for facilities y such that reassigning y and all its members to x 
 *    would save cost, realize this closing and reassignment.
 * 
 * If the cost of this operation is negative (i.e., if this entire operation
 * saves cost), perform this operation and return the amount of cost saved;
 * otherwise, do nothing.
 */

/* numcenters will be updated to reflect the new number of centers */
/* z is the facility cost, x is the number of this point in the array 
   points */




double pgain(long x, Points *points, double z, long int *numcenters)
{
	//  printf("pgain pthread %d begin\n",pid);

	//my block
	long bsize = points->num/nproc;
	long k1; //= bsize * pid;
	long k2; //= k1 + bsize;
	//if( pid == nproc-1 ) k2 = points->num;
	Point* po = points->p;
	unsigned long num_points = points->num;
	int i;
	//global lower fields
	double gl_cost_of_opening_x = 0; //removed static qualifier
	int gl_number_of_centers_to_close = 0; //removed static qualifier

	/*For each center, we have a *lower* field that indicates 
	  how much we will save by closing the center. 
	  Each thread has its own copy of the *lower* fields as an array.
	  We first build a table to index the positions of the *lower* fields. 
	  */
	//Build table of indexes for centers. As at each iteration the centers
	//are less, pgain does not need so much memory, so dependencies can use
	//less space.
	int count = 0;
	for (int i = 0; i < num_points; i++) {
		if (is_center[i]) {
			center_table[i] = count++;
		}
	}

	memset(gl_lower, 0, count*sizeof(double));
	for (int p = 0; p < nproc; p++) {
		k1 = bsize*p; k2 = k1 + bsize; if (p == nproc-1) k2 = points->num;
		//#pragma oss target device(smp) copy_deps(ended)
		// nanos_current_socket(p%NUMANODES);
#pragma oss task concurrent([num_points]po,[count]gl_lower) \
		inout(switch_membership[k1:k2]) \
		firstprivate(k1,k2,p,count) shared(gl_cost_of_opening_x,gl_number_of_centers_to_close) label("PGAIN-UPDT-COSTS")
		{
			//int thid = omp_get_thread_num();
			//fprintf(stderr,"Task of processor %d is being executed by thread %d\n",p,thid);
			int stride = count;
			int cl = CACHE_LINE/sizeof(double);

			if( stride % cl != 0 ) { 
				stride = cl * ( stride / cl + 1);
			}
			memset(switch_membership + k1, 0, (k2-k1)*sizeof(bool));
			double local_cost_of_opening_x = 0.;
			double* local_gl_lower;
			local_gl_lower = (double*) calloc(stride,sizeof(double));

			for (int i = k1; i < k2; i++ ) {
				float x_cost = dist(points->p[i], points->p[x], points->dim) 
					* points->p[i].weight;
				float current_cost = points->p[i].cost;
				if ( x_cost < current_cost ) {
					// point i would save cost just by switching to x
					// (note that i cannot be a median, 
					// or else dist(p[i], p[x]) would be 0)

					switch_membership[i] = 1;
					local_cost_of_opening_x += (x_cost - current_cost);

				} else {

					// cost of assigning i to x is at least current assignment cost of i

					// consider the savings that i's **current** median would realize
					// if we reassigned that median and all its members to x;
					// note we've already accounted for the fact that the median
					// would save z by closing; now we have to subtract from the savings
					// the extra cost of reassigning that median and its members 
					int assign = points->p[i].assign;
					local_gl_lower[center_table[assign]] += current_cost - x_cost;
				}
			}
			for (int i = 0; i < count; i++) {
#pragma oss atomic
				gl_lower[i] += local_gl_lower[i];
			}
#pragma oss atomic
			gl_cost_of_opening_x += local_cost_of_opening_x;

			free(local_gl_lower);
		}
	}
#pragma oss taskwait
	// at this time, we can calculate the cost of opening a center
	// at x; if it is negative, we'll go through with opening it
	for (int i = 0; i < num_points; i++) {
		if (is_center[i]) {
			gl_lower[center_table[i]] += z;
			if (gl_lower[center_table[i]] > 0) {
				++gl_number_of_centers_to_close;
				gl_cost_of_opening_x -= gl_lower[center_table[i]];
			}
		}
	}
	gl_cost_of_opening_x += z;
	// Now, check whether opening x would save cost; if so, do it, and
	// otherwise do nothing
	if ( gl_cost_of_opening_x < 0 ) {
		//  we'd save money by opening x; we'll do it
		for (int p = 0; p < nproc; p++) {
			k1 = bsize*p; k2 = k1 + bsize; if (p == nproc-1) k2 = points->num;
			// nanos_current_socket(p%NUMANODES);
#pragma oss task inout(po[k1:k2]) in(switch_membership[k1:k2],[count]gl_lower) label("PGAIN-SAVE-MONEY")
			{
				for ( int i = k1; i < k2; i++ ) {
					bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
					if ( switch_membership[i] || close_center ) {
						// Either i's median (which may be i itself) is closing,
						// or i is closer to x than to its current median
						points->p[i].cost = points->p[i].weight *
							dist(points->p[i], points->p[x], points->dim);
						points->p[i].assign = x;
					}
				}
				for( int i = k1; i < k2; i++ ) {
					if( is_center[i] && gl_lower[center_table[i]] > 0 ) {
						is_center[i] = false;
					}
				}
				if( x >= k1 && x < k2 ) {
					is_center[x] = true;
				}
			}
		}
#pragma oss taskwait
		*numcenters = *numcenters + 1 - gl_number_of_centers_to_close;
	}
	else {
		gl_cost_of_opening_x = 0;  // the value we'll return
	}

	return -gl_cost_of_opening_x;
}




/* facility location on the points using local search */
/* z is the facility cost, returns the total cost and # of centers */
/* assumes we are seeded with a reasonable solution */
/* cost should represent this solution's cost */
/* halt if there is < e improvement after iter calls to gain */
/* feasible is an array of numfeasible points which may be centers */

float pFL(Points *points, int *feasible, int numfeasible,
		float z, long *k, double cost, long iter, float e)
{
	long i;
	long x;
	double change;
	long numberOfPoints;
	numberOfPoints = points->num;

	/* continue until we run iter iterations without improvement */
	/* stop instead if improvement is less than e */

	change = cost;
	while (change/cost > 1.0*e) {
		change = 0.0;

		/* randomize order in which centers are considered */

		intshuffle(feasible, numfeasible);
		for (i=0;i<iter;i++) {
			x = i%numfeasible;
			//z == facility cost, k == numcenters, feasible == array of possible centers
			change += pgain(feasible[x], points, z, k);
		}
		cost -= change;
	}
	return(cost);
}




int selectfeasible_fast(Points *points, int **feasible, int kmin)
{
	int numfeasible = points->num;
	if (numfeasible > (ITER*kmin*log((double)kmin)))
		numfeasible = (int)(ITER*kmin*log((double)kmin));
	*feasible = (int *)malloc(numfeasible*sizeof(int));

	float* accumweight;
	float totalweight;

	/* 
	   Calcuate my block. 
	   For now this routine does not seem to be the bottleneck, so it is not parallelized. 
	   When necessary, this can be parallelized by setting k1 and k2 to 
	   proper values and calling this routine from all threads ( it is called only
	   by thread 0 for now ). 
	   Note that when parallelized, the randomization might not be the same and it might
	   not be difficult to measure the parallel speed-up for the whole program. 
	   */
	//  long bsize = numfeasible;
	long k1 = 0;
	long k2 = numfeasible;

	float w;
	int l,r,k;

	/* not many points, all will be feasible */
	if (numfeasible == points->num) {
		for (int i=k1;i<k2;i++)
			(*feasible)[i] = i;
		return numfeasible;
	}
	accumweight= (float*)malloc(sizeof(float)*points->num);

	accumweight[0] = points->p[0].weight;
	totalweight=0;
	for( int i = 1; i < points->num; i++ ) {
		accumweight[i] = accumweight[i-1] + points->p[i].weight;
	}
	totalweight=accumweight[points->num-1];

	for(int i=k1; i<k2; i++ ) {
		w = (lrand48()/(float)INT_MAX)*totalweight;
		//binary search
		l=0;
		r=points->num-1;
		if( accumweight[0] > w )  { 
			(*feasible)[i]=0; 
			continue;
		}
		while( l+1 < r ) {
			k = (l+r)/2;
			if( accumweight[k] > w ) {
				r = k;
			} 
			else {
				l=k;
			}
		}
		(*feasible)[i]=r;
	}

	free(accumweight); 

	return numfeasible;
}




/* compute approximate kmedian on the points */
float pkmedian(Points *points, long kmin, long kmax, long* kfinal)
{
	int i;
	double cost;
	double lastcost;
	double hiz, loz, z;

	static long k;
	static int *feasible;
	static int numfeasible;
	static double* hizs;

	hiz = loz = 0.0;
	long numberOfPoints = points->num;
	long ptDimension = points->dim;

	//my block
	long bsize = points->num/nproc;
	long k1;
	long k2;

	/*printf("BEGINNING OF PKMEDIAN\n");
	  printf("bsize: %d ---- k1: %d ---- k2: %d\n",bsize,k1,k2);
	  for (int j = 0; j < points->num; j++) {
	  printf("Point %d ---- \n",j);
	  for (k = 0; k < points->dim; k++) {
	  printf("\tCoord %d: %f\n",k,points->p[j].coord[k]);
	  }
	  printf("\tWeight: %f\n",points->p[j].weight);
	  printf("\tAssign: %f\n",points->p[j].assign);
	  printf("\tCost: %f\n",points->p[j].cost);

	  }*/
	Point* po = points->p;
	for (int p = 0; p < nproc; p++) {
		//Point& is only 1 point, not an array
		k1 = bsize * p; k2 = k1 + bsize; 
		if (p == nproc-1) k2 = points->num;
		// nanos_current_socket(p%NUMANODES);
#pragma oss task in(po[k1:k2]) concurrent(hiz) firstprivate(k1,k2) label("PKMEDIAN-HIZ")
		{
			double myhiz = 0;
			for (long kk=k1;kk < k2; kk++ ) {
				myhiz += dist(points->p[kk], points->p[0],
						ptDimension)*points->p[kk].weight;
			}
#pragma oss atomic
			hiz += myhiz;
		}
	}

	//#pragma oss taskwait

	/* NEW: Check whether more centers than points! */
	if (points->num <= kmax) {
		/* just return all points as facilities */
		for (int p = 0; p < nproc; p++) {
			k1 = bsize * p; k2 = k1 + bsize; 
			if (p == nproc-1) k2 = points->num;
			// nanos_current_socket(p%NUMANODES);
#pragma oss task inout(po[k1:k2]) firstprivate(k1,k2) label("PKMEDIAN-ALL-FL")
			{
				for (long kk=k1;kk<k2;kk++) {
					points->p[kk].assign = kk;
					points->p[kk].cost = 0;
				}
			}
		}
#pragma oss taskwait

		cost = 0;
		*kfinal = k;
		return cost;
	}
#pragma oss taskwait
	loz=0.0; z = (hiz+loz)/2.0;

	shuffle(points);
	cost = pspeedy(points, z, &k);
	//printf("Cost after pspeedy 1: %f\n",cost);
	i=0;
	/* give speedy SP chances to get at least kmin/2 facilities */
	while ((k < kmin)&&(i<SP)) {
		cost = pspeedy(points, z, &k);
		i++;
	}

	//printf("Cost after pspeedy 2: %f\n",cost);
	/* if still not enough facilities, assume z is too high */
	while (k < kmin) {
		if (i >= SP) {hiz=z; z=(hiz+loz)/2.0; i=0;}
		shuffle(points);
		cost = pspeedy(points, z, &k);
		i++;
	}

	//printf("Cost after pspeedy 3: %f\n",cost);
	/* now we begin the binary search for real */
	/* must designate some points as feasible centers */
	/* this creates more consistancy between FL runs */
	/* helps to guarantee correct # of centers at the end */

	numfeasible = selectfeasible_fast(points,&feasible,kmin);
	for( int i = 0; i< points->num; i++ ) {
		is_center[points->p[i].assign]= true;
	}


	while(1) {
		/* first get a rough estimate on the FL solution */
		lastcost = cost;
		cost = pFL(points, feasible, numfeasible,
				z, &k, cost, (long)(ITER*kmax*log((double)kmax)), 0.1);

		//printf("Cost after pFL 1: %f\n",cost);
		/* if number of centers seems good, try a more accurate FL */
		if (((k <= (1.1)*kmax)&&(k >= (0.9)*kmin))||
				((k <= kmax+2)&&(k >= kmin-2))) {

			/* may need to run a little longer here before halting without
			   improvement */
			cost = pFL(points, feasible, numfeasible,
					z, &k, cost, (long)(ITER*kmax*log((double)kmax)), 0.001);
		}

		//printf("Cost after pFL 2: %f\n",cost);
		if (k > kmax) {
			/* facilities too cheap */
			/* increase facility cost and up the cost accordingly */
			loz = z; z = (hiz+loz)/2.0;
			cost += (z-loz)*k;
		}
		if (k < kmin) {
			/* facilities too expensive */
			/* decrease facility cost and reduce the cost accordingly */
			hiz = z; z = (hiz+loz)/2.0;
			cost += (z-hiz)*k;
		}

		/* if k is good, return the result */
		/* if we're stuck, just give up and return what we have */
		if (((k <= kmax)&&(k >= kmin))||((loz >= (0.999)*hiz)) )
		{ 
			break;
		}
	}
	/*printf("ENDING OF PKMEDIAN\n");
	  printf("bsize: %d ---- k1: %d ---- k2: %d\n",bsize,k1,k2);
	  for (int j = 0; j < points->num; j++) {
	  printf("Point %d ---- \n",j);
	  for (k = 0; k < points->dim; k++) {
	  printf("\tCoord %d: %f\n",k,points->p[j].coord[k]);
	  }
	  printf("\tWeight: %f\n",points->p[j].weight);
	  printf("\tAssign: %f\n",points->p[j].assign);
	  printf("\tCost: %f\n",points->p[j].cost);

	  }*/
	//clean up...
	free(feasible); 
	*kfinal = k;

	return cost;
}





/* compute the means for the k clusters */
int contcenters(Points *points)
{
	long i, ii;
	float relweight;

	for (i=0;i<points->num;i++) {
		/* compute relative weight of this point to the cluster */
		if (points->p[i].assign != i) {
			relweight=points->p[points->p[i].assign].weight + points->p[i].weight;
			relweight = points->p[i].weight/relweight;
			for (ii=0;ii<points->dim;ii++) {
				points->p[points->p[i].assign].coord[ii] *= 1.0-relweight;
				points->p[points->p[i].assign].coord[ii] += points->p[i].coord[ii]*relweight;
			}
			points->p[points->p[i].assign].weight += points->p[i].weight;
		}
	}

	return 0;
}

/* copy centers from points to centers */
void copycenters(Points *points, Points* centers, long* centerIDs, long offset)
{
	long i;
	long k;

	bool *is_a_median = (bool *) calloc(points->num, sizeof(bool));

	/* mark the centers */
	for ( i = 0; i < points->num; i++ ) {
		is_a_median[points->p[i].assign] = 1;
	}

	k=centers->num;

	/* count how many  */
	for ( i = 0; i < points->num; i++ ) {
		if ( is_a_median[i] ) {
			memcpy( centers->p[k].coord, points->p[i].coord, points->dim * sizeof(float));
			centers->p[k].weight = points->p[i].weight;
			centerIDs[k] = i + offset;
			k++;
		}
	}

	centers->num = k;

	free(is_a_median);
}

struct pkmedian_arg_t
{
	Points* points;
	long kmin;
	long kmax;
	long* kfinal;
};

void* localSearchSub(void* arg_) {

	pkmedian_arg_t* arg= (pkmedian_arg_t*)arg_;
	pkmedian(arg->points,arg->kmin,arg->kmax,arg->kfinal);

	return NULL;
}


void localSearch( Points* points, long kmin, long kmax, long* kfinal ) {
	pkmedian_arg_t* arg = new pkmedian_arg_t;//[nproc];

	arg->points = points;
	arg->kmin = kmin;
	arg->kmax = kmax;
	arg->kfinal = kfinal;

	localSearchSub(arg);


	delete[] arg;
}


class PStream {
	public:
		virtual size_t read( float* dest, int dim, int num ) = 0;
		virtual int ferror() = 0;
		virtual int feof() = 0;
		virtual ~PStream() {
		}
};

//synthetic stream
class SimStream : public PStream {
	public:
		SimStream(long n_ ) {
			n = n_;
		}
		size_t read( float* dest, int dim, int num ) {
			size_t count = 0;
			long k1; long k2; long bsize = num/nproc;
			for (int p = 0; p < nproc; p++)
			{
				k1 = p*bsize; k2 = k1+bsize; if (p == nproc-1) k2 = num;
				// nanos_current_socket(p%NUMANODES);
#pragma oss task out(dest[k1*dim:k2*dim]) firstprivate(k1,k2) label("SIMSTREAM-COORD-INIT") concurrent(count)
				{
					//int thid = omp_get_thread_num();
					//fprintf(stderr,"Task of processor %d is being executed by thread %d\n",p,thid);
					for( int i = k1; i < k2 && n > 0; i++ ) {
						for( int k = 0; k < dim; k++ ) {
							dest[i*dim + k] = lrand48()/(float)INT_MAX;
						}
						n--;
#pragma oss atomic
						count++;
					}
				}
			}
#pragma oss taskwait
			return count;
		}
		int ferror() {
			return 0;
		}
		int feof() {
			return n <= 0;
		}
		~SimStream() { 
		}
	private:
		long n;
};

class FileStream : public PStream {
	public:
		FileStream(char* filename) {
			fp = fopen( filename, "rb");
			if( fp == NULL ) {
				fprintf(stderr,"error opening file %s\n.",filename);
				exit(1);
			}
		}
		size_t read( float* dest, int dim, int num ) {
			return std::fread(dest, sizeof(float)*dim, num, fp); 
		}
		int ferror() {
			return std::ferror(fp);
		}
		int feof() {
			return std::feof(fp);
		}
		~FileStream() {
			fprintf(stderr,"closing file stream\n");
			fclose(fp);
		}
	private:
		FILE* fp;
};

void outcenterIDs( Points* centers, long* centerIDs, char* outfile ) {
	FILE* fp = fopen(outfile, "w");
	if( fp==NULL ) {
		fprintf(stderr, "error opening %s\n",outfile);
		exit(1);
	}
	int* is_a_median = (int*)calloc( sizeof(int), centers->num );
	for( int i =0 ; i< centers->num; i++ ) {
		is_a_median[centers->p[i].assign] = 1;
	}

	for( int i = 0; i < centers->num; i++ ) {
		if( is_a_median[i] ) {
			fprintf(fp, "%lu\n", centerIDs[i]);
			fprintf(fp, "%lf\n", centers->p[i].weight);
			for( int k = 0; k < centers->dim; k++ ) {
				fprintf(fp, "%lf ", centers->p[i].coord[k]);
			}
			fprintf(fp,"\n\n");
		}
	}
	fclose(fp);
}

void streamCluster( PStream* stream, 
		long kmin, long kmax, int dim,
		long chunksize, long centersize, char* outfile )
{
	long k1;
	long k2;
	long bsize = chunksize/nproc;

	float* block = (float*)malloc( chunksize*dim*sizeof(float) );
	float* centerBlock = (float*)malloc(centersize*dim*sizeof(float) );
	long* centerIDs = (long*)malloc(centersize*dim*sizeof(long));

	if( block == NULL ) { 
		fprintf(stderr,"not enough memory for a chunk!\n");
		exit(1);
	}

	Points points;
	points.dim = dim;
	points.num = chunksize;
	points.p = 
		(Point *)malloc(chunksize*sizeof(Point));
	Point* po = points.p;
	for(int p = 0; p < nproc; p++) {
		k1 = p*bsize; k2 = k1 + bsize; if (p == nproc-1) k2 = chunksize;
		// nanos_current_socket(p%NUMANODES);
#pragma oss task out(po[k1:k2]) firstprivate(k1,k2,block) label("POINT-COORD-INIT")
		{
			for( int i = k1; i < k2; i++ ) {
				po[i].coord = &block[i*dim];
			}
		}
	}

	Points centers;
	centers.dim = dim;
	centers.p = 
		(Point *)malloc(centersize*sizeof(Point));
	centers.num = 0;
	bsize = centersize/nproc;
	po = centers.p;
	for (int p = 0; p < nproc; p++)
	{
		k1 = p*bsize; k2 = k1 + bsize; if (p == nproc-1) k2 = centersize;
		// nanos_current_socket(p%NUMANODES);
#pragma oss task out(po[k1:k2]) firstprivate(k1,k2,centerBlock) label("CENTERS-COORD-WEIGHT-INIT")
		{
			for( int i = k1; i < k2; i++ ) {
				po[i].coord = &centerBlock[i*dim];
				po[i].weight = 1.0;
			}
		}
	}
	long IDoffset = 0;
	long kfinal;
	while(1) {

		size_t numRead  = stream->read(block, dim, chunksize ); 
		fprintf(stderr,"read %lu points\n",numRead);
		if( stream->ferror() || numRead < (unsigned int)chunksize && !stream->feof() ) {
			fprintf(stderr, "error reading data!\n");
			exit(1);
		}

		points.num = numRead;
		po = points.p;
		bsize = points.num/nproc;
		for (int p = 0; p < nproc; p++)
		{
			k1 = p*bsize; k2 = k1 + bsize; if (p == nproc-1) k2 = points.num;
			// nanos_current_socket(p%NUMANODES);
#pragma oss task out(po[k1:k2]) firstprivate(k1,k2) label("POINTS-WEIGHT-INIT")
			{
				for( int i = k1; i < k2; i++ ) {
					points.p[i].weight = 1.0;
				}
			}
		}

		switch_membership = (bool*)malloc(points.num*sizeof(bool));
		is_center = (bool*)calloc(points.num,sizeof(bool));
		center_table = (int*)malloc(points.num*sizeof(int));
		gl_lower = (double*)malloc(points.num*sizeof(double));

		//fprintf(stderr,"center_table = 0x%08x\n",(int)center_table);
		//fprintf(stderr,"is_center = 0x%08x\n",(int)is_center);

		localSearch(&points,kmin, kmax,&kfinal); // parallel
#pragma oss taskwait
		//fprintf(stderr,"finish local search\n");
		contcenters(&points); /* sequential */
		if( kfinal + centers.num > centersize ) {
			//here we don't handle the situation where # of centers gets too large. 
			fprintf(stderr,"oops! no more space for centers\n");
			exit(1);
		}

		copycenters(&points, &centers, centerIDs, IDoffset); /* sequential */
		IDoffset += numRead;

		free(is_center);
		free(switch_membership);
		free(center_table);
		free(gl_lower);

		if( stream->feof() ) {
			break;
		}
	}

	//finally cluster all temp centers
	switch_membership = (bool*)malloc(centers.num*sizeof(bool));
	is_center = (bool*)calloc(centers.num,sizeof(bool));
	center_table = (int*)malloc(centers.num*sizeof(int));
	gl_lower = (double*)malloc(centers.num*sizeof(double*));

	localSearch( &centers, kmin, kmax ,&kfinal ); // parallel
#pragma oss taskwait  
	contcenters(&centers);
	outcenterIDs( &centers, centerIDs, outfile);
}

int main(int argc, char **argv)
{
	char *outfilename = new char[MAXNAMESIZE];
	char *infilename = new char[MAXNAMESIZE];
	long kmin, kmax, n, chunksize, clustersize;
	int dim;

	double program_tstart;
	double roi_tstart;
	double program_elapsed;
	double roi_elapsed;
#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
	fprintf(stderr,"PARSEC Benchmark Suite Version "__PARSEC_XSTRING(PARSEC_VERSION)"\n");
	fflush(NULL);
#else
	fprintf(stderr,"PARSEC Benchmark Suite\n");
	fflush(NULL);
#endif //PARSEC_VERSION


	program_tstart = getusec_();
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin(__parsec_streamcluster);
#endif

	printf("argc:%d\n", argc);
	if (argc<11) {
		fprintf(stderr,"usage: %s k1 k2 d n chunksize clustersize infile outfile nproc ndivs\n",
				argv[0]);
		fprintf(stderr,"  k1:          Min. number of centers allowed\n");
		fprintf(stderr,"  k2:          Max. number of centers allowed\n");
		fprintf(stderr,"  d:           Dimension of each data point\n");
		fprintf(stderr,"  n:           Number of data points\n");
		fprintf(stderr,"  chunksize:   Number of data points to handle per step\n");
		fprintf(stderr,"  clustersize: Maximum number of intermediate centers\n");
		fprintf(stderr,"  infile:      Input file (if n<=0)\n");
		fprintf(stderr,"  outfile:     Output file\n");
		fprintf(stderr,"  nproc:       Number of threads to use (Ignored in OmpSs/OpenMP 4.0 versions, use NX_ARGS or OMP_NUM_THREADS respectively)\n");
		fprintf(stderr,"  ndivs:      Number of loop divisions, influencing the total number of tasks used in OmpSs/OpenMP 4.0\n");
		fprintf(stderr,"\n");
		fprintf(stderr, "if n > 0, points will be randomly generated instead of reading from infile.\n");
		exit(1);
	}



	kmin = atoi(argv[1]);
	kmax = atoi(argv[2]);
	dim = atoi(argv[3]);
	n = atoi(argv[4]);
	chunksize = atoi(argv[5]);
	clustersize = atoi(argv[6]);
	strcpy(infilename, argv[7]);
	strcpy(outfilename, argv[8]);
	if(argc < 10)
		nproc = DEFAULT_NDIVS;
	else
		nproc = atoi(argv[10]);

	/*Requires Nanox runtime*/
	// nanos_get_num_sockets(&NUMANODES);
	// fprintf(stdout,"NUMANODES: %d\n",NUMANODES);
	srand48(SEED);
	PStream* stream;
	if( n > 0 ) {
		stream = new SimStream(n);
	}
	else {
		stream = new FileStream(infilename);
	}


	roi_tstart = getusec_();
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_begin();
#endif

	streamCluster(stream, kmin, kmax, dim, chunksize, clustersize, outfilename );

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_roi_end();
#endif
	roi_elapsed = getusec_() - roi_tstart;

	delete stream;

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif

	program_elapsed = getusec_() - program_tstart;

	fprintf(stdout,"PROGRAM TIME:\t %f",program_elapsed/1e6);
	printf("\n");
	fprintf(stdout,"ROI TIME:\t %f",roi_elapsed/1e6);
	printf("\n");
	return 0;
}

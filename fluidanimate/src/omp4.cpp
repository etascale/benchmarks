//Code originally written by Richard O. Lee
//Modified by Christian Bienia and Christian Fensch
//This OpenMP 4.0 version written by Dimitrios Chasapis - Barcelona Supercomputing Center

#include <cstdlib>
#include <cstring>

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdint.h>
//#include <pthread.h>
#include <assert.h>

#include <omp.h>

#define NDIVS 8

//#include "parsec_barrier.hpp"

//#include "extrae_user_events.h"

//#ifdef ENABLE_PARSEC_HOOKS
//#include <hooks.h>
//#endif

static inline int isLittleEndian() {
  union {
    uint16_t word;
    uint8_t byte;
  } endian_test;

  endian_test.word = 0x00FF;
  return (endian_test.byte == 0xFF);
}

union __float_and_int {
  uint32_t i;
  float    f;
};

static inline float bswap_float(float x) {
  union __float_and_int __x;

   __x.f = x;
   __x.i = ((__x.i & 0xff000000) >> 24) | ((__x.i & 0x00ff0000) >>  8) |
           ((__x.i & 0x0000ff00) <<  8) | ((__x.i & 0x000000ff) << 24);

  return __x.f;
}

static inline int bswap_int32(int x) {
  return ( (((x) & 0xff000000) >> 24) | (((x) & 0x00ff0000) >>  8) |
           (((x) & 0x0000ff00) <<  8) | (((x) & 0x000000ff) << 24) );
}

////////////////////////////////////////////////////////////////////////////////

// note: icc-optimized version of this class gave 15% more
// performance than our hand-optimized SSE3 implementation
class Vec3
{
public:
    float x, y, z;

    Vec3() {}
    Vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

    float   GetLengthSq() const         { return x*x + y*y + z*z; }
    float   GetLength() const           { return sqrtf(GetLengthSq()); }
    Vec3 &  Normalize()                 { return *this /= GetLength(); }

    Vec3 &  operator += (Vec3 const &v) { x += v.x;  y += v.y; z += v.z; return *this; }
    //Vec3 &  operator += (Vec3 const &v) { if(v.x != 0.0f) x += v.x; if(v.y != 0.0f) y += v.y; if(v.z != 0.0f) z += v.z; return *this; }
    Vec3 &  operator -= (Vec3 const &v) { x -= v.x;  y -= v.y; z -= v.z; return *this; }
    Vec3 &  operator *= (float s)       { x *= s;  y *= s; z *= s; return *this; }
    Vec3 &  operator /= (float s)       { x /= s;  y /= s; z /= s; return *this; }

    Vec3    operator + (Vec3 const &v) const    { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3    operator - () const                 { return Vec3(-x, -y, -z); }
    Vec3    operator - (Vec3 const &v) const    { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3    operator * (float s) const          { return Vec3(x*s, y*s, z*s); }
    Vec3    operator / (float s) const          { return Vec3(x/s, y/s, z/s); }
	
    float   operator * (Vec3 const &v) const    { return x*v.x + y*v.y + z*v.z; }
};

////////////////////////////////////////////////////////////////////////////////

// there is a current limitation of 16 particles per cell
// (this structure use to be a simple linked-list of particles but, due to
// improved cache locality, we get a huge performance increase by copying
// particles instead of referencing them)
struct Cell
{
	Vec3 p[16];
	Vec3 hv[16];
	Vec3 v[16];
	Vec3 a[16];
	float density[16];
};
void init_cell(Cell &cell) {
	for (int i = 0; i < 16; ++i) {
		cell.p[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.hv[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.v[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.a[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.density[i] = 0.0f;
	}
}

/*struct Cell2
{
	Vec3 p[16];
	Vec3 hv[16];
	Vec3 v[16];
	Vec3 a[16];
	float density[16];
	bool tocat[16];
};
void init_cell2(Cell2 &cell) {
	for (int i = 0; i < 16; ++i) {
		cell.p[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.hv[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.v[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.a[i] = Vec3(0.0f, 0.0f, 0.0f);
		cell.density[i] = 0.0f;
		cell.tocat[i] = false;
	}
}*/

////////////////////////////////////////////////////////////////////////////////

const float timeStep = 0.005f;
const float doubleRestDensity = 2000.f;
const float kernelRadiusMultiplier = 1.695f;
const float stiffness = 1.5f;
const float viscosity = 0.4f;
const Vec3 externalAcceleration(0.f, -9.8f, 0.f);
const Vec3 domainMin(-0.065f, -0.08f, -0.065f);
const Vec3 domainMax(0.065f, 0.1f, 0.065f);

float restParticlesPerMeter, h, hSq;
float densityCoeff, pressureCoeff, viscosityCoeff;

int nx, ny, nz;			// number of grid cells in each dimension
Vec3 delta;				// cell dimensions
int origNumParticles = 0;
int numParticles = 0;
int numCells = 0;
Cell *cells = 0;
Cell *cells2 = 0;
int *cnumPars = 0;
int *cnumPars2 = 0;

int XDIVS = 1;	// number of partitions in X
int ZDIVS = 1;	// number of partitions in Z

#define NUM_GRIDS  ((XDIVS) * (ZDIVS))

struct Grid
{
	int sx, sy, sz;
	int ex, ey, ez;
} *grids;
#define XSIZE(i) (grids[i].ex - grids[i].sx)
#define YSIZE(i) (grids[i].ey - grids[i].sy)
#define ZSIZE(i) (grids[i].ez - grids[i].sz)
#define GRIDSIZE(i) (XSIZE(i)*YSIZE(i)*ZSIZE(i))
#define INGRID(i, x, y, z) (grids[i].sx <= x && x < grids[i].ex && grids[i].sy <= y && y < grids[i].ey && grids[i].sz <= z && z < grids[i].ez)
int gridtoind(int x, int y, int z) {
	int i = 0, ind = 0;
	while (!INGRID(i, x, y, z)) {
		ind += GRIDSIZE(i);
		i++;
	}
	x -= grids[i].sx;
	y -= grids[i].sy;
	z -= grids[i].sz;
	ind += (z*YSIZE(i) + y)*XSIZE(i) + x;
	return ind;
}
void indtogrid(int ind, int &x, int &y, int &z) {
	int i = 0;
	while (ind >= GRIDSIZE(i)) {
		ind -= GRIDSIZE(i);
		i++;
	}
	x = ind%XSIZE(i) + grids[i].sx;
	ind /= XSIZE(i);
	y = ind%YSIZE(i) + grids[i].sy;
	ind /= YSIZE(i);
	z = ind + grids[i].sz;
}

bool *border;			// flags which cells lie on grid boundaries
/*struct Cell_list {
	int index;
	Cell2 cell;
	int cnumPars;
	Cell_list *next;
} **private_cells;

Cell_list *get_cell(int i, int index) {
	Cell_list *list = private_cells[i], *ant = NULL;
	while (list and list->index != index) {
		ant = list;
		list = list->next;
	}
	if (!list) {
		list = new Cell_list;
		list->index = index;
		init_cell2(list->cell);
		list->cnumPars = 0;
		list->next = NULL;
		if (ant) ant->next = list;
		else private_cells[i] = list;
	}
	return list;
}*/

bool is_zero(float f, float eps = 0.0000000000001) {
	return fabs(f) < eps;
}

bool is_zero(Vec3 v, float eps = 0.0000000000001) {
	return is_zero(v.x, eps) && is_zero(v.y, eps) && is_zero(v.z, eps);
}

//pthread_attr_t attr;
//pthread_t *thread;
//pthread_mutex_t **mutex;	// used to lock cells in RebuildGrid and also particles in other functions
omp_lock_t **lock;
//pthread_barrier_t barrier;	// global barrier used by all threads
/*typedef struct __thread_args {
  int tid;			//thread id, determines work partition
  int frames;			//number of frames to compute
} thread_args;*/			//arguments for threads

////////////////////////////////////////////////////////////////////////////////

/*
 * hmgweight
 *
 * Computes the hamming weight of x
 *
 * x      - input value
 * lsb    - if x!=0 position of smallest bit set, else -1
 *
 * return - the hamming weight
 */
unsigned int hmgweight(unsigned int x, int *lsb) {
  unsigned int weight=0;
  unsigned int mask= 1;
  unsigned int count=0;
 
  *lsb=-1;
  while(x > 0) {
    unsigned int temp;
    temp=(x&mask);
    if((x&mask) == 1) {
      weight++;
      if(*lsb == -1) *lsb = count;
    }
    x >>= 1;
    count++;
  }

  return weight;
}

void InitSim(char const *fileName, unsigned int threadnum)
{
	//Compute partitioning based on square root of number of threads
	//NOTE: Other partition sizes are possible as long as XDIVS * ZDIVS == threadnum,
	//      but communication is minimal (and hence optimal) if XDIVS == ZDIVS
	int lsb;
	if(hmgweight(threadnum,&lsb) != 1) {
		std::cerr << "Number of tasks must be a power of 2" << std::endl;
		exit(1);
	}
	XDIVS = 1<<(lsb/2);
	ZDIVS = 1<<(lsb/2);
	if(XDIVS*ZDIVS != threadnum) XDIVS*=2;
	assert(XDIVS * ZDIVS == threadnum);

	//thread = new pthread_t[NUM_GRIDS];
	grids = new struct Grid[NUM_GRIDS];

	//Load input particles
	std::cout << "Loading file \"" << fileName << "\"..." << std::endl;
	std::ifstream file(fileName, std::ios::binary);
	assert(file);

	file.read((char *)&restParticlesPerMeter, 4);
	file.read((char *)&origNumParticles, 4);
        if(!isLittleEndian()) {
          restParticlesPerMeter = bswap_float(restParticlesPerMeter);
          origNumParticles      = bswap_int32(origNumParticles);
        }
	numParticles = origNumParticles;

	h = kernelRadiusMultiplier / restParticlesPerMeter;
	hSq = h*h;
	const float pi = 3.14159265358979f;
	float coeff1 = 315.f / (64.f*pi*pow(h,9.f));
	float coeff2 = 15.f / (pi*pow(h,6.f));
	float coeff3 = 45.f / (pi*pow(h,6.f));
	float particleMass = 0.5f*doubleRestDensity / (restParticlesPerMeter*restParticlesPerMeter*restParticlesPerMeter);
	densityCoeff = particleMass * coeff1;
	pressureCoeff = 3.f*coeff2 * 0.5f*stiffness * particleMass;
	viscosityCoeff = viscosity * coeff3 * particleMass;

	Vec3 range = domainMax - domainMin;
	nx = (int)(range.x / h);
	ny = (int)(range.y / h);
	nz = (int)(range.z / h);
	assert(nx >= 1 && ny >= 1 && nz >= 1);
	numCells = nx*ny*nz;
	std::cout << "Number of cells: " << numCells << std::endl;
	delta.x = range.x / nx;
	delta.y = range.y / ny;
	delta.z = range.z / nz;
	assert(delta.x >= h && delta.y >= h && delta.z >= h);

	assert(nx >= XDIVS && nz >= ZDIVS);
	int gi = 0;
	int sx, sz, ex, ez;
	ex = 0;
	for(int i = 0; i < XDIVS; ++i)
	{
		sx = ex;
		ex = int(float(nx)/float(XDIVS) * (i+1) + 0.5f);
		assert(sx < ex);

		ez = 0;
		for(int j = 0; j < ZDIVS; ++j, ++gi)
		{
			sz = ez;
			ez = int(float(nz)/float(ZDIVS) * (j+1) + 0.5f);
			assert(sz < ez);

			grids[gi].sx = sx;
			grids[gi].ex = ex;
			grids[gi].sy = 0;
			grids[gi].ey = ny;
			grids[gi].sz = sz;
			grids[gi].ez = ez;
		}
	}
	assert(gi == NUM_GRIDS);

	/*for (int i = 0; i < NUM_GRIDS; ++i) {
		int x = grids[i].ex - grids[i].sx;
		int y = grids[i].ey - grids[i].sy;
		int z = grids[i].ez - grids[i].sz;	
		std::cerr << x << " " << y << " " << z << " " << x*y*z << std::endl;
	}
	exit(0);*/

	border = new bool[numCells];
	for(int i = 0; i < NUM_GRIDS; ++i)
		for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
			for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
				for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
				{
					//int index = (iz*ny + iy)*nx + ix;
					int index = gridtoind(ix, iy, iz);
					border[index] = false;
					for(int dk = -1; dk <= 1; ++dk)
						for(int dj = -1; dj <= 1; ++dj)
							for(int di = -1; di <= 1; ++di)
							{
								int ci = ix + di;
								int cj = iy + dj;
								int ck = iz + dk;

								if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
								if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
								if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

								if( ci < grids[i].sx || ci >= grids[i].ex ||
									cj < grids[i].sy || cj >= grids[i].ey ||
									ck < grids[i].sz || ck >= grids[i].ez )
									border[index] = true;
							}
				}

	//pthread_attr_init(&attr);
	//pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	lock = new omp_lock_t *[numCells];
	for (int i = 0; i < numCells; ++i) {
		if (border[i]) {
			lock[i] = new omp_lock_t[16];
			for (int j = 0; j < 16; ++j)
				omp_init_lock(&lock[i][j]);
		}
	}
/*	mutex = new pthread_mutex_t *[numCells];
	for(int i = 0; i < numCells; ++i)
	{
		int n = (border[i] ? 16 : 1);
		mutex[i] = new pthread_mutex_t[n];
		for(int j = 0; j < n; ++j)
			pthread_mutex_init(&mutex[i][j], NULL);
	}
//	pthread_barrier_init(&barrier, NULL, NUM_GRIDS);
*/
	cells = new Cell[numCells];
	cells2 = new Cell[numCells];
	cnumPars = new int[numCells];
	cnumPars2 = new int[numCells];
	//private_cells = new Cell_list*[threadnum];
	assert(cells && cells2 && cnumPars && cnumPars2);

	/*for (int i = 0; i < threadnum; ++i) {
		private_cells[i] = NULL;
	}*/

	/*for (int  i = 0; i < numCells; ++i) {
		init_cell(cells[i]);
		cnumPars[i] = 0;
	}*/

	memset(cnumPars2, 0, numCells*sizeof(int));

	float px, py, pz, hvx, hvy, hvz, vx, vy, vz;
	for(int i = 0; i < origNumParticles; ++i)
	{
		file.read((char *)&px, 4);
		file.read((char *)&py, 4);
		file.read((char *)&pz, 4);
		file.read((char *)&hvx, 4);
		file.read((char *)&hvy, 4);
		file.read((char *)&hvz, 4);
		file.read((char *)&vx, 4);
		file.read((char *)&vy, 4);
		file.read((char *)&vz, 4);
                if(!isLittleEndian()) {
                  px  = bswap_float(px);
                  py  = bswap_float(py);
                  pz  = bswap_float(pz);
                  hvx = bswap_float(hvx);
                  hvy = bswap_float(hvy);
                  hvz = bswap_float(hvz);
                  vx  = bswap_float(vx);
                  vy  = bswap_float(vy);
                  vz  = bswap_float(vz);
                }

		int ci = (int)((px - domainMin.x) / delta.x);
		int cj = (int)((py - domainMin.y) / delta.y);
		int ck = (int)((pz - domainMin.z) / delta.z);

		if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
		if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
		if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

		//int index = (ck*ny + cj)*nx + ci;
		int index = gridtoind(ci, cj, ck);
		Cell &cell = cells2[index];

		int np = cnumPars2[index];
		if(np < 16)
		{
			cell.p[np].x = px;
			cell.p[np].y = py;
			cell.p[np].z = pz;
			cell.hv[np].x = hvx;
			cell.hv[np].y = hvy;
			cell.hv[np].z = hvz;
			cell.v[np].x = vx;
			cell.v[np].y = vy;
			cell.v[np].z = vz;
			++cnumPars2[index];
		}
		else
			--numParticles;
	}
	std::cout << "Number of particles: " << numParticles << " (" << origNumParticles-numParticles << " skipped)" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////

void SaveFile2(char const *fileName)
{
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

	std::ofstream file(fileName, std::ios::binary);
	assert(file);

	  file << restParticlesPerMeter << std::endl;
	  file << origNumParticles << std::endl;

	int count = 0;
	for(int i = 0; i < numCells; ++i)
	{
		Cell const &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			  file << cell.p[j].x << std::endl;
			  file << cell.p[j].y << std::endl;
			  file << cell.p[j].z << std::endl;
			  file << cell.hv[j].x << std::endl;
			  file << cell.hv[j].y << std::endl;
			  file << cell.hv[j].z << std::endl;
			  file << cell.v[j].x << std::endl;
			  file << cell.v[j].y << std::endl;
			  file << cell.v[j].z << std::endl;
		}
	}

	int numSkipped = origNumParticles - numParticles;
	float zero = 0.f;

	for(int i = 0; i < numSkipped; ++i)
	{
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
		file << zero << std::endl;
	}
}

void SaveFileAll(char const *fileName)
{
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

	std::ofstream file(fileName, std::ios::binary);
	assert(file);

	file.write((char *)&restParticlesPerMeter, 4);
	file.write((char *)&origNumParticles,      4);

	int count = 0;
	for(int i = 0; i < numCells; ++i)
	{
		Cell const &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
			file.write((char *)&cell.p[j].x,  4);
			file.write((char *)&cell.p[j].y,  4);
			file.write((char *)&cell.p[j].z,  4);
			file.write((char *)&cell.hv[j].x, 4);
			file.write((char *)&cell.hv[j].y, 4);
			file.write((char *)&cell.hv[j].z, 4);
			file.write((char *)&cell.v[j].x,  4);
			file.write((char *)&cell.v[j].y,  4);
			file.write((char *)&cell.v[j].z,  4);
			file.write((char *)&cell.a[j].x,  4);
			file.write((char *)&cell.a[j].y,  4);
			file.write((char *)&cell.a[j].z,  4);
			file.write((char *)&cell.density[j],  4);
		}
	}

	int numSkipped = origNumParticles - numParticles;
	float zero = 0.f;

	for(int i = 0; i < numSkipped; ++i)
	{
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
	}
}

void SaveFile(char const *fileName)
{
	std::cout << "Saving file \"" << fileName << "\"..." << std::endl;

	std::ofstream file(fileName, std::ios::binary);
	assert(file);

        if(!isLittleEndian()) {
          float restParticlesPerMeter_le;
          int   origNumParticles_le;

          restParticlesPerMeter_le = bswap_float(restParticlesPerMeter);
          origNumParticles_le      = bswap_int32(origNumParticles);
	  file.write((char *)&restParticlesPerMeter_le, 4);
	  file.write((char *)&origNumParticles_le,      4);
        } else {
	  file.write((char *)&restParticlesPerMeter, 4);
	  file.write((char *)&origNumParticles,      4);
        }

	int count = 0;
	/*for(int i = 0; i < numCells; ++i)
	{
		int x, y, z;
		//indtogrid(i, x, y, z);
		//i = (z*ny + y)*nx + x;
		x = i%nx;
		i /= nx;
		y = i%ny;
		i /= ny;
		z = i;*/
	int i;
	for (int z = 0; z < nz; ++z)
	 for (int y = 0; y < ny; ++y)
	  for (int x = 0; x < nx; ++x)
	   {
		i = gridtoind(x, y, z);
		Cell const &cell = cells[i];
		int np = cnumPars[i];
		for(int j = 0; j < np; ++j)
		{
                        if(!isLittleEndian()) {
                          float px, py, pz, hvx, hvy, hvz, vx,vy, vz;

                          px  = bswap_float(cell.p[j].x);
                          py  = bswap_float(cell.p[j].y);
                          pz  = bswap_float(cell.p[j].z);
                          hvx = bswap_float(cell.hv[j].x);
                          hvy = bswap_float(cell.hv[j].y);
                          hvz = bswap_float(cell.hv[j].z);
                          vx  = bswap_float(cell.v[j].x);
                          vy  = bswap_float(cell.v[j].y);
                          vz  = bswap_float(cell.v[j].z);

			  file.write((char *)&px,  4);
			  file.write((char *)&py,  4);
			  file.write((char *)&pz,  4);
			  file.write((char *)&hvx, 4);
			  file.write((char *)&hvy, 4);
			  file.write((char *)&hvz, 4);
			  file.write((char *)&vx,  4);
			  file.write((char *)&vy,  4);
			  file.write((char *)&vz,  4);
                        } else {
			  file.write((char *)&cell.p[j].x,  4);
			  file.write((char *)&cell.p[j].y,  4);
			  file.write((char *)&cell.p[j].z,  4);
			  file.write((char *)&cell.hv[j].x, 4);
			  file.write((char *)&cell.hv[j].y, 4);
			  file.write((char *)&cell.hv[j].z, 4);
			  file.write((char *)&cell.v[j].x,  4);
			  file.write((char *)&cell.v[j].y,  4);
			  file.write((char *)&cell.v[j].z,  4);
                        }
			++count;
		}
	}
	assert(count == numParticles);

	int numSkipped = origNumParticles - numParticles;
	float zero = 0.f;
        if(!isLittleEndian()) {
          zero = bswap_float(zero);
        }

	for(int i = 0; i < numSkipped; ++i)
	{
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
		file.write((char *)&zero, 4);
	}
}

////////////////////////////////////////////////////////////////////////////////

void CleanUpSim()
{
//	pthread_attr_destroy(&attr);

/*	for(int i = 0; i < numCells; ++i)
	{
		int n = (border[i] ? 16 : 1);
		for(int j = 0; j < n; ++j)
//			pthread_mutex_destroy(&mutex[i][j]);
//		delete[] mutex[i];
	}
//	pthread_barrier_destroy(&barrier);
//	delete[] mutex;
*/
	for (int i = 0; i < numCells; ++i)
	{
		if (border[i]) {
			for (int j = 0; j < 16; ++j)
				omp_destroy_lock(&lock[i][j]);
			delete[] lock[i];
		}
	}
	delete[] lock;
	delete[] border;

	delete[] cells;
	delete[] cells2;
	delete[] cnumPars;
	delete[] cnumPars2;
//	delete[] thread;
	delete[] grids;
}

////////////////////////////////////////////////////////////////////////////////

void ClearParticlesMT(int i)
{
	//this iteration gives continues area of the array
	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
		                //int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				//printf("Clearing Particles:%d (x=%d,y=%d,z=%d)=%d\n", i, ix, iy, iz, index);
				cnumPars[index] = 0;
			}
}

////////////////////////////////////////////////////////////////////////////////

void RebuildGridMT(int i)
{
	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
        		        //int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				Cell const &cell2 = cells2[index];
				int np2 = cnumPars2[index];
				for(int j = 0; j < np2; ++j)
				{
					int ci = (int)((cell2.p[j].x - domainMin.x) / delta.x);
					int cj = (int)((cell2.p[j].y - domainMin.y) / delta.y);
					int ck = (int)((cell2.p[j].z - domainMin.z) / delta.z);

					if(ci < 0) ci = 0; else if(ci > (nx-1)) ci = nx-1;
					if(cj < 0) cj = 0; else if(cj > (ny-1)) cj = ny-1;
					if(ck < 0) ck = 0; else if(ck > (nz-1)) ck = nz-1;

					//int index2 = (ck*ny + cj)*nx + ci;
					int index2 = gridtoind(ci, cj, ck);
					// this assumes that particles cannot travel more than one grid cell per time step
					int np = 0;
					Cell &cell = cells[index2];
					if(border[index2])
					{
						//printf("Clearing Particles:%d (x=%d,y=%d,z=%d)=%d\n", i, ci, cj, ck, index2);
						//pthread_mutex_lock(&mutex[index2][0]);
						//#pragma omp critical
						//omp_set_lock(&lock[index2][0]);
						#pragma omp atomic
						cnumPars[index2]++;
						#pragma omp atomic						
						np += cnumPars[index2];
						//omp_unset_lock(&lock[index2][0]);
						//pthread_mutex_unlock(&mutex[index2][0]);
						/*Cell_list *clist = get_cell(i, index2);
						np = (clist->cnumPars)++;
						cell = &(clist->cell);*/
					}
					else {
						np = cnumPars[index2]++;
					}

					cell.p[np].x = cell2.p[j].x;
					cell.p[np].y = cell2.p[j].y;
					cell.p[np].z = cell2.p[j].z;
					cell.hv[np].x = cell2.hv[j].x;
					cell.hv[np].y = cell2.hv[j].y;
					cell.hv[np].z = cell2.hv[j].z;
					cell.v[np].x = cell2.v[j].x;
					cell.v[np].y = cell2.v[j].y;
					cell.v[np].z = cell2.v[j].z;
					/*if(border[index2])
					{
						Cell_list *clist = get_cell(i, index2);
						np = (clist->cnumPars)++;
						Cell2 &cell = clist->cell;
						cell.p[np].x = cell2.p[j].x;
						cell.p[np].y = cell2.p[j].y;
						cell.p[np].z = cell2.p[j].z;
						cell.hv[np].x = cell2.hv[j].x;
						cell.hv[np].y = cell2.hv[j].y;
						cell.hv[np].z = cell2.hv[j].z;
						cell.v[np].x = cell2.v[j].x;
						cell.v[np].y = cell2.v[j].y;
						cell.v[np].z = cell2.v[j].z;
					}
					else {
						np = cnumPars[index2]++;
						Cell &cell = cells[index2];
						cell.p[np].x = cell2.p[j].x;
						cell.p[np].y = cell2.p[j].y;
						cell.p[np].z = cell2.p[j].z;
						cell.hv[np].x = cell2.hv[j].x;
						cell.hv[np].y = cell2.hv[j].y;
						cell.hv[np].z = cell2.hv[j].z;
						cell.v[np].x = cell2.v[j].x;
						cell.v[np].y = cell2.v[j].y;
						cell.v[np].z = cell2.v[j].z;
					}*/
				}
			}
}

/*void RebuildGridJoin(int numtasks) {
	for (int i = 0; i < numtasks; ++i) {
		Cell_list *clist =  private_cells[i];
		while (clist) {
			for (int j = 0; j < clist->cnumPars; ++j) {
				int np = cnumPars[clist->index] + j;
				Cell &cell = cells[clist->index];
				Cell2 &cell2 = clist->cell;
				cell.p[np].x = cell2.p[j].x;
				cell.p[np].y = cell2.p[j].y;
				cell.p[np].z = cell2.p[j].z;
				cell.hv[np].x = cell2.hv[j].x;
				cell.hv[np].y = cell2.hv[j].y;
				cell.hv[np].z = cell2.hv[j].z;
				cell.v[np].x = cell2.v[j].x;
				cell.v[np].y = cell2.v[j].y;
				cell.v[np].z = cell2.v[j].z;
				
			}
			cnumPars[clist->index] += clist->cnumPars;
			Cell_list *old = clist;
			clist = clist->next;
			delete old;
		}
		private_cells[i] = NULL;
	}
}*/

////////////////////////////////////////////////////////////////////////////////

int InitNeighCellList(int ci, int cj, int ck, int *neighCells)
{
	int numNeighCells = 0;

	for(int di = -1; di <= 1; ++di)
		for(int dj = -1; dj <= 1; ++dj)
			for(int dk = -1; dk <= 1; ++dk)
			{
				int ii = ci + di;
				int jj = cj + dj;
				int kk = ck + dk;
				if(ii >= 0 && ii < nx && jj >= 0 && jj < ny && kk >= 0 && kk < nz)
				{
					//int index = (kk*ny + jj)*nx + ii;
					int index = gridtoind(ii, jj, kk);
					if(cnumPars[index] != 0)
					{
						neighCells[numNeighCells] = index;
						++numNeighCells;
					}
				}
			}

	return numNeighCells;
}

////////////////////////////////////////////////////////////////////////////////

void InitDensitiesAndForcesMT(int i)
{
	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
		                //int index = (iz*ny + iy)*nx + ix;
					//std::cerr << ix << " " << iy << " " << iz << " ";
				int index = gridtoind(ix, iy, iz);
					//std::cerr << "ok" << std::endl;
				Cell &cell = cells[index];
				int np = cnumPars[index];
				for(int j = 0; j < np; ++j)
				{
					cell.density[j] = 0.f;
					cell.a[j] = externalAcceleration;
				}
			}
}

////////////////////////////////////////////////////////////////////////////////

void ComputeDensitiesMT(int i)
{
	int neighCells[27];

	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
					//std::cerr << ix << " " << iy << " " << iz << " ";
				int index = gridtoind(ix, iy, iz);
					//std::cerr << "ok" << std::endl;
				int np = cnumPars[index];
				if(np == 0)
					continue;

				int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

				Cell &cell = cells[index];

				for(int j = 0; j < np; ++j)
					for(int inc = 0; inc < numNeighCells; ++inc)
					{
						int indexNeigh = neighCells[inc];
						Cell &neigh = cells[indexNeigh];
						int numNeighPars = cnumPars[indexNeigh];
						for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j])
							{
								float distSq = (cell.p[j] - neigh.p[iparNeigh]).GetLengthSq();
								if(distSq < hSq)
								{
									float t = hSq - distSq;
									float tc = t*t*t;

									if(border[index])
									{
										//pthread_mutex_lock(&mutex[index][j]);
										//#pragma omp critical
										//omp_set_lock(&lock[index][j]);
										#pragma omp atomic
										cell.density[j] += tc;
										//omp_unset_lock(&lock[index][j]);
										//pthread_mutex_unlock(&mutex[index][j]);
										/*Cell_list *clist = get_cell(i, index);
										clist->cell.density[j] += tc;
										clist->cell.tocat[j] = true;*/
									}
									else
										cell.density[j] += tc;

									if(border[indexNeigh])
									{
										//pthread_mutex_lock(&mutex[indexNeigh][iparNeigh]);
										//#pragma omp critical
										//omp_set_lock(&lock[indexNeigh][iparNeigh]);
										#pragma omp atomic										
										neigh.density[iparNeigh] += tc;
										//omp_unset_lock(&lock[indexNeigh][iparNeigh]);
										//pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh]);
										/*Cell_list *clist = get_cell(i, indexNeigh);
										clist->cell.density[iparNeigh] += tc;
										clist->cell.tocat[iparNeigh] = true;*/
									}
									else
										neigh.density[iparNeigh] += tc;
								}
							}
					}
			}
}

/*void ComputeDensitiesJoin(int numtasks) {
	for (int i = 0; i < numtasks; ++i) {
		Cell_list *clist = private_cells[i];
		while(clist) {
			for (int j = 0; j < cnumPars[clist->index]; ++j) {
				//if(!is_zero(clist->cell.density[j])) cells[clist->index].density[j] += clist->cell.density[j];
				//if(clist->cell.density[j] != 0.0f) cells[clist->index].density[j] += clist->cell.density[j];
				if(clist->cell.tocat[j]) cells[clist->index].density[j] += clist->cell.density[j];
			}
			Cell_list *old = clist;
			clist = clist->next;
			delete old;
		}
		private_cells[i] = NULL;
	}
}*/

////////////////////////////////////////////////////////////////////////////////

void ComputeDensities2MT(int i)
{
	const float tc = hSq*hSq*hSq;
	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				Cell &cell = cells[index];
				int np = cnumPars[index];
				for(int j = 0; j < np; ++j)
				{
					cell.density[j] += tc;
					cell.density[j] *= densityCoeff;
				}
			}
}

////////////////////////////////////////////////////////////////////////////////

void ComputeForcesMT(int i)
{
	int neighCells[27];

	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				int np = cnumPars[index];
				if(np == 0)
					continue;

				int numNeighCells = InitNeighCellList(ix, iy, iz, neighCells);

				Cell &cell = cells[index];

				for(int j = 0; j < np; ++j)
					for(int inc = 0; inc < numNeighCells; ++inc)
					{
						int indexNeigh = neighCells[inc];
						Cell &neigh = cells[indexNeigh];
						int numNeighPars = cnumPars[indexNeigh];
						for(int iparNeigh = 0; iparNeigh < numNeighPars; ++iparNeigh)
							if(&neigh.p[iparNeigh] < &cell.p[j])
							{
								Vec3 disp = cell.p[j] - neigh.p[iparNeigh];
								float distSq = disp.GetLengthSq();
								if(distSq < hSq)
								{
									float dist = sqrtf(std::max(distSq, 1e-12f));
									float hmr = h - dist;

									Vec3 acc = disp * pressureCoeff * (hmr*hmr/dist) * (cell.density[j]+neigh.density[iparNeigh] - doubleRestDensity);
									acc += (neigh.v[iparNeigh] - cell.v[j]) * viscosityCoeff * hmr;
									acc /= cell.density[j] * neigh.density[iparNeigh];

									if(border[index])
									{
										//pthread_mutex_lock(&mutex[index][j]);
										//#pragma omp critical
										omp_set_lock(&lock[index][j]);
										//#pragma omp atomic										
										cell.a[j] += acc;
										omp_unset_lock(&lock[index][j]);
										//pthread_mutex_unlock(&mutex[index][j]);
										/*Cell_list *clist = get_cell(i, index);
										clist->cell.a[j] += acc;
										clist->cell.tocat[j] = true;*/
									}
									else
										cell.a[j] += acc;

									if(border[indexNeigh])
									{
										//pthread_mutex_lock(&mutex[indexNeigh][iparNeigh]);
										//#pragma omp critical
										omp_set_lock(&lock[indexNeigh][iparNeigh]);
										//#pragma omp atomic
										neigh.a[iparNeigh] -= acc;
										omp_unset_lock(&lock[indexNeigh][iparNeigh]);
										//pthread_mutex_unlock(&mutex[indexNeigh][iparNeigh]);
										/*Cell_list *clist = get_cell(i, indexNeigh);
										clist->cell.a[iparNeigh] -= acc;
										clist->cell.tocat[iparNeigh] = true;*/
									}
									else
										neigh.a[iparNeigh] -= acc;
								}
							}
					}
			}
}

/*void ComputeForcesJoin(int numtasks) {
	for (int i = 0; i < numtasks; ++i) {
		Cell_list *clist = private_cells[i];
		while(clist) {
			for (int j = 0; j < cnumPars[clist->index]; ++j) {
				//if(!is_zero(clist->cell.a[j])) cells[clist->index].a[j] += clist->cell.a[j];
				//if(clist->cell.a[j].x != 0.0f || clist->cell.a[j].y != 0.0f || clist->cell.a[j].z != 0.0f) cells[clist->index].a[j] += clist->cell.a[j];
				if(clist->cell.tocat[j]) cells[clist->index].a[j] += clist->cell.a[j];
			}
			Cell_list *old = clist;
			clist = clist->next;
			delete old;
		}
		private_cells[i] = NULL;
	}
}*/

////////////////////////////////////////////////////////////////////////////////

void ProcessCollisionsMT(int i)
{
	const float parSize = 0.0002f;
	const float epsilon = 1e-10f;
	const float stiffness = 30000.f;
	const float damping = 128.f;

	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				Cell &cell = cells[index];
				int np = cnumPars[index];
				for(int j = 0; j < np; ++j)
				{
					Vec3 pos = cell.p[j] + cell.hv[j] * timeStep;

					float diff = parSize - (pos.x - domainMin.x);
					if(diff > epsilon)
						cell.a[j].x += stiffness*diff - damping*cell.v[j].x;

					diff = parSize - (domainMax.x - pos.x);
					if(diff > epsilon)
						cell.a[j].x -= stiffness*diff + damping*cell.v[j].x;

					diff = parSize - (pos.y - domainMin.y);
					if(diff > epsilon)
						cell.a[j].y += stiffness*diff - damping*cell.v[j].y;

					diff = parSize - (domainMax.y - pos.y);
					if(diff > epsilon)
						cell.a[j].y -= stiffness*diff + damping*cell.v[j].y;

					diff = parSize - (pos.z - domainMin.z);
					if(diff > epsilon)
						cell.a[j].z += stiffness*diff - damping*cell.v[j].z;

					diff = parSize - (domainMax.z - pos.z);
					if(diff > epsilon)
						cell.a[j].z -= stiffness*diff + damping*cell.v[j].z;
				}
			}
}

////////////////////////////////////////////////////////////////////////////////

void AdvanceParticlesMT(int i)
{
	for(int iz = grids[i].sz; iz < grids[i].ez; ++iz)
		for(int iy = grids[i].sy; iy < grids[i].ey; ++iy)
			for(int ix = grids[i].sx; ix < grids[i].ex; ++ix)
			{
                		//int index = (iz*ny + iy)*nx + ix;
				int index = gridtoind(ix, iy, iz);
				Cell &cell = cells[index];
				int np = cnumPars[index];
				for(int j = 0; j < np; ++j)
				{
					Vec3 v_half = cell.hv[j] + cell.a[j]*timeStep;
					cell.p[j] += v_half * timeStep;
					cell.v[j] = cell.hv[j] + v_half;
					cell.v[j] *= 0.5f;
					cell.hv[j] = v_half;
				}
			}
}

////////////////////////////////////////////////////////////////////////////////

/*inline int li(int i) {
	return (grids[i].sz*ny + grids[i].sy)*nx + grids[i].sx;
}

inline int ls(int i) {
	return (grids[i].ez*ny + grids[i].ey)*nx + grids[i].ex;
}

inline int li1(int i) {
	return ((grids[i].sz-1)*ny + grids[i].sy-1)*nx + grids[i].sx-1;
}

inline int ls1(int i) {
	return ((grids[i].ez+1)*ny + grids[i].ey+1)*nx + grids[i].ex+1;
}*/

void AdvanceFrameMT(int ntasks)
{
	#pragma omp parallel
	{
		#pragma omp single
		{
				//std::cerr << "Xivato 0" << std::endl;
			int blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(out: cnumPars[blocini:size]) //label(ClearParticles)
				ClearParticlesMT(i);
				blocini += size;
			}
			//#pragma omp taskwait on([numCells]cnumPars)
				//SaveFileAll("bolcat1.fluid");
				//std::cerr << "Xivato 1" << std::endl;
			#pragma omp taskwait
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(in: cells2[blocini:size], cnumPars2[blocini:size]) depend(inout: cnumPars[blocini:size]) depend(out: cells[blocini:size]) //label(RebuildGrid)
				RebuildGridMT(i);
				blocini += size;
			}
			#pragma omp taskwait
			//#pragma omp task in([ntasks]private_cells) inout([numCells]cnumPars, [numCells]cells)
			//RebuildGridJoin(ntasks);
				//#pragma omp taskwait
				//SaveFileAll("bolcat2.fluid");
				//std::cerr << "Xivato 2" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(in: cnumPars[blocini:size]) depend(inout: cells[blocini:size]) //label(InitDensitiesAndForces)
				InitDensitiesAndForcesMT(i);
				blocini += size;
			}
			#pragma omp taskwait //on([numCells]cells)
				//SaveFileAll("bolcat3.fluid");
				//std::cerr << "Xivato 3" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(in: cnumPars[blocini:size]) depend(inout: cells[blocini:size]) //label(ComputeDensities)
				ComputeDensitiesMT(i);
				blocini += size;
			}
			#pragma omp taskwait
			//#pragma omp task in([ntasks]private_cells, [numCells]cnumPars) inout([numCells]cells)
			//ComputeDensitiesJoin(ntasks);
				//#pragma omp taskwait
			//#pragma omp taskwait on([numCells]cells)//
				//SaveFileAll("bolcat4.fluid");
				//std::cerr << "Xivato 4" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(inout: cells[blocini:size]) //label(ComputeDensities2)
				ComputeDensities2MT(i);
				blocini += size;
			}
			#pragma omp taskwait //on([numCells]cells)
				//SaveFileAll("bolcat5.fluid");
				//std::cerr << "Xivato 5" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(in: cnumPars[blocini:size]) depend(inout: cells[blocini:size]) //label(ComputeForces)
				ComputeForcesMT(i);
				blocini += size;
			}
			#pragma omp taskwait
			//#pragma omp task in([ntasks]private_cells, [numCells]cnumPars) inout([numCells]cells)
			//ComputeForcesJoin(ntasks);
				//#pragma omp taskwait
			//#pragma omp taskwait on([numCells]cells)//
				//SaveFileAll("bolcat6.fluid");
				//std::cerr << "Xivato 6" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(in: cnumPars[blocini:size]) depend(inout: cells[blocini:size]) //label(ProcessCollisions)
				ProcessCollisionsMT(i);
				blocini += size;
			}
			#pragma omp taskwait //on([numCells]cells)
				//SaveFileAll("bolcat7.fluid");
				//std::cerr << "Xivato 7" << std::endl;
			blocini = 0;
			for (int i = 0; i < ntasks; ++i) {
				int size = GRIDSIZE(i);
				#pragma omp task //depend(inout: cells[blocini:size]) //label(AdvanceParticles)
				AdvanceParticlesMT(i);
				blocini += size;
			}
			#pragma omp taskwait
		//SaveFileAll("bolcat8.fluid");
		//std::cerr << "Xivato 8" << std::endl;
		} //end of single
	}	//end of parallel region
}

/*void *AdvanceFramesMT(void *args)
{
	thread_args *targs = (thread_args *)args;

	for(int i = 0; i < targs->frames; ++i)
		AdvanceFrameMT(targs->tid);

	return NULL;
}*/

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{

  if(argc < 5 || argc >= 7)
  {
	std::cout << "Usage: " << argv[0] << " <tasknum> <framenum> <.fluid input file> <.fluid output file> [ndivs]" << std::endl;
	std::cout << "Warning: Argument tasknum is ignored! Use OMP_NUM_THREADS for setting thread number and ndivs argument to influence the number of tasks (usually ndivs works well if it's equal to the number of cores)." << std::endl; 
    return -1;
  }

  int threadnum = NDIVS;
  if(argc > 5) {
	threadnum = atoi(argv[5]);	
  }


	int framenum = atoi(argv[2]);

	//Check arguments
	if(threadnum < 1) {
		std::cerr << "<ndivs> must at least be 1" << std::endl;
		return -1;
	}
	if(framenum < 1) {
		std::cerr << "<framenum> must at least be 1" << std::endl;
		return -1;
	}

	InitSim(argv[3], threadnum);

	//thread_args targs[threadnum];
//ROI begins
	int startt = time(NULL);
	for (int i = 0; i < framenum; ++i) {
		AdvanceFrameMT(threadnum);
	}
	/*for(int i = 0; i < threadnum; ++i) {
		targs[i].tid = i;
		targs[i].frames = framenum;
		pthread_create(&thread[i], &attr, AdvanceFramesMT, &targs[i]);
	}
	// *** PARALLEL PHASE *** //
	for(int i = 0; i < threadnum; ++i) {
		pthread_join(thread[i], NULL);
	}*/
	std::cout << "Critical code execution time: " << time(NULL) - startt << std::endl;
//ROI ends
	if(argc > 4)
		SaveFile(argv[4]);

	CleanUpSim();

	return 0;
}

////////////////////////////////////////////////////////////////////////////////

//------------------------------------------------------------------------
//      ____                        _      _
//     / ___|____ _   _ ____   ____| |__  | |
//    | |   / ___| | | |  _  \/ ___|  _  \| |
//    | |___| |  | |_| | | | | |___| | | ||_|
//     \____|_|  \_____|_| |_|\____|_| |_|(_) Media benchmarks
//                           
//	  2006, Intel Corporation, licensed under Apache 2.0 
//
//  file :	 ParticleFilterOMP.h
//  author : Scott Ettinger - scott.m.ettinger@intel.com
//
//  description : OpenMP parallelized version of the particle filter
//					object derived from ParticleFilter.h
//		
//  modified : Dimitrios Chasapis, Marc Casas - Barcelona Supercomputing Center
//--------------------------------------------------------------------------

#ifndef PARTICLEFILTEROMP_H
#define PARTICLEFILTEROMP_H

#if defined(HAVE_CONFIG_H)
# include "config.h"
#endif

#define WORKUNIT_SIZE_NEWPARTICLES 32
#define WORKUNIT_SIZE_CALCWEIGHTS 32

#include "ParticleFilter.h"
#include <omp.h>
extern int work_units;

template<class T> 
class ParticleFilterOMPSS : public ParticleFilter<T> {

	using ParticleFilter<T>:: mModel;
	using ParticleFilter<T>:: mWeights;
	using ParticleFilter<T>:: mParticles;
	using ParticleFilter<T>:: mNewParticles;
	using ParticleFilter<T>:: mBestParticle;
	using ParticleFilter<T>:: mNParticles;
	using ParticleFilter<T>:: mMinParticles;
	using ParticleFilter<T>:: mBins;  
	using ParticleFilter<T>:: mRnd;
	using ParticleFilter<T>::mInitialized;
	using ParticleFilter<T>::mCdf;
	using ParticleFilter<T>::mSamples;
	typedef typename ParticleFilter<T>::fpType fpType;
	typedef typename ParticleFilter<T>::Vectorf Vectorf;

public:
	ParticleFilterOMPSS() { return; };
	~ParticleFilterOMPSS(){ return; };

  //Update filter to a new set of particles - returns false if model observation fails
	//calls model to get observation at given time, and to get likelihoods of each particle
	bool Update(fpType timeVal);
	
protected:
	std::vector<int> mIndex;																//list of particles to regenerate

	//calculate particle weights - threaded version 
	void CalcWeights(std::vector<Vectorf > &particles);										//calculate particle weights based on model likelihood

	//New particle generation - threaded version 
	void GenerateNewParticles(int k);
	
	void DoCalcLikelihoods(T *mModel, Vectorf *mParticles, float *mWeights, unsigned char *mValid, int bsize)
	{
		int n = omp_get_thread_num();
		//printf("n=%d\n", n);
		for(int i = 0; i < bsize; i++)
		{	
			bool vflag;
			mWeights[i] = mModel->LogLikelihood(mParticles[i], vflag, n);	//compute log-likelihood weights for each particle
			mValid[i] = vflag ? 1 : 0;
		}
	}
	
	void DoGenerateNewParticle(T *mModel, Vectorf *mNewParticles, Vectorf *mParticles, int *mIndex, 
														RandomGenerator *mRnd, int k, int bsize) 
	{
		for(int i = 0; i < bsize; i++) { //distribute new particles randomly according to model stdDevs
			mNewParticles[i] = mParticles[mIndex[i]]; //add new particle for each entry in each bin distributed randomly about duplicated particle
			this->AddGaussianNoise(mNewParticles[i], mModel->StdDevs()[k], mRnd[i]);
		}
	}
};

//Calculate particle weights (mWeights) and find highest likelihood particle. 
//computes an optimal annealing factor and scales the likelihoods. 
template<class T>
void ParticleFilterOMPSS<T>::CalcWeights(std::vector<Vectorf > &particles)
{
	std::vector<unsigned char> valid(particles.size());
	mBestParticle = 0;
	fpType total = 0, best = 0, minWeight = 1e30f, annealingFactor = 1;
	mWeights.resize(particles.size());
	
	int np = (int)particles.size(), j;
	//int n = omp_get_max_threads();
	int n = work_units;
	int step = np/n;
	int remainder = np%n;
	std::vector<float> *_particles = &particles[0];
	float *_mWeights = &mWeights[0];
	unsigned char *_valid = &valid[0];
	//printf("np=%d\n", np);
	for(j = 0; j < np; j+=step) 
	{	
		int bsize = (j+step)>np?remainder:step;
		//#pragma omp task out(_mWeights[j], _valid[j]) in(_particles[j]) firstprivate(j, bsize) label(CalcLikelihoods)
		DoCalcLikelihoods(mModel, &_particles[j], &_mWeights[j], &_valid[j], bsize);
	}
	//#pragma omp taskwait //DoGenerateNewParticle DoCalcLikelihoods barrier
	uint i = 0;
	while(i < particles.size())
	{	if(!valid[i])																		//if not valid(model prior), remove the particle from the list
		{	particles[i] = particles[particles.size() - 1];
			mWeights[i] = mWeights[particles.size() - 1];
			valid[i] = valid[valid.size() - 1];
			particles.pop_back(); mWeights.pop_back(); valid.pop_back();
		}
		else
			minWeight = std::min(mWeights[i++], minWeight);									//find minimum log-likelihood
	}
	if((int)particles.size() < mMinParticles) return;										//bail out if not enough valid particles
	mWeights -= minWeight;																	//shift weights to zero for numerical stability
	if(mModel->StdDevs().size() > 1) 
		annealingFactor = BetaAnnealingFactor(mWeights, 0.5f);								//calculate annealing factor if more than 1 step
	for(i = 0; i < mWeights.size(); i++)
	{	double wa = annealingFactor * mWeights[i];
		mWeights[i] = (float)exp(wa);														//exponentiate log-likelihoods scaled by annealing factor
		total += mWeights[i];																//save sum of all weights
		if(i == 0 || mWeights[i] > best)													//find highest likelihood particle
		{	best = mWeights[i];
			mBestParticle = i;
		}
	}
	mWeights *= fpType(1.0) / total;														//normalize weights
}

//generate new particles distributed with std deviation given by the model annealing parameter - threaded
template<class T> 
void ParticleFilterOMPSS<T>::GenerateNewParticles(int k)
{	int p = 0;
	mNewParticles.resize(mNParticles);
	mIndex.resize(mNParticles);
	for(int i = 0; i < (int)mBins.size(); i++)	 	 	 	 	 	 	 	 	 	
		for(uint j = 0; j < mBins[i]; j++)													//index particles to be regenerated
			mIndex[p++] = i;

	//int np = omp_get_max_threads();
	int np = work_units;
	int step = mNParticles/np;
	int remainder = mNParticles%np;
 	std::vector<float> *_mNewParticles = &mNewParticles[0];
	std::vector<float> *_mParticles = &mParticles[0];
	RandomGenerator *_mRnd = &mRnd[0];
	int *_mIndex = &mIndex[0];

	for(int j=0; j < mNParticles; j+=step) {
		int bsize = (j+step)>mNParticles?remainder:step;
		int index = mIndex[j];
		#pragma omp task inout(_mNewParticles[j]) in(_mParticles, _mIndex[j], _mRnd[j]) firstprivate(index, j, k, bsize) label(GenerateNewParticles)
		DoGenerateNewParticle(mModel, &_mNewParticles[j], _mParticles, &_mIndex[j], &_mRnd[j], k, bsize);
	}
	#pragma omp taskwait
}

//Particle filter update (model and observation updates must be called first)  
template<class T>
bool ParticleFilterOMPSS<T>::Update(fpType timeval)								//weights have already been computed from previous step or initialization
{						
	//printf("OMPSSY OMPSS\n");
	if(!mInitialized)														//check for proper initialization
	{	std::cout << "Update Error : Particles not initialized" << std::endl; 
		return false;
	}	
	if(!mModel->GetObservation(timeval))
	{	std::cout << "Update Error : Model observation failed for time : " << timeval << std::endl;
		return false;
	}
	//First Phase Complete
	printf("first phase complete!\n");
	for(int k = (int)mModel->StdDevs().size() - 1; k >= 0 ; k--)			//loop over all annealing steps starting with highest
	{	
		this->CalcCDF(mWeights, mCdf);											//Monte Carlo re-sampling 
		this->Resample(mCdf, mBins, mSamples, mNParticles);		
		bool minValid = false;
		printf("Resampling done!\n");
		while(!minValid) {
			//this->GenerateNewParticles(k);
			int p = 0;
			mNewParticles.resize(mNParticles);
			mIndex.resize(mNParticles);
			
			for(int i = 0; i < (int)mBins.size(); i++)	 	 	 	 	 	 	 	 	 	
				for(uint j = 0; j < mBins[i]; j++)													//index particles to be regenerated
					mIndex[p++] = i;
			
			//int np = omp_get_max_threads();
			int np = work_units;
			int step = mNParticles/np;
			int remainder = mNParticles%np;
			std::vector<float> *_mNewParticles = &mNewParticles[0];
			std::vector<float> *_mParticles = &mParticles[0];
			RandomGenerator *_mRnd = &mRnd[0];
			int *_mIndex = &mIndex[0];
			
			for(int j=0; j < mNParticles; j+=step) {
				int bsize = (j+step)>mNParticles?remainder:step;
				int index = mIndex[j];
				#pragma omp task inout(_mNewParticles[j]) in(_mParticles, _mIndex[j], _mRnd[j]) firstprivate(index, j, k, bsize) label(DoGenerateNewParticles)
				DoGenerateNewParticle(mModel, &_mNewParticles[j], _mParticles, &_mIndex[j], &_mRnd[j], k, bsize);
			}
			//#pragma omp taskwait
			//this->CalcWeights(mNewParticles);										//calculate particle weights and remove any invalid particles
			//printf("mNewParticles size = %d\n", mNewParticles.size());
			std::vector<unsigned char> valid(mNewParticles.size());
			mBestParticle = 0;
			fpType total = 0, best = 0, minWeight = 1e30f, annealingFactor = 1;
			mWeights.resize(mNewParticles.size());
			
			np = (int)mNewParticles.size();
			int j;
			//int n = omp_get_max_threads();
			int n = work_units;
			step = np/n;
			remainder = np%n;
			std::vector<float> *_particles = &mNewParticles[0];
			float *_mWeights = &mWeights[0];
			unsigned char *_valid = &valid[0];
			//printf("np=%d\n", np);
			for(j = 0; j < np; j+=step) 
			{	
				int bsize = (j+step)>np?remainder:step;
				#pragma omp task out(_mWeights[j], _valid[j]) in(_particles[j]) firstprivate(j, bsize) label(DoCalcLikelihoods)
				DoCalcLikelihoods(mModel, &_particles[j], &_mWeights[j], &_valid[j], bsize);
			} 
			#pragma omp taskwait //DoGenerateNewParticle DoCalcLikelihoods barrier
			//#pragma omp task out(_particles[0;np], _mWeights[0;np], )
			uint i = 0;
			while(i < mNewParticles.size())
			{	if(!valid[i])																		//if not valid(model prior), remove the particle from the list
				{	mNewParticles[i] = mNewParticles[mNewParticles.size() - 1];
					mWeights[i] = mWeights[mNewParticles.size() - 1];
					valid[i] = valid[valid.size() - 1];
					mNewParticles.pop_back(); mWeights.pop_back(); valid.pop_back();
				}
				else
					minWeight = std::min(mWeights[i++], minWeight);									//find minimum log-likelihood
			}
			if((int)mNewParticles.size() >= mMinParticles) {										//bail out if not enough valid particles
				mWeights -= minWeight;																	//shift weights to zero for numerical stability
				if(mModel->StdDevs().size() > 1) 
					annealingFactor = BetaAnnealingFactor(mWeights, 0.5f);								//calculate annealing factor if more than 1 step
				for(i = 0; i < mWeights.size(); i++)
				{	double wa = annealingFactor * mWeights[i];
					mWeights[i] = (float)exp(wa);														//exponentiate log-likelihoods scaled by annealing factor
					total += mWeights[i];																//save sum of all weights
					if(i == 0 || mWeights[i] > best)													//find highest likelihood particle
					{	best = mWeights[i];
						mBestParticle = i;
					}
				}
				mWeights *= fpType(1.0) / total;														//normalize weights
				minValid = (int)mNewParticles.size() >= mMinParticles;			//repeat if not enough valid particles
			}
			//#pragma omp taskwait
			if(!minValid) 
				std::cout << "Not enough valid particles - Resampling!!!" << std::endl;
		}
		mParticles = mNewParticles;											//save new particle set
	}
	return true;
}

#endif


/*
	Copyright 2014-2015 Raul Vidal Ortiz
	Distributed under the same license as the rest of PhysBAM
*/


#ifndef __OMPSS_H__
#define __OMPSS_H__

#include <omp.h>
#include <sys/time.h>
#include <iostream>
namespace PhysBAM
{
    class OMPSS_POOL
    {
	private:
	    static OMPSS_POOL* singleton_instance;
	    unsigned long number_of_divisions;
	public:
        double elapsed_frame;
        double elapsed_simulation;
        double frame_tstart;
        double frame_tend;
        double simulation_tstart;
        double simulation_tend;
	    static inline OMPSS_POOL* Singleton()
	    {
		    if (!singleton_instance) singleton_instance = new OMPSS_POOL();
		    return singleton_instance;
	    }

        double getusec_()
        {
            struct timeval time;
            gettimeofday(&time, 0);
            return ((double)time.tv_sec * (double)1e6 + (double)time.tv_usec);
        }
	    
//#####################################################################
	    OMPSS_POOL();
	    ~OMPSS_POOL();
        void printBSClogo();
	    unsigned long Get_n_divisions();
	    bool Set_n_divisions(unsigned long n);
//#####################################################################
    };

}
#endif //__OMPSS_H__

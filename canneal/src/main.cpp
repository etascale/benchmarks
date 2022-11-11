// main.cpp
//
// Created by Daniel Schwartz-Narbonne on 13/04/07.
// Modified by Christian Bienia for the PARSEC suite
// OmpSs/OpenMP 4.0 versions written by Dimitrios Chasapis - Barcelona Supercomputing Center
//
// Copyright 2007-2008 Princeton University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.


#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>

#ifdef ENABLE_THREADS
#include <pthread.h>
#endif

#ifdef ENABLE_PARSEC_HOOKS
#include <hooks.h>
#endif

#include "annealer_types.h"
#include "annealer_thread.h"
#include "netlist.h"
#include "rng.h"

#if defined ENABLE_EXTRAE

#include "extrae.h"
#ifdef __cplusplus
extern "C" {
#endif
extrae_type_t manualevent;
unsigned extrae_nvalues;
char ** extrae_descriptions;
extrae_value_t* extrae_values;
void extrae_register_event_types() {

  manualevent = EXTRAEUSERFUNCTION;
  int error = 0;
  extrae_nvalues = 3;
  
  extrae_values = (extrae_value_t *)calloc(extrae_nvalues,sizeof(extrae_value_t));
  for (int i = 0; i < extrae_nvalues; i++) extrae_values[i] = i;
  extrae_descriptions = (char**)calloc(extrae_nvalues,sizeof(char*));
  char description0[] = "END";
  char description1[] = "SWAPTASK";
  char description2[] = "BARRIER";
  extrae_descriptions[0] = (char*)calloc(strlen(description0)+1,sizeof(char));
  extrae_descriptions[1] = (char*)calloc(strlen(description1)+1,sizeof(char));
  extrae_descriptions[2] = (char*)calloc(strlen(description2)+1,sizeof(char));
  strncpy(extrae_descriptions[0],description0,strlen(description0)+1);
  strncpy(extrae_descriptions[1],description1,strlen(description1)+1);
  strncpy(extrae_descriptions[2],description2,strlen(description2)+1);
  char et_description[] = "manual_events";
  char* event_type_description = (char*)calloc(256,sizeof(char));
  strncpy(event_type_description,et_description,strlen(et_description));
  

  Extrae_define_event_type(&manualevent,event_type_description,&extrae_nvalues,extrae_values,extrae_descriptions); 

  return;
}
#ifdef __cplusplus
}
#endif
#endif

using namespace std;

void* entry_pt(void*);



int main (int argc, char * const argv[]) {
  
#ifdef ENABLE_PARSEC_HOOKS
   __parsec_bench_begin(__parsec_canneal);
#endif
	srandom(3);

#if defined(ENABLE_OMPSS) || defined(ENABLE_OMP2) || defined(ENABLE_OMP4)
	if(argc != 6 && argc != 7) {
		cout << "Usage: " << argv[0] << " NTHREADS NSWAPS TEMP NETLIST NSTEPS [NTASKS]\nWarning: NTHREADS is ingored! Use NX_ARGS for OMPSs or OMP_NUM_THREADS for OpenMP 4.0\n" << endl;
		exit(1);
	}

	int num_threads = argv[1];
	if(argc > 6) {
		num_threads = atoi(argv[6]);
	}

#else

	if(argc != 6 && argc != 7) {
		cout << "Usage: " << argv[0] << " NTHREADS NSWAPS TEMP NETLIST NSTEPS [NTASKS]\nWarning: NTASKS has no effect in the serial version" << endl;
		exit(1);
	}	

	int num_threads = atoi(argv[1]);
#endif
	//argument 1 is numthreads
	cout << "Taskcount: " << num_threads << endl;
#if defined(ENABLE_THREADS) || defined(ENABLE_OMPSS) || defined (ENABLE_OMP2) || defined(ENABLE_OMP4)
	if (num_threads < 1){
		cout << "NTHREADS/NTASKS must be at least 1" <<endl;
		exit(1);
	}
#else
	if (num_threads != 1){
		cout << "NTHREADS must be 1 (serial version)" <<endl;
		exit(1);
	}
#endif

#if defined ENABLE_EXTRAE
    extrae_register_event_types();
#endif
		
	//argument 2 is the num moves / temp
	int swaps_per_temp = atoi(argv[2]);
	cout << swaps_per_temp << " swaps per temperature step" << endl;

	//argument 3 is the start temp
	int start_temp =  atoi(argv[3]);
	cout << "start temperature: " << start_temp << endl;
	
	//argument 4 is the netlist filename
	string filename(argv[4]);
	cout << "netlist filename: " << filename << endl;
	
	//argument 5 (optional) is the number of temperature steps before termination
	int number_temp_steps = -1;
        if(argc >= 6) {
		number_temp_steps = atoi(argv[5]);
		cout << "number of temperature steps: " << number_temp_steps << endl;
        }
        
	//now that we've read in the commandline, run the program
	netlist my_netlist(filename);
	
	annealer_thread a_thread(&my_netlist,num_threads,swaps_per_temp,start_temp,number_temp_steps);
	
#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_begin();
#endif

//**** ROI begins ****//
	int startt = time(NULL);
#ifdef ENABLE_THREADS
	std::vector<pthread_t> threads(num_threads);
	void* thread_in = static_cast<void*>(&a_thread);
	for(int i=0; i<num_threads; i++){
		pthread_create(&threads[i], NULL, entry_pt,thread_in);
	}
	for (int i=0; i<num_threads; i++){
		pthread_join(threads[i], NULL);
	}
#else
	a_thread.Run();
#endif
	std::cout << "Critical code execution time: " << time(NULL) - startt << std::endl;
//**** ROI ends ****//

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_roi_end();
#endif
	
	cout << "Final routing is: " << my_netlist.total_routing_cost() << endl;

#ifdef ENABLE_PARSEC_HOOKS
    __parsec_bench_end();
#endif

	return 0;
}

void* entry_pt(void* data)
{
	annealer_thread* ptr = static_cast<annealer_thread*>(data);
	ptr->Run();
}

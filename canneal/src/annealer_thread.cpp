// annealer_thread.cpp
//
// Created by Daniel Schwartz-Narbonne on 14/04/07.
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

#ifdef ENABLE_THREADS
#include <pthread.h>
#endif

#include <cassert>
#include "annealer_thread.h"
#include "location_t.h"
#include "annealer_types.h"
#include "netlist_elem.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include "rng.h"

using std::cout;
using std::endl;


//*****************************************************************************************
//
//*****************************************************************************************
void annealer_thread::Run()
#ifdef ENABLE_OMPSS
{
	int accepted_good_moves=0;
	int accepted_bad_moves=-1;
	double T = _start_temp;
	Rng *rng; //store of randomness
	rng = new Rng[_ntasks];

	int temp_steps_completed=0; 
	while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
		T = T / 1.5;
		int acc_good[_ntasks];
		int acc_bad[_ntasks];
		for (int j = 0; j < _ntasks; ++j) {
			#pragma omp task firstprivate(j) out(acc_good[j], acc_bad[j]) default(shared) label(swap_task)
			{
			long a_id;
			long b_id;
			netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng[j]);
			netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng[j]);
			acc_good[j] = 0;
			acc_bad[j] = 0;
			int iters = _swaps_per_temp/_ntasks;
			if (_swaps_per_temp%_ntasks > j) ++iters;
			for (int i = 0; i < iters; i++){
				//get a new element. Only get one new element, so that reuse should help the cache
				a = b;
				a_id = b_id;
				b = _netlist->get_random_element(&b_id, a_id, &rng[j]);
				
				routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
				move_decision_t is_good_move = accept_move(delta_cost, T, &rng[j]);

				//make the move, and update stats:
				if (is_good_move == move_decision_accepted_bad){
					acc_bad[j]++;
					_netlist->swap_locations(a,b);
				} else if (is_good_move == move_decision_accepted_good){
					acc_good[j]++;
					_netlist->swap_locations(a,b);
				} else if (is_good_move == move_decision_rejected){
				  //no need to do anything for a rejected move
				}
			}
			}
		}
		temp_steps_completed++;
		#pragma omp taskwait
	}
}
#elif defined ENABLE_OMP4
{
	#pragma omp parallel
	{
		#pragma omp single
		{ 
			int accepted_good_moves=0;
			int accepted_bad_moves=-1;
			double T = _start_temp;
			Rng *rng; //store of randomness
			rng = new Rng[_ntasks];

			int temp_steps_completed=0; 
			while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
				T = T / 1.5;
				int acc_good[_ntasks];
				int acc_bad[_ntasks];
				for (int j = 0; j < _ntasks; ++j) {
					//#pragma omp task firstprivate(j) depend(out: acc_good[j], acc_bad[j]) default(shared)
					#pragma omp task firstprivate(j) default(shared)
					{
					long a_id;
					long b_id;
					netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng[j]);
					netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng[j]);
					acc_good[j] = 0;
					acc_bad[j] = 0;
					int iters = _swaps_per_temp/_ntasks;
					if (_swaps_per_temp%_ntasks > j) ++iters;
					for (int i = 0; i < iters; i++){
						//get a new element. Only get one new element, so that reuse should help the cache
						a = b;
						a_id = b_id;
						b = _netlist->get_random_element(&b_id, a_id, &rng[j]);
				
						routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
						move_decision_t is_good_move = accept_move(delta_cost, T, &rng[j]);

						//make the move, and update stats:
						if (is_good_move == move_decision_accepted_bad){
							acc_bad[j]++;
							_netlist->swap_locations(a,b);
						} else if (is_good_move == move_decision_accepted_good){
							acc_good[j]++;
							_netlist->swap_locations(a,b);
						} else if (is_good_move == move_decision_rejected){
							//no need to do anything for a rejected move
						}
					}
					}
				}
				temp_steps_completed++;
				#pragma omp taskwait
			}
		} // end of single
	} //end of parallel region
}
#elif defined ENABLE_OMP2
{
	#pragma omp parallel
	{
			int accepted_good_moves=0;
			int accepted_bad_moves=-1;
			double T = _start_temp;
			Rng *rng; //store of randomness
			rng = new Rng[_ntasks];

			int temp_steps_completed=0; 
			while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
				T = T / 1.5;
				int acc_good[_ntasks];
				int acc_bad[_ntasks];
				#pragma omp for schedule(SCHED_POLICY)
				for (int j = 0; j < _ntasks; ++j) {
					long a_id;
					long b_id;
					netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng[j]);
					netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng[j]);
					acc_good[j] = 0;
					acc_bad[j] = 0;
					int iters = _swaps_per_temp/_ntasks;
					if (_swaps_per_temp%_ntasks > j) ++iters;
					for (int i = 0; i < iters; i++){
						//get a new element. Only get one new element, so that reuse should help the cache
						a = b;
						a_id = b_id;
						b = _netlist->get_random_element(&b_id, a_id, &rng[j]);
				
						routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
						move_decision_t is_good_move = accept_move(delta_cost, T, &rng[j]);

						//make the move, and update stats:
						if (is_good_move == move_decision_accepted_bad){
							acc_bad[j]++;
							_netlist->swap_locations(a,b);
						} else if (is_good_move == move_decision_accepted_good){
							acc_good[j]++;
							_netlist->swap_locations(a,b);
						} else if (is_good_move == move_decision_rejected){
							//no need to do anything for a rejected move
						}
					}
				}
				temp_steps_completed++;
				#pragma omp taskwait
			}
	} //end of parallel region
}
#else
{
	int accepted_good_moves=0;
	int accepted_bad_moves=-1;
	double T = _start_temp;
	Rng rng; //store of randomness
	
	long a_id;
	long b_id;
	netlist_elem* a = _netlist->get_random_element(&a_id, NO_MATCHING_ELEMENT, &rng);
	netlist_elem* b = _netlist->get_random_element(&b_id, NO_MATCHING_ELEMENT, &rng);

	int temp_steps_completed=0; 
	while(keep_going(temp_steps_completed, accepted_good_moves, accepted_bad_moves)){
        #if defined ENABLE_EXTRAE
        Extrae_eventandcounters(manualevent,SWAPTASK_START);
        #endif
		
        T = T / 1.5;
		accepted_good_moves = 0;
		accepted_bad_moves = 0;
		
		
        for (int i = 0; i < _swaps_per_temp; i++){
			//get a new element. Only get one new element, so that reuse should help the cache
			a = b;
			a_id = b_id;
			b = _netlist->get_random_element(&b_id, a_id, &rng);
			
			routing_cost_t delta_cost = calculate_delta_routing_cost(a,b);
			move_decision_t is_good_move = accept_move(delta_cost, T, &rng);

			//make the move, and update stats:
			if (is_good_move == move_decision_accepted_bad){
				accepted_bad_moves++;
				_netlist->swap_locations(a,b);
			} else if (is_good_move == move_decision_accepted_good){
				accepted_good_moves++;
				_netlist->swap_locations(a,b);
			} else if (is_good_move == move_decision_rejected){
				//no need to do anything for a rejected move
			}
		}
        
        #if defined ENABLE_EXTRAE
        Extrae_eventandcounters(manualevent,END);
        #endif
		
        temp_steps_completed++;
#ifdef ENABLE_THREADS
        #if defined ENABLE_EXTRAE
        Extrae_eventandcounters(manualevent,BARRIER_START);
        #endif
		pthread_barrier_wait(&_barrier);
        #if defined ENABLE_EXTRAE
        Extrae_eventandcounters(manualevent,END);
        #endif
#endif
	}
}
#endif

//*****************************************************************************************
//
//*****************************************************************************************
annealer_thread::move_decision_t annealer_thread::accept_move(routing_cost_t delta_cost, double T, Rng* rng)
{
	//always accept moves that lower the cost function
	if (delta_cost < 0){
		return move_decision_accepted_good;
	} else {
		double random_value = rng->drand();
		double boltzman = exp(- delta_cost/T);
		if (boltzman > random_value){
			return move_decision_accepted_bad;
		} else {
			return move_decision_rejected;
		}
	}
}


//*****************************************************************************************
//  If get turns out to be expensive, I can reduce the # by passing it into the swap cost fcn
//*****************************************************************************************
routing_cost_t annealer_thread::calculate_delta_routing_cost(netlist_elem* a, netlist_elem* b)
{
	location_t* a_loc = a->present_loc.Get();
	location_t* b_loc = b->present_loc.Get();
	
	routing_cost_t delta_cost = a->swap_cost(a_loc, b_loc);
	delta_cost += b->swap_cost(b_loc, a_loc);

	return delta_cost;
}

//*****************************************************************************************
//  Check whether design has converged or maximum number of steps has reached
//*****************************************************************************************
bool annealer_thread::keep_going(int temp_steps_completed, int accepted_good_moves, int accepted_bad_moves)
{
	bool rv;

	if(_number_temp_steps == -1) {
		//run until design converges
		rv = _keep_going_global_flag && (accepted_good_moves > accepted_bad_moves);
		if(!rv) _keep_going_global_flag = false; // signal we have converged
	} else {
		//run a fixed amount of steps
		rv = temp_steps_completed < _number_temp_steps;
	}

	return rv;
}


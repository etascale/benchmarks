//#####################################################################
// Copyright 2003-2006, Zhaosheng Bao, Ron Fedkiw, Geoffrey Irving, Igor Neverov, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
// OmpSs/OpenMP 4.0 versions by Raul Vidal Ortiz - Barcelona Supercomputing Center
//#####################################################################
// Class DEFORMABLE_OBJECT
//#####################################################################
#include "OMPSS_DEFORMABLE_OBJECT.h"
#include "../Forces_And_Torques/SOLIDS_FORCES.h"
#include "../Arrays/OMPSS_ARRAY_PARALLEL_OPERATIONS.h"
#include "../Arrays/ARRAY_RANGE.h"
#include "../Utilities/LOG.h"
#include "../Utilities/DEBUG_UTILITIES.h"
#include "../Thread_Utilities/OMPSS_POOL.h"
#ifdef ENABLE_OMPEXTRAE
	#include "../Utilities/EXTRAE.h"
#endif

using namespace PhysBAM;

//#####################################################################
// Function Save_Velocity
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Save_Velocity()
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Euler_Step_Position
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Euler_Step_Position (const T dt)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Euler_Step_Velocity
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Euler_Step_Velocity (const T dt, const T time)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Euler_Step_Position_And_Velocity
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Euler_Step_Position_And_Velocity (const T dt, const T time)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Predictor_Corrector_Step_Velocity
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Predictor_Corrector_Step_Velocity (const T dt, const T time, const int corrector_steps)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Predictor_Corrector_Integrate_Velocity
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Predictor_Corrector_Integrate_Velocity (const T start_time, const T end_time, const int corrector_steps)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Backward_Euler_Step_Velocity_With_Fallback
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Backward_Euler_Step_Velocity_With_Fallback (const T dt, const T time, const T convergence_tolerance, const int max_iterations, const bool verbose)
{
	NOT_IMPLEMENTED();
}

//#####################################################################
// Function Backward_Euler_Step_Velocity
//#####################################################################
// returns true if converged, false if not (in which case particles->V is reset to old values)
// assumes all solids_forces are linear in velocity, with a symmetric positive definite Jacobian.
#ifndef AGGREGATE_CG_OPERATIONS
template<class T, class TV> bool DEFORMABLE_OBJECT<T, TV>::
Backward_Euler_Step_Velocity (const T dt, const T time, const T convergence_tolerance, const int max_iterations, const bool use_forward_euler_initial_guess, int* iterations_used,
			      const bool damping_only)
{
	NOT_IMPLEMENTED();
}

#else
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Backward_Euler_Step_Velocity_CG_Helper_I (long thread_id, void* helper_raw)
{
	NOT_IMPLEMENTED();
}

template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Backward_Euler_Step_Velocity_CG_Helper_II (long thread_id, void* helper_raw)
{
	NOT_IMPLEMENTED();
}

template<class T, class TV> bool DEFORMABLE_OBJECT<T, TV>::
Backward_Euler_Step_Velocity (const T dt, const T time, const T convergence_tolerance, const int max_iterations, const bool use_forward_euler_initial_guess, int* iterations_used,
			      const bool damping_only)
{
	NOT_IMPLEMENTED();
}

#endif
//#####################################################################
// Function One_Newton_Step_Toward_Steady_State
//#####################################################################
#ifndef AGGREGATE_CG_OPERATIONS
template<class T, class TV> bool DEFORMABLE_OBJECT<T, TV>::
One_Newton_Step_Toward_Steady_State (const T convergence_tolerance, const int max_iterations, const T time, ARRAY<TV>& dX, const bool balance_external_forces_only,
				     int* iterations_used, const bool update_positions_and_state)
{
	NOT_IMPLEMENTED();
}

#else

/*CG HELPER I*/
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
CG_Helper_I(int p, T beta)
{
    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(1);
    #endif

    int START = (*particles.particle_ranges)(p).x; int END = (*particles.particle_ranges)(p).y;
    for (int i = START; i <= END; i++) S_full (i) = beta * S_full (i) + R_full (i); //S_full == dX_full
    Force_Differential_Internal (S_full, F_full, p); // S_full == dX_full

    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(0);
    #endif
}

/*CG HELPER II*/
#ifdef ENABLE_OPENMP
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
CG_Helper_II(int p, double& partial_S_dot_Q, T time)
{
    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(1);
    #endif
    
    double& local_S_dot_Q = partial_S_dot_Q;
    ARRAY_RANGE<ARRAY<TV> > S(S_full, (*particles.particle_ranges)(p));
    //Force_Differential(S_full,F_full,p);
    Force_Differential_External (S_full, F_full, p);
    external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (F_full, time, id_number, p);
    local_S_dot_Q = -ARRAY<TV>::template Vector_Dot_Product<double> (S, F_full.Range ((*particles.particle_ranges)(p)));
    
    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(0);
    #endif
}
#elif defined ENABLE_OMPSS
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
CG_Helper_II(int p, double& S_dot_Q, T time)
{
    double local_S_dot_Q = 0.; 
    ARRAY_RANGE<ARRAY<TV> > S(S_full, (*particles.particle_ranges)(p));
    //Force_Differential(S_full,F_full,p);
    Force_Differential_External (S_full, F_full, p);
    external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (F_full, time, id_number, p);
    local_S_dot_Q = -ARRAY<TV>::template Vector_Dot_Product<double> (S, F_full.Range ((*particles.particle_ranges)(p)));
    #pragma omp atomic
    S_dot_Q += local_S_dot_Q;
}
#endif

/*CG HELPER III*/
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
CG_Helper_III(int p,ARRAY<TV>& dX_full, 
                    double& rho_new, double& supnorm, double rho, double S_dot_Q)
{
    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(1);
    #endif

    double local_rho_new = 0.;
    double local_supnorm = 0.;
    T local_alpha = (T) (rho / S_dot_Q);
    int START = (*particles.particle_ranges)(p).x; int END = (*particles.particle_ranges)(p).y;
    for (int i = START; i <= END; i++)
    {
        dX_full (i) += local_alpha * S_full (i);
        R_full (i) += local_alpha * F_full (i);
        double s2 = R_full (i).Magnitude_Squared();
        local_rho_new += s2;
        local_supnorm = max(local_supnorm,s2);
    }
    #pragma omp atomic
    rho_new += local_rho_new;
    #pragma omp critical(supnormreduction) 
    supnorm = max(local_supnorm,supnorm); 
    /* currently this does not return the correct maximum, needs an array 
     * and then do the max reduction with one element of the array as 
     * initialization
     */

    #ifdef ENABLE_OMPEXTRAE
    Extrae_user_function(0);
    #endif
    
    /* False sharing version */
    /* for(int i=1;i<=dX.m;i++)
        {
          dX(i)+=alpha*S(i);
          R(i)+=alpha*negative_Q(i);
          double s2=R(i).Magnitude_Squared();
          rho_new_partial(p)+=s2;
          supnorm_partial(p)=max(supnorm,s2);
        }
    */
}

#ifdef ENABLE_OPENMP

/************OPENMP START*************/
template<class T, class TV> bool DEFORMABLE_OBJECT<T, TV>::
One_Newton_Step_Toward_Steady_State (const T convergence_tolerance, const int max_iterations, const T time, ARRAY<TV>& dX_full, const bool balance_external_forces_only,
									int* iterations_used, const bool update_positions_and_state)
{
	LOG::cout << std::setprecision(std::numeric_limits<long double>::digits10);
	LOG::Push_Scope ("NRS:", "NRS:");
	int i, N = particles.number;
	LOG::Time ("NRS - Initialize:");
	dX_full.Resize_Array (N); // an initial guess might be passed in for dX, otherwise it's zero
	R_full.Resize_Array (N, false, false);
	LOG::Time ("NRS - Boundary conditions 1:");
	external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (dX_full, time, id_number); //Serial, 100-150 microsecs
    
    ARRAY<TV> dF_fd(N);
    ARRAY<TV> dF_avif(N);
	
	if (update_positions_and_state) external_forces_and_velocities->Set_External_Positions (particles.X.array, time, id_number); //Serial update positions
	
	//LOG& logger = LOG::Get_Instance();

	LOG::Stop_Time();

	if (update_positions_and_state) 
	{
        Update_Position_Based_State(); //update state (stiffness, forces), rigid body (jaw, cranium) and deformable body
	}
	//#pragma omp taskwait
    Force_Differential (dX_full, dF_fd); //No barrier, includes a Clear_Parallel
	if (!balance_external_forces_only)
	{
        Add_Velocity_Independent_Forces (dF_avif);
	}
    #pragma omp taskwait
    R_full+=dF_fd;
    R_full+=dF_avif;
	LOG::Time ("NRS - Boundary conditions 2:");
	external_forces_and_velocities->Add_External_Forces (R_full, time, id_number);
	external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (R_full, time, id_number); //Serial, 100-150 microsecs
	LOG::Stop_Time();
	LOG::Push_Scope ("NRS - Compute residual:", "NRS - Compute residual:");
	double rho = 0, supnorm = 0;
	ARRAY_PARALLEL_OPERATIONS<TV, T, TV>::Dot_Product_Parallel (R_full, R_full, *particles.particle_ranges,rho); //barrier
	ARRAY_PARALLEL_OPERATIONS<TV, T, TV>::Maximum_Magnitude_Squared_Parallel (R_full, *particles.particle_ranges,supnorm); //barrier
	
	supnorm = sqrt (supnorm);
	LOG::Pop_Scope(); // from Compute residual

	if (supnorm <= convergence_tolerance) //if(print_diagnostics)LOG::cout<<"no cg iterations needed"<<std::endl;
	{
		if (iterations_used) *iterations_used = 0;

		LOG::Pop_Scope(); // from NRS
		return true;
	} // exit early

	LOG::Time ("NRS - Copy initial guess:");
	ARRAY<TV>& negative_Q_full = F_full;
	negative_Q_full.Resize_Array (N, false, false);
	S_full.Resize_Array (N, false, false);
	LOG::Stop_Time();
	LOG::Push_Scope ("CGI:", "CGI:");
	int iterations;
	T beta = 0;
	LOG::cout << "rho before cg iters: " << rho << " || supnorm before cg iters: " << supnorm << std::endl;
    /*Additional references in order to be able to put them as dependencies*/
    ARRAY<TV>& local_R_full = R_full;
    ARRAY<TV>& local_S_full = S_full; ARRAY<TV>& local_F_full = F_full; SOLIDS_PARTICLE<T, TV>& local_particles = particles;
    double partial_S_dot_Q[particles.particle_ranges->m];
    double S_dot_Q = 0.;
    double rho_new = 0;
#if defined USE_WORKSHARING_FOR || USE_HYBRID
    long gl_iterations = 1;
    #if defined USE_HYBRID
       omp_set_nested(1);
    #endif
    #pragma omp parallel shared(partial_S_dot_Q,S_dot_Q,rho_new,dX_full,local_R_full,local_S_full,beta,rho,supnorm,gl_iterations) firstprivate(iterations)
    {
#endif
        for (iterations = 1; iterations <= max_iterations; iterations++)
        {
            //LOG::Time("CGI - Update solution I:");
            S_dot_Q = 0.;
            rho_new = 0;
            //LOG::cout << "CG ITERATION " << iterations << "*******" << std::endl;
            #if defined USE_TASKS && !defined USE_HYBRID
                    //#pragma omp task depend(in:local_S_full) shared(local_S_full,local_F_full,local_particles) firstprivate(beta)
                {
                for (int p=1; p<=particles.particle_ranges->m; p++)
                {
                    #pragma omp task firstprivate(p,beta) depend(in:local_S_full) shared(local_S_full,local_F_full,local_particles) //firstprivate(p,beta)
                    {
                        CG_Helper_I(p,beta);
                    }
                }
                }
                //The master thread does not steal any task from the nested ones.
                //It seems that the creator thread only steals from the tasks created by him.
                //Removing barrier.
                #pragma omp task depend(inout:local_S_full) shared(local_S_full)
                {
                    /*This task is here in order to not have all previous tasks serialized. This way, we 
                        avoid barriers and master thread can continue generating tasks*/
                    int p = 0;
                }
                for (int p = 1; p <= particles.particle_ranges->m; p++)
                {
                    #pragma omp task depend(in:local_S_full,S_dot_Q) \
                    shared(partial_S_dot_Q,local_F_full,local_particles,S_dot_Q,local_S_full) \
                    firstprivate(p) 
                    CG_Helper_II(p,partial_S_dot_Q[p-1],time);
                }
                //LOG::cout << "Alpha: " << alpha << " || rho: " << rho << " || S_dot_Q: " << S_dot_Q << std::endl;
                /* This task is here in order to have CG_Helper_III to depend on S_dot_Q. Else, 
                 * we would have a barrier, stopping task creation
                 */
                #pragma omp task depend(inout:S_dot_Q) shared(S_dot_Q,partial_S_dot_Q)
                {
                    #ifdef ENABLE_OMPEXTRAE
                    Extrae_user_function(1);
                    #endif
                    for (int i = 0; i < particles.particle_ranges->m; i++) S_dot_Q += partial_S_dot_Q[i];
                    #ifdef ENABLE_OMPEXTRAE
                    Extrae_user_function(0);
                    #endif
                }
                //LOG::Stop_Time();
                //LOG::Time("CGI - Update solution II:");
                for (int p = 1; p <= particles.particle_ranges->m; p++)
                {
                    #pragma omp task depend(in:S_dot_Q) \
                    shared(S_dot_Q,dX_full,local_R_full,local_F_full,local_S_full,rho_new,supnorm,local_particles) \
                    firstprivate(p) 
                    CG_Helper_III(p,dX_full,rho_new,supnorm,rho,S_dot_Q);
                    
                }
                #pragma omp taskwait
                //LOG::cout << "rho_new after Helper III: " << rho_new << " || supnorm " << supnorm << std::endl;
                supnorm = sqrt (supnorm);
                //LOG::Stop_Time();
                //LOG::cout << "cg supnorm before comparing against convergence: " << supnorm << std::endl;
                if (supnorm <= convergence_tolerance) {
                    //LOG::cout << "cg supnorm converted after " << iterations << " Converged value:" << supnorm << std::endl;
                    break;
                }
                beta = (T) (rho_new / rho);
                rho = rho_new;
            
            /**************WORKSHARING_FOR******************/
            #elif defined USE_WORKSHARING_FOR || USE_HYBRID
                #pragma omp for schedule(static,1)
                for (int p = 1; p <= particles.particle_ranges->m; p++)
                    CG_Helper_I(p,beta);
            
                #pragma omp for schedule(static,1) 
                for (int p = 1; p <= particles.particle_ranges->m; p++)
                    CG_Helper_II(p,partial_S_dot_Q[p-1],time);
                
                #pragma omp single
                {
                    for (int p = 0; p<particles.particle_ranges->m; p++)
                        S_dot_Q += partial_S_dot_Q[p];
                }

                //LOG::Stop_Time();
                //LOG::Time("CGI - Update solution II:");
                
                #pragma omp for schedule(static,1)
                for (int p = 1; p <= particles.particle_ranges->m; p++)
                    CG_Helper_III(p,dX_full,rho_new,supnorm,rho,S_dot_Q);
                //LOG::cout << "rho_new after Helper III: " << rho_new << " || supnorm " << supnorm << std::endl;
                
                #pragma omp single
                {
                    supnorm = sqrt (supnorm);
                }
                //LOG::Stop_Time();
                //LOG::cout << "cg supnorm before comparing against convergence: " << supnorm << std::endl;
                if (supnorm <= convergence_tolerance) {
                    #pragma omp atomic write
                    gl_iterations = iterations;
                    //LOG::cout << "cg supnorm converted after " << iterations << " Converged value:" << supnorm << std::endl;
                    break;
                }
                #pragma omp single
                {
                    beta = (T) (rho_new / rho);
                    rho = rho_new;
                    gl_iterations = iterations+1;
                }
            #endif //USE_WORKSHARING_FOR
        }
#if defined USE_WORKSHARING_FOR || USE_HYBRID
    } // omp parallel
        iterations = gl_iterations;
    #if defined USE_HYBRID
        omp_set_nested(0);
    #endif
#endif

	if (iterations_used) *iterations_used = min (iterations, max_iterations);
	LOG::Pop_Scope(); // from CGI

	if (iterations <= max_iterations) 
    {	
		//LOG::cout << "cg iterations = " << iterations << std::endl;
		LOG::Pop_Scope(); // from NRS
		return true;
	}
	else
	{
        LOG::cout << "cg not converged after " << max_iterations << " iterations  - error = " << supnorm << std::endl;
		LOG::Pop_Scope(); // from NRS
		return false;
	}

	LOG::Pop_Scope();
	LOG::cout.unsetf(std::ios_base::floatfield);
}
/************OPENMP END***************/
#elif defined ENABLE_OMPSS //OMPSS

/************OMPSS START**************/
template<class T, class TV> bool DEFORMABLE_OBJECT<T, TV>::
One_Newton_Step_Toward_Steady_State (const T convergence_tolerance, const int max_iterations, const T time, ARRAY<TV>& dX_full, const bool balance_external_forces_only,
									int* iterations_used, const bool update_positions_and_state)
{
	LOG::cout << std::setprecision(std::numeric_limits<long double>::digits10);
	LOG::Push_Scope ("NRS:", "NRS:");
	int i, N = particles.number;
	LOG::Time ("NRS - Initialize:");
	dX_full.Resize_Array (N); // an initial guess might be passed in for dX, otherwise it's zero
	R_full.Resize_Array (N, false, false);
	LOG::Time ("NRS - Boundary conditions 1:");
	external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (dX_full, time, id_number); //Serial, 100-150 microsecs
	
	if (update_positions_and_state) external_forces_and_velocities->Set_External_Positions (particles.X.array, time, id_number); //update positions Serial
	
	//LOG& logger = LOG::Get_Instance();

    ARRAY<TV> dF_fd(N);
    ARRAY<TV> dF_avif(N);

	LOG::Stop_Time();

	if (update_positions_and_state) 
	{
        Update_Position_Based_State(); //update state (stiffness, forces), rigid body (jaw, cranium) and deformable body
	}
	//#pragma omp taskwait
    Force_Differential (dX_full, dF_fd); //Add force differential, has barrier. This call includes a Clear_Parallel
	if (!balance_external_forces_only)
	{
        Add_Velocity_Independent_Forces (dF_avif);
	}
    #pragma omp taskwait
    R_full += dF_fd;
    R_full += dF_avif;
	LOG::Time ("NRS - Boundary conditions 2:");
	external_forces_and_velocities->Add_External_Forces (R_full, time, id_number);
	external_forces_and_velocities->Zero_Out_Enslaved_Position_Nodes (R_full, time, id_number); //Serial, 100-150 microsecs
	LOG::Stop_Time();
	LOG::Push_Scope ("NRS - Compute residual:", "NRS - Compute residual:");
	double rho = 0, supnorm = 0;
	ARRAY_PARALLEL_OPERATIONS<TV, T, TV>::Dot_Product_Parallel (R_full, R_full, *particles.particle_ranges,rho); //barrier
	ARRAY_PARALLEL_OPERATIONS<TV, T, TV>::Maximum_Magnitude_Squared_Parallel (R_full, *particles.particle_ranges,supnorm); //barrier
	
	supnorm = sqrt (supnorm);
	LOG::Pop_Scope(); // from Compute residual

	if (supnorm <= convergence_tolerance) //if(print_diagnostics)LOG::cout<<"no cg iterations needed"<<std::endl;
	{
		if (iterations_used) *iterations_used = 0;

		LOG::Pop_Scope(); // from NRS
		return true;
	} // exit early

	LOG::Time ("NRS - Copy initial guess:");
	ARRAY<TV>& negative_Q_full = F_full;
	negative_Q_full.Resize_Array (N, false, false);
	S_full.Resize_Array (N, false, false);
	LOG::Stop_Time();
	LOG::Push_Scope ("CGI:", "CGI:");
	int iterations;
	T beta = 0;
	LOG::cout << "rho before cg iters: " << rho << " || supnorm before cg iters: " << supnorm << std::endl;
    for (iterations = 1; iterations <= max_iterations; iterations++)
	{
//        LOG::Time("CGI - Update solution I:");
		double S_dot_Q = 0.;
		//LOG::cout << "CG ITERATION " << iterations << "*******" << std::endl;
        #if defined USE_TASKS && !defined USE_HYBRID
            #pragma omp task inout(S_full) shared(F_full,particles) firstprivate(beta) label (CG-HELPER-I)
            {
                for (int p =1; p <= particles.particle_ranges->m; p++)
                {
                    #pragma omp task shared(F_full,particles) firstprivate(p,beta) label(CG-HELPER-I-nested)
                    CG_Helper_I(p,beta);
                }
                #pragma omp taskwait
            }
            for (int p = 1; p <= particles.particle_ranges->m; p++)
            {
                #pragma omp task concurrent(S_dot_Q) in(S_full) \
                shared(F_full,particles) \
                firstprivate(p) label(CG-HELPER-II)
                {
                    CG_Helper_II(p,S_dot_Q,time);
                }
            }
            //LOG::cout << "Alpha: " << alpha << " || rho: " << rho << " || S_dot_Q: " << S_dot_Q << std::endl;
            //LOG::Stop_Time();
            //LOG::Time("CGI - Update solution II:");
            double rho_new = 0;
            for (int p = 1; p <= particles.particle_ranges->m; p++)
            {
                #pragma omp task in(S_dot_Q) \
                shared(dX_full,R_full,F_full,S_full,rho_new,supnorm,particles) \
                firstprivate(p) label(CG-HELPER-III)
                {
                    CG_Helper_III(p,dX_full,rho_new,supnorm,rho,S_dot_Q);
                }
                
            }
        #elif defined USE_WORKSHARING_FOR || defined USE_HYBRID
            #pragma omp for schedule(static,1) label(CG-HELPER-I-FOR)
            for (int p = 1; p <= particles.particle_ranges->m; p++)
                CG_Helper_I(p,beta);
            #pragma omp for schedule(static,1) shared(S_dot_Q) label(CG-HELPER-II-FOR)
            for (int p = 1; p <= particles.particle_ranges->m; p++)
                CG_Helper_II(p,S_dot_Q,time);

            double rho_new = 0;

            #pragma omp for schedule(static,1) shared(dX_full,rho_new,supnorm,rho,S_dot_Q) label(CG-HELPER-III-FOR)
            for (int p = 1; p <= particles.particle_ranges->m; p++)
                CG_Helper_III(p,dX_full,rho_new,supnorm,rho,S_dot_Q);
        #endif
		#pragma omp taskwait
		//LOG::cout << "rho_new after Helper III: " << rho_new << " || supnorm " << supnorm << std::endl;

		supnorm = sqrt (supnorm);

//        LOG::Stop_Time();
		//LOG::cout << "cg supnorm before comparing against convergence: " << supnorm << std::endl;
		if (supnorm <= convergence_tolerance) {
			//LOG::cout << "cg supnorm converted after " << iterations << " Converged value:" << supnorm << std::endl;
			break;
		}

		beta = (T) (rho_new / rho);
		rho = rho_new;
	}

	if (iterations_used) *iterations_used = min (iterations, max_iterations);

	LOG::Pop_Scope(); // from CGI

	if (iterations <= max_iterations)
	{
		//LOG::cout << "cg iterations = " << iterations << std::endl;
		LOG::Pop_Scope(); // from NRS
		return true;
	}
	else
	{
        LOG::cout << "cg not converged after " << max_iterations << " iterations  - error = " << supnorm << std::endl;
		LOG::Pop_Scope(); // from NRS
		return false;
	}

	LOG::Pop_Scope();
	LOG::cout.unsetf(std::ios_base::floatfield);
}
/************OMPSS END*************/
#endif //OMPSS
#endif
//#####################################################################
// Function Update_Position_Based_State
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Update_Position_Based_State()
{
	LOG::Push_Scope ("UPBS:", "UPBS:");

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_position_based_state) solids_forces (k)->Update_Position_Based_State();
	LOG::Pop_Scope();
}
//#####################################################################
// Function Delete_Position_Based_State
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Delete_Position_Based_State()
{
	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_position_based_state) solids_forces (k)->Delete_Position_Based_State();
}
//#####################################################################
// Function Add_Velocity_Independent_Forces
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Add_Velocity_Independent_Forces (ARRAY<TV>& F) const
{
	LOG::Push_Scope ("AVIF:", "AVIF:");

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_velocity_independent_forces) solids_forces (k)->Add_Velocity_Independent_Forces (F);

	LOG::Pop_Scope();
}
//#####################################################################
// Function Add_Velocity_Dependent_Forces
//#####################################################################
// can depend on position too
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Add_Velocity_Dependent_Forces (ARRAY<TV>& F) const
{
	LOG::Push_Scope ("AVDF:", "AVDF:");

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_velocity_dependent_forces) solids_forces (k)->Add_Velocity_Dependent_Forces (F);

	LOG::Pop_Scope();
}
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Add_Velocity_Dependent_Forces (ARRAY<TV>& F, const int partition_id) const
{
	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_velocity_dependent_forces) solids_forces (k)->Add_Velocity_Dependent_Forces (F, partition_id);
}
//#####################################################################
// Function Force_Differential
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Force_Differential (const ARRAY<TV>& dX, ARRAY<TV>& dF) const
{
	//LOG::Push_Scope("AFD:","AFD:");
	//LOG::Time("AFD - Initialize:");
	dF.Resize_Array (particles.number);
	ARRAY_PARALLEL_OPERATIONS<TV, T, TV>::Clear_Parallel (dF, *particles.particle_ranges);
	//LOG::Stop_Time();
	for (int k = 1; k <= solids_forces.m; k++) solids_forces (k)->Add_Force_Differential (dX, dF);

	//LOG::Pop_Scope();
}
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Force_Differential (const ARRAY<TV>& dX, ARRAY<TV>& dF, const int partition_id) const
{
	assert (dF.m == particles.number);
	ARRAY<TV>::copy (TV(), dF.Range ( (*particles.particle_ranges) (partition_id)));

	for (int k = 1; k <= solids_forces.m; k++) solids_forces (k)->Add_Force_Differential (dX, dF, partition_id);
}
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Force_Differential_Internal (const ARRAY<TV>& dX, ARRAY<TV>& dF, const int partition_id) const
{
	assert (dF.m == particles.number);
	ARRAY<TV>::copy (TV(), dF.Range ( (*particles.particle_ranges) (partition_id)));

	for (int k = 1; k <= solids_forces.m; k++) solids_forces (k)->Add_Force_Differential_Internal (dX, dF, partition_id);
}

template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Force_Differential_External (const ARRAY<TV>& dX, ARRAY<TV>& dF, const int partition_id) const
{
	assert (dF.m == particles.number);
	for (int k = 1; k <= solids_forces.m; k++) solids_forces (k)->Add_Force_Differential_External (dX, dF, partition_id);
}
//#####################################################################
// Function Enforce_Definiteness
//#####################################################################
template<class T, class TV> void DEFORMABLE_OBJECT<T, TV>::
Enforce_Definiteness (const bool enforce_definiteness_input)
{
	for (int i = 1; i <= solids_forces.m; i++) solids_forces (i)->Enforce_Definiteness (enforce_definiteness_input);
}
//#####################################################################
// Function CFL
//#####################################################################
template<class T, class TV> T DEFORMABLE_OBJECT<T, TV>::
CFL (const bool verbose)
{
	T dt_elastic_and_damping = CFL_Elastic_And_Damping(), dt_strain_rate = CFL_Strain_Rate();

	if (verbose)
	{
		LOG::cout << "dt_elastic_and_damping = " << dt_elastic_and_damping << std::endl;
		LOG::cout << "dt_strain_rate = " << dt_strain_rate << std::endl;
		LOG::cout << "min = " << min (dt_elastic_and_damping, dt_strain_rate) << std::endl;
	}

	return min (dt_elastic_and_damping, dt_strain_rate);
}
//#####################################################################
// Function CFL_Elastic_And_Damping
//#####################################################################
template<class T, class TV> T DEFORMABLE_OBJECT<T, TV>::
CFL_Elastic_And_Damping()
{
	T dt_elastic = CFL_Elastic();
	T dt_damping = FLT_MAX;

	if (!implicit_damping) dt_damping = CFL_Damping();

	T one_over_dt_full = (1 / dt_elastic + 1 / dt_damping) / cfl_number;

	if (one_over_dt_full  > 1 / FLT_MAX) return 1 / one_over_dt_full;
	else return FLT_MAX;
}
//#####################################################################
// Function CFL_Elastic
//#####################################################################
template<class T, class TV> T DEFORMABLE_OBJECT<T, TV>::
CFL_Elastic()
{
	T hertz = 0;

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_velocity_independent_forces) hertz += 1 / solids_forces (k)->CFL_Velocity_Independent();

	if (hertz > 1 / FLT_MAX) return 1 / hertz;
	else return FLT_MAX;
}
//#####################################################################
// Function CFL_Damping
//#####################################################################
template<class T, class TV> T DEFORMABLE_OBJECT<T, TV>::
CFL_Damping()
{
	T hertz = 0;

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->use_velocity_dependent_forces) hertz += 1 / solids_forces (k)->CFL_Velocity_Dependent();

	if (hertz > 1 / FLT_MAX) return 1 / hertz;
	else return FLT_MAX;
}
//#####################################################################
// Function CFL_Strain_Rate
//#####################################################################
template<class T, class TV> T DEFORMABLE_OBJECT<T, TV>::
CFL_Strain_Rate()
{
	T hertz = 0;

	for (int k = 1; k <= solids_forces.m; k++) if (solids_forces (k)->limit_time_step_by_strain_rate) hertz = max (hertz, 1 / solids_forces (k)->CFL_Strain_Rate()); // otherwise not included

	if (hertz > 1 / FLT_MAX) return 1 / hertz;
	else return FLT_MAX;
}
//#####################################################################


template class DEFORMABLE_OBJECT<float, VECTOR_3D<float> >;
template class DEFORMABLE_OBJECT<double, VECTOR_3D<double> >;
template class DEFORMABLE_OBJECT<float, VECTOR_2D<float> >;
template class DEFORMABLE_OBJECT<double, VECTOR_2D<double> >;

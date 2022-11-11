//#####################################################################
// Copyright 2004-2005, Ron Fedkiw, Joseph Teran, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
// OmpSs/OpenMP 4.0 versions by Raul Vidal Ortiz - Barcelona Supercomputing Center
//#####################################################################
// Class COLLISION_PENALTY_FORCES
//#####################################################################
#include "COLLISION_PENALTY_FORCES.h"

using namespace PhysBAM;

template<class T> void COLLISION_PENALTY_FORCES<T>::
Update_Forces_And_Derivatives_Helper(int p)
{
    #if defined ENABLE_OMPEXTRAE 
        Extrae_user_function(1);
    #endif
    int index = check_collision (p);
    collision_force (p) = VECTOR_3D<T>();
    collision_force_derivative (p) = SYMMETRIC_MATRIX_3X3<T>();

    for (int r = 1; r <= collision_body_list->collision_bodies.m; r++) if (!skip_collision_body (r))
    {
        int collision_body_particle_index = 0;

        if (collision_body_list_id == r) collision_body_particle_index = index;

        T phi_value;
        int aggregate = -1;
        VECTOR_3D<T> normal = collision_body_list->collision_bodies (r)->Implicit_Surface_Extended_Normal (particles.X (index), phi_value, aggregate, collision_body_particle_index);

        if (phi_value <= 0)
        {
            collision_force (p) += stiffness * (-phi_value + separation_parameter) * normal;

            if (collision_body_list_id == r) collision_force_derivative (p) -= self_collision_reciprocity_factor * stiffness * SYMMETRIC_MATRIX_3X3<T>::Outer_Product (normal);
            else collision_force_derivative (p) -= stiffness * SYMMETRIC_MATRIX_3X3<T>::Outer_Product (normal);
        }
        else if (phi_value < collision_body_list->collision_bodies (r)->collision_thickness)
        {
            collision_force (p) += stiffness * separation_parameter * (T) exp (-phi_value / separation_parameter) * normal;

            if (collision_body_list_id == r)
                collision_force_derivative (p) -= self_collision_reciprocity_factor * stiffness * (T) exp (-phi_value / separation_parameter) * SYMMETRIC_MATRIX_3X3<T>::Outer_Product (normal);
            else collision_force_derivative (p) -= stiffness * (T) exp (-phi_value / separation_parameter) * SYMMETRIC_MATRIX_3X3<T>::Outer_Product (normal);
        }
    }
    #if defined ENABLE_OMPEXTRAE 
        Extrae_user_function(0);
    #endif
}
#ifdef ENABLE_OPENMP
/*************************************************************************
**************************************************************************
**************************************************************************
                OPENMP IMPLEMENTATION
**************************************************************************
**************************************************************************
**************************************************************************/
    template<class T> void COLLISION_PENALTY_FORCES<T>::
    Update_Forces_And_Derivatives()
    {
        OMPSS_POOL& pool = *OMPSS_POOL::Singleton();
        unsigned long ndivs = pool.Get_n_divisions(); unsigned long BSIZE = check_collision.m/ndivs;
        #ifdef USE_TASKS
            for (unsigned long i = 0; i < ndivs; i++)
            {
                int START = i*BSIZE+1; int END = START+BSIZE; if (i == ndivs-1) END = check_collision.m;
                int* check_collision_bp = check_collision.base_pointer; VECTOR_3D<T>* collision_force_bp = collision_force.base_pointer;
                SYMMETRIC_MATRIX_3X3<T>* collision_force_derivative_bp = collision_force_derivative.base_pointer;
                ARRAY<int>* check_collision_pointer = &check_collision; ARRAY<VECTOR_3D<T> >* collision_force_pointer = &collision_force;
                ARRAY<SYMMETRIC_MATRIX_3X3<T> > * collision_force_derivative_pointer = &collision_force_derivative;
                #pragma omp task depend(in:check_collision_pointer[0:1],collision_force_pointer[0:1],collision_force_derivative_pointer[0:1])\
                                depend(inout:check_collision_bp[START:END-START+1],collision_force_bp[START:END-START+1],collision_force_derivative_bp[START:END-START+1]) \
                                firstprivate(START,END) 
                {
                    for (int p = START; p <= END; p++)
                        Update_Forces_And_Derivatives_Helper(p);
                }
            }
            //#pragma omp taskwait
        #elif defined USE_WORKSHARING_FOR
            #pragma omp parallel for schedule(static,BSIZE)
            for (int p = 1; p <= check_collision.m; p++)
                Update_Forces_And_Derivatives_Helper(p);
        #endif
    }
    
    template<class T> void COLLISION_PENALTY_FORCES<T>::
    Add_Velocity_Independent_Forces (ARRAY<VECTOR_3D<T> >& F) const
	{
		LOG::Time ("AVIF (CPF):");
        #ifdef USE_TASKS
            VECTOR_3D<T>* Fbp = F.base_pointer; long size = F.m; const ARRAY<int>& ref_check_collision = check_collision;
            long collision_size = check_collision.m;
            ARRAY<int>* check_collision_pointer = &check_collision; ARRAY<VECTOR_3D<T> >* collision_force_pointer = &collision_force;
            ARRAY<SYMMETRIC_MATRIX_3X3<T> > * collision_force_derivative_pointer = &collision_force_derivative;
            const ARRAY<VECTOR_3D<T> >& ref_collision_force = collision_force; ARRAY<VECTOR_3D<T> >* Fp = &F;
            int* check_collision_bp = check_collision.base_pointer; VECTOR_3D<T>* collision_force_bp = collision_force.base_pointer;
            SYMMETRIC_MATRIX_3X3<T>* collision_force_derivative_bp = collision_force_derivative.base_pointer;
            #pragma omp task depend(inout:Fp[0:1],Fbp[:size]) depend(in:check_collision_pointer[0:1],\
                                    collision_force_pointer[0:1],check_collision_bp[:collision_size],collision_force_bp[:collision_size])
                                    //shared(ref_collision_force,F,Fbp,ref_check_collision)
            {
                #ifdef ENABLE_OMPEXTRAE
                Extrae_user_function(1);
                #endif
                for (int p = 1; p <= check_collision.m; p++) (*Fp) (check_collision (p)) += collision_force (p);
                #ifdef ENABLE_OMPEXTRAE
                Extrae_user_function(0);
                #endif
            }
        #else 
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int p = 1; p <= check_collision.m; p++) F (check_collision (p)) += collision_force (p);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        #endif
		LOG::Stop_Time();
	}

    template<class T> void COLLISION_PENALTY_FORCES<T>::
	Add_Force_Differential (const ARRAY<VECTOR_3D<T> >& dX, ARRAY<VECTOR_3D<T> >& dF) const
	{
		//LOG::Time("AFD (CPF):");
        #ifdef USE_TASKS
            VECTOR_3D<T>* dXbp = dX.base_pointer; VECTOR_3D<T>* dFbp = dF.base_pointer; long size = dX.m; const ARRAY<int>& ref_check_collision = check_collision;
            ARRAY<VECTOR_3D<T> >* dXp = &dX; ARRAY<VECTOR_3D<T> >* dFp = &dF; long collision_size = check_collision.m;
            const ARRAY<SYMMETRIC_MATRIX_3X3<T> >& ref_collision_force_derivative = collision_force_derivative;
            ARRAY<int>* check_collision_pointer = &check_collision; ARRAY<VECTOR_3D<T> >* collision_force_pointer = &collision_force;
            ARRAY<SYMMETRIC_MATRIX_3X3<T> > * collision_force_derivative_pointer = &collision_force_derivative;
            int* check_collision_bp = check_collision.base_pointer; VECTOR_3D<T>* collision_force_bp = collision_force.base_pointer;
            SYMMETRIC_MATRIX_3X3<T>* collision_force_derivative_bp = collision_force_derivative.base_pointer;
            #pragma omp task depend(in:dXp[0:1],dXbp[:size]) depend(inout:dFp[0:1],dFbp[:size],check_collision_pointer[0:1],collision_force_derivative_pointer[0:1],\
                                    check_collision_bp[:collision_size],\
                                    collision_force_derivative_bp[:collision_size]) //firstprivate(dX,dXbp,dF,dFbp,ref_collision_force_derivative,ref_check_collision)
            {
                #ifdef ENABLE_OMPEXTRAE
                Extrae_user_function(1);
                #endif
                for (int p = 1; p <= check_collision.m; p++) (*dFp) (check_collision (p)) += collision_force_derivative (p) * (*dXp) (check_collision (p));
                #ifdef ENABLE_OMPEXTRAE
                Extrae_user_function(0);
                #endif
            }
        #else
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int p = 1; p <= check_collision.m; p++) dF (check_collision (p)) += collision_force_derivative (p) * dX (check_collision (p));
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        #endif
		//LOG::Stop_Time();
	}
#elif defined ENABLE_OMPSS
/*************************************************************************
**************************************************************************
**************************************************************************
                OMPSS IMPLEMENTATION
**************************************************************************
**************************************************************************
**************************************************************************/
    template<class T> void COLLISION_PENALTY_FORCES<T>::
    Update_Forces_And_Derivatives()
    {
        OMPSS_POOL& pool = *OMPSS_POOL::Singleton();
        unsigned long ndivs = pool.Get_n_divisions(); unsigned long BSIZE = check_collision.m/ndivs;
        #ifdef USE_TASKS
            for (unsigned long i = 0; i < ndivs; i++)
            {
                int START = i*BSIZE+1; int END = START+BSIZE; if (i == ndivs-1) END = check_collision.m;
                int* check_collision_bp = check_collision.base_pointer; VECTOR_3D<T>* collision_force_bp = collision_force.base_pointer;
                SYMMETRIC_MATRIX_3X3<T>* collision_force_derivative_bp = collision_force_derivative.base_pointer;
                #pragma omp task concurrent(check_collision,collision_force,collision_force_derivative) \
                                concurrent(collision_force_derivative_bp,collision_force_bp,check_collision_bp) \
                                firstprivate(START,END) label(UCPF)
                {
                    for (int p = START; p <= END; p++)
                        Update_Forces_And_Derivatives_Helper(p);
                }
            }
            //#pragma omp taskwait
        #elif defined USE_WORKSHARING_FOR
            #pragma omp for schedule(static,BSIZE) label(UCPF-FOR)
            for (int p = 1; p <= check_collision.m; p++)
                Update_Forces_And_Derivatives_Helper(p);
        #endif
    }
    
    template<class T> void COLLISION_PENALTY_FORCES<T>::
	Add_Velocity_Independent_Forces (ARRAY<VECTOR_3D<T> >& F) const
	{
		LOG::Time ("AVIF (CPF):");
		#ifdef USE_TASKS
            #pragma omp task inout(F) in(collision_force,check_collision) label (AVIF-CPF)
            for (int p = 1; p <= check_collision.m; p++) F (check_collision (p)) += collision_force (p);
        #else
            for (int p = 1; p <= check_collision.m; p++) F (check_collision (p)) += collision_force (p);
		#endif
        LOG::Stop_Time();
	}

    template<class T> void COLLISION_PENALTY_FORCES<T>::
	Add_Force_Differential (const ARRAY<VECTOR_3D<T> >& dX, ARRAY<VECTOR_3D<T> >& dF) const
	{
		//LOG::Time("AFD (CPF):");
        #ifdef USE_TASKS
            #pragma omp task inout(dF) in(dX,collision_force_derivative,check_collision) label(AFD-CPF)
            for (int p = 1; p <= check_collision.m; p++) dF (check_collision (p)) += collision_force_derivative (p) * dX (check_collision (p));
		#else
            for (int p = 1; p <= check_collision.m; p++) dF (check_collision (p)) += collision_force_derivative (p) * dX (check_collision (p));
        #endif
        //LOG::Stop_Time();
	}
#endif
template class COLLISION_PENALTY_FORCES<float>;
template class COLLISION_PENALTY_FORCES<double>;

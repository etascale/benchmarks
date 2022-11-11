//#####################################################################
// Copyright 2005-2006, Andrew Selle, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//   OPENMP - OMPSS
// Copyright 2014-2015 Raul Vidal - Barcelona Supercomputing Center
//#####################################################################
#include "../Arrays/ARRAY.h"
#include "../Matrices_And_Vectors/VECTOR_2D.h"
#include "OMPSS_ARRAY_PARALLEL_OPERATIONS.h"
#include "../Matrices_And_Vectors/VECTOR_3D.h"
//Order between <<includes>> of SYMMETRIC_MATRIX and MATRIX_3X3 changed
#include "../Matrices_And_Vectors/SYMMETRIC_MATRIX_3X3.h"
#include "../Matrices_And_Vectors/MATRIX_3X3.h"
#include "../Math_Tools/max.h"
#include "../Thread_Utilities/OMPSS_POOL.h"

#ifdef ENABLE_OMPEXTRAE
	#include "../Utilities/EXTRAE.h"
#endif
using namespace PhysBAM;
#ifdef ENABLE_OPENMP
//#####################################################################
// ARRAY_PARALLEL_OPERATIONS OPENMP IMPLEMENTATION
//#####################################################################

//Currently shared variables do not have copy directionality, this gives a warning.
//Because Mercurium does not instatiate parametric types, we had to hide those types
//to the compiler using a typedef, so we could define an array section

//#####################################################################
// Function Clear_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Clear_Parallel (ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        ARRAY<T>* array_output_pointer = &array_output; T* bp_output = array_output.base_pointer; long SIZE = array_output.m;
        ARRAY<VECTOR_2D<int> >* ranges_pointer = &ranges;
        #pragma omp task depend(inout:array_output_pointer,bp_output[:SIZE]) firstprivate(ranges_pointer)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            ARRAY<VECTOR_2D<int> >& ranges = *ranges_pointer; ARRAY<T>& array_output = *array_output_pointer;
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output(j) = T();
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        //#pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = T();
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Copy_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Copy_Array_Parallel (const ARRAY<T>& array_input_1, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1) shared(array_output,array_input_1,ranges)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) = array_input_1 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = array_input_1 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Scaled_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1,array_input_2) shared(array_output,array_input_1,array_input_2,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) = scalar_element_input * array_input_1 (j) + array_input_2 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = scalar_element_input * array_input_1 (j) + array_input_2 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Scaled_Normalized_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Normalized_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const ARRAY<TS>& scalar_array_input_3, const TS scalar_element_input,
		ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1,array_input_2,scalar_array_input_3) shared(array_output,array_input_1,array_input_2,scalar_array_input_3,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) = (scalar_element_input / scalar_array_input_3 (j)) * array_input_1 (j) + array_input_2 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = (scalar_element_input / scalar_array_input_3 (j)) * array_input_1 (j) + array_input_2 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1,array_input_2) shared(array_output,array_input_1,array_input_2,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j) + array_input_2 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j) + array_input_2 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Array_Parallel (const ARRAY<T>& array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1) shared(array_output,array_input_1,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Normalized_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Normalized_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<TS>& scalar_array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:array_input_1,scalar_array_input_2) shared(array_output,array_input_1,scalar_array_input_2,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) += (scalar_element_input / scalar_array_input_2 (j)) * array_input_1 (j);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += (scalar_element_input / scalar_array_input_2 (j)) * array_input_1 (j);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Scale_Normalize_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scale_Normalize_Array_Parallel (const ARRAY<TS>& scalar_array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:array_output) depend(in:scalar_array_input_1) shared(array_output,scalar_array_input_1,ranges)\
        firstprivate(scalar_element_input)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) array_output (j) *= (scalar_element_input / scalar_array_input_1 (j));
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) firstprivate(scalar_element_input)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) *= (scalar_element_input / scalar_array_input_1 (j));
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}

//#####################################################################
//                  OPENMP IMPLEMENTATION OF THE LAST FOUR FUNCTIONS
//#####################################################################

//#####################################################################
// Function Dot_Product_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:result) depend(in:vector_array_input_1,vector_array_input_2) shared(result,vector_array_input_1,vector_array_input_2,ranges)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) result += (double) TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) shared(result)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
            for (int j = START; j <= END; j++) a += (double) TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            //#pragma omp atomic
            result +=a;
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Scaled_Dot_Product_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<TS>& scalar_array_input_3, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:result) depend(in:vector_array_input_1,vector_array_input_2,scalar_array_input_3) \
                            shared(result,vector_array_input_1,vector_array_input_2,scalar_array_input_3,ranges)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y;
                for (int j = START; j <= END; j++) result += (double) scalar_array_input_3 (j) * TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp parallel for schedule(static,1) shared(result)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
			for (int j = START; j <= END; j++) a += (double) scalar_array_input_3 (j) * TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            //#pragma omp atomic
            result += a;
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Maximum_Magnitude_Squared_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Maximum_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:result) depend(in:vector_array_input_1) \
                            shared(result,vector_array_input_1,ranges)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
                for (int j = START; j <= END; j++) a = max<double> (a, vector_array_input_1 (j).Magnitude_Squared());
                #pragma omp critical(APO_MMSP)
                result = max<double> (result, a);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp declare reduction(my_max: double : omp_out = omp_out < omp_in ? omp_in : omp_out) initializer(omp_priv = 0.) 
        //#pragma omp parallel for schedule(static,1) reduction(my_max:result)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
			for (int j = START; j <= END; j++) a = max<double> (a, vector_array_input_1 (j).Magnitude_Squared());
			//#pragma omp critical(APO_MMSP)
			result = max<double> (result, a);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
//#####################################################################
// Function Maximum_Scaled_Magnitude_Squared_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Maximum_Scaled_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TS>& scalar_array_input_2, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        #pragma omp task depend(inout:result) depend(in:vector_array_input_1,scalar_array_input_2) \
                            shared(result,vector_array_input_1,scalar_array_input_2,ranges)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            for (int i = 1; i <= ranges.m; i++)
            {
                int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
                for (int j = START; j <= END; j++) a = max<double> (a, scalar_array_input_2 (j) * vector_array_input_1 (j).Magnitude_Squared());
                #pragma omp critical(APO_MSMSP)
                result = max<double> (result, a);
            }
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp declare reduction(my_max: double : omp_out = omp_out < omp_in ? omp_in : omp_out) initializer(omp_priv = 0.)
        //#pragma omp parallel for schedule(static,1) reduction(my_max:result)
        for (int i = 1; i <= ranges.m; i++)
        {
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(1);
            #endif
            int START = ranges(i).x; int END = ranges(i).y; double a = 0.0;
			for (int j = START; j <= END; j++) a = max<double> (a, scalar_array_input_2 (j) * vector_array_input_1 (j).Magnitude_Squared());
			//#pragma omp critical(APO_MSMSP)
			result = max<double> (result, a);
            #ifdef ENABLE_OMPEXTRAE
            Extrae_user_function(0);
            #endif
        }
    #endif
}
#elif defined ENABLE_OMPSS //OMPSS
//#####################################################################
// ARRAY_PARALLEL_OPERATIONS OMPSS IMPLEMENTATION
//#####################################################################

//Currently shared variables do not have copy directionality, this gives a warning.
//Because Mercurium does not instatiate parametric types, we had to hide those types
//to the compiler using a typedef, so we could define an array section

//#####################################################################
// Function Clear_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Clear_Parallel (ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
	#ifdef USE_TASKS 
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start = array_output(START);
            #pragma omp task inout(start)\
             concurrent(array_output) firstprivate(START,END) label(APO-CP)
            for (int j = START; j <= END; j++) array_output (j) = T();
            
        }
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output) label(APO-CP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = T();
        }
    #endif
	//#pragma omp taskwait
}
//#####################################################################
// Function Copy_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Copy_Array_Parallel (const ARRAY<T>& array_input_1, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START);
            #pragma omp task inout(start_output) in (start_input1)\
             concurrent(array_output,array_input_1) firstprivate(START,END) label(APO-CAP)
            for (int j = START; j <= END; j++) array_output (j) = array_input_1 (j);
            
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1) label(APO-CAP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = array_input_1 (j);
        }
    #endif
}
//#####################################################################
// Function Scaled_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START); const T& start_input2 = array_input_2(START);
            #pragma omp task inout(start_output) in (start_input1,start_input2,scalar_element_input)\
             concurrent(array_output,array_input_1,array_input_2) firstprivate(START,END) label(APO-SAPP)
            for (int j = START; j <= END; j++) array_output (j) = scalar_element_input * array_input_1 (j) + array_input_2 (j);
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1,array_input_2,scalar_element_input) label(APO-SAPP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = scalar_element_input * array_input_1 (j) + array_input_2 (j);
        }
    #endif
}
//#####################################################################
// Function Scaled_Normalized_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Normalized_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const ARRAY<TS>& scalar_array_input_3, const TS scalar_element_input,
		ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START); const T& start_input2 = array_input_2(START); const TS& start_input3 = scalar_array_input_3(START);
            #pragma omp task inout(start_output) in (start_input1,start_input2,start_input3,scalar_element_input)\
             concurrent(array_output,array_input_1,array_input_2,scalar_array_input_3) firstprivate(START,END) label(APO-SNAPP)
            for (int j = START; j <= END; j++) array_output (j) = (scalar_element_input / scalar_array_input_3 (j)) * array_input_1 (j) + array_input_2 (j);
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1,array_input_2,scalar_array_input_3,scalar_element_input) label(APO-SNAPP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) = (scalar_element_input / scalar_array_input_3 (j)) * array_input_1 (j) + array_input_2 (j);
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Array_Plus_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START); const T& start_input2 = array_input_2(START);
            #pragma omp task inout(start_output) in (start_input1,start_input2,scalar_element_input)\
             concurrent(array_output,array_input_1,array_input_2) firstprivate(START,END) label(APO-ASAPAP)
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j) + array_input_2 (j);
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1,array_input_2,scalar_element_input) label(APO-ASAPAP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j) + array_input_2 (j);
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Array_Parallel (const ARRAY<T>& array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START);
            #pragma omp task inout(start_output) in (start_input1,scalar_element_input)\
            concurrent(array_output,array_input_1) firstprivate(START,END) label(APO-ASAP)
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j);
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1,scalar_element_input) label(APO-ASAP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += scalar_element_input * array_input_1 (j);
        }
    #endif
}
//#####################################################################
// Function Add_Scaled_Normalized_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Add_Scaled_Normalized_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<TS>& scalar_array_input_2, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const T& start_input1 = array_input_1(START); const TS& start_input2 = scalar_array_input_2(START);
            #pragma omp task inout(start_output) in (start_input1, start_input2,scalar_element_input)\
             concurrent(array_output,scalar_array_input_2,array_input_1) firstprivate(START,END) label(APO-ASNAP)
            for (int j = START; j <= END; j++) array_output (j) += (scalar_element_input / scalar_array_input_2 (j)) * array_input_1 (j);
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,array_input_1,scalar_array_input_2,scalar_element_input) label(APO-ASNAP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) += (scalar_element_input / scalar_array_input_2 (j)) * array_input_1 (j);
        }
    #endif
}
//#####################################################################
// Function Scale_Normalize_Array_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scale_Normalize_Array_Parallel (const ARRAY<TS>& scalar_array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            T& start_output = array_output(START); const TS& start_input1 = scalar_array_input_1(START);
            #pragma omp task inout(start_output) in (start_input1,scalar_element_input)\
            concurrent(array_output,scalar_array_input_1) firstprivate(START,END) label(APO-SNAP)
            for (int j = START; j <= END; j++) array_output (j) *= (scalar_element_input / scalar_array_input_1 (j));
        }
        #pragma omp taskwait
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) shared(array_output,scalar_array_input_1,scalar_element_input) label(APO-SNAP-FOR)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            for (int j = START; j <= END; j++) array_output (j) *= (scalar_element_input / scalar_array_input_1 (j));
        }
    #endif
}

//#####################################################################
//                  OMPSS IMPLEMENTATION OF THE LAST FOUR FUNCTIONS
//#####################################################################

//#####################################################################
// Function Dot_Product_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            const TV& start_input1 = vector_array_input_1(START); const TV& start_input2 = vector_array_input_2(START);
            #pragma omp task in(start_input1,start_input2) concurrent(result)\
            concurrent(vector_array_input_1,vector_array_input_2) firstprivate(START,END) label(APO-DPP)
            {
                double a = 0.0;
                for (int j = START; j <= END; j++) a += (double) TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
                #pragma omp atomic
                result += a;
            } 
        }
        #pragma omp taskwait on (result)
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) label(APO-DPP-FOR) shared(result,vector_array_input_1,vector_array_input_2)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            double a = 0.0;
            for (int j = START; j <= END; j++) a += (double) TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            //#pragma omp atomic
            result += a;
        }
    #endif
}
//#####################################################################
// Function Scaled_Dot_Product_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Scaled_Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<TS>& scalar_array_input_3, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            const TV& start_input1 = vector_array_input_1(START); const TV& start_input2 = vector_array_input_2(START); const TS& start_input3 = scalar_array_input_3(START);
            #pragma omp task in(start_input1,start_input2,start_input3) concurrent(result)\
            concurrent(vector_array_input_1,vector_array_input_2,scalar_array_input_3) firstprivate(START,END) label(APO-SDPP)
            {
                double a = 0.0;
                for (int j = START; j <= END; j++) a += (double) scalar_array_input_3 (j) * TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
                #pragma omp atomic
                result += a;
            }
        }
        #pragma omp taskwait on (result)
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp for schedule(static,1) label(APO-SDPP-FOR) shared(result,vector_array_input_1,vector_array_input_2,scalar_array_input_3)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            double a = 0.0;
            for (int j = START; j <= END; j++) a += (double) scalar_array_input_3 (j) * TV::Dot_Product (vector_array_input_1 (j), vector_array_input_2 (j));
            //#pragma omp atomic
            result += a;
        }
    #endif
}
//#####################################################################
// Function Maximum_Magnitude_Squared_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Maximum_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            const TV& start_input1 = vector_array_input_1(START); 
            #pragma omp task in(start_input1) concurrent(result)\
            concurrent(vector_array_input_1) firstprivate(START,END) label(APO-MMSP)
            {
                double a = 0.0;
                for (int j = START; j <= END; j++) a = max<double> (a, vector_array_input_1 (j).Magnitude_Squared());
                #pragma omp critical(APO_MMSP)
                result = max<double> (result, a);
            }
        }
        #pragma omp taskwait on (result)
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp declare reduction(my_max: double : omp_out = omp_out < omp_in ? omp_in : omp_out) initializer(omp_priv = 0.)
        //#pragma omp for schedule(static,1) reduction(my_max:result) label(APO-MMSP-FOR) shared(vector_array_input_1)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            double a = 0.0;
            for (int j = START; j <= END; j++) a = max<double> (a, vector_array_input_1 (j).Magnitude_Squared());
            //#pragma omp critical(APO_MMSP_FOR)
            result = max<double> (result, a);
        }
    #endif
}
//#####################################################################
// Function Maximum_Scaled_Magnitude_Squared_Parallel
//#####################################################################
template<class T, class TS, class TV> void ARRAY_PARALLEL_OPERATIONS<T, TS, TV>::
Maximum_Scaled_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TS>& scalar_array_input_2, const ARRAY<VECTOR_2D<int> >& ranges, double& result)
{
    #ifdef USE_TASKS
        int START = 0; int END = 0;
        for (int i = 1; i <= ranges.m; i++)
        {
            START = ranges(i).x; END = ranges(i).y;
            const TV& start_input1 = vector_array_input_1(START); const TS& start_input2 = scalar_array_input_2(START);
            #pragma omp task in(start_input1, start_input2) concurrent(result)\
            concurrent(scalar_array_input_2,vector_array_input_1) firstprivate(START,END) label(APO-MSMSP)
            {
                double a = 0.0;
                for (int j = START; j <= END; j++) a = max<double> (a, scalar_array_input_2 (j) * vector_array_input_1 (j).Magnitude_Squared());
                #pragma omp critical(APO_MSMSP)
                result = max<double> (result, a);
            }
        }
        #pragma omp taskwait on (result)
    #elif defined USE_WORKSHARING_FOR
        //#pragma omp declare reduction(my_max: double : omp_out = omp_out < omp_in ? omp_in : omp_out) initializer(omp_priv = 0.)
        //#pragma omp for schedule(static,1) reduction(my_max:result) label(APO-MSMSP-FOR) shared(vector_array_input_1)
        for (int i = 1; i <= ranges.m; i++)
        {
            int START = ranges(i).x; int END = ranges(i).y;
            double a = 0.0;
            for (int j = START; j <= END; j++) a = max<double> (a, scalar_array_input_2 (j) * vector_array_input_1 (j).Magnitude_Squared());
            //#pragma omp critical(APO_MSMSP)
            result = max<double> (result, a);
        }
    #endif
}
#endif //OMPSS
template class ARRAY_PARALLEL_OPERATIONS<VECTOR_2D<float>, float, VECTOR_2D<float> >;
template class ARRAY_PARALLEL_OPERATIONS<VECTOR_3D<float>, float, VECTOR_3D<float> >;
template class ARRAY_PARALLEL_OPERATIONS<VECTOR_2D<double>, double, VECTOR_2D<double> >;
template class ARRAY_PARALLEL_OPERATIONS<VECTOR_3D<double>, double, VECTOR_3D<double> >;
template class ARRAY_PARALLEL_OPERATIONS<MATRIX_3X3<float>, float, VECTOR_3D<float> >;
template class ARRAY_PARALLEL_OPERATIONS<SYMMETRIC_MATRIX_3X3<float>, float, VECTOR_3D<float> >;
template class ARRAY_PARALLEL_OPERATIONS<MATRIX_3X3<double>, double, VECTOR_3D<double> >;
template class ARRAY_PARALLEL_OPERATIONS<SYMMETRIC_MATRIX_3X3<double>, double, VECTOR_3D<double> >;


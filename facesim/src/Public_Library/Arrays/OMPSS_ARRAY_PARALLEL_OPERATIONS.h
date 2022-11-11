//#####################################################################
// Copyright 2005-2006, Andrew Selle, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#ifndef __ARRAY_PARALLEL_OPERATIONS__
#define __ARRAY_PARALLEL_OPERATIONS__

namespace PhysBAM
{

template<class T> class ARRAY;
template<class T> class VECTOR_2D;

template<class T, class TS, class TV>
class ARRAY_PARALLEL_OPERATIONS
{
public:
	static void Clear_Parallel (ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges);
	static void Copy_Array_Parallel (const ARRAY<T>& array_input_1, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges);
	static void Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output,
			const ARRAY<VECTOR_2D<int> >& ranges);
	static void Scaled_Normalized_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const ARRAY<TS>& scalar_array_input_3,
			const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges);
	static void Add_Scaled_Array_Plus_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<T>& array_input_2, const TS scalar_element_input, ARRAY<T>& array_output,
			const ARRAY<VECTOR_2D<int> >& ranges);
	static void Add_Scaled_Array_Parallel (const ARRAY<T>& array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges);
	static void Add_Scaled_Normalized_Array_Parallel (const ARRAY<T>& array_input_1, const ARRAY<TS>& scalar_array_input_2, const TS scalar_element_input, ARRAY<T>& array_output,
			const ARRAY<VECTOR_2D<int> >& ranges);
	static void Scale_Normalize_Array_Parallel (const ARRAY<TS>& scalar_array_input_1, const TS scalar_element_input, ARRAY<T>& array_output, const ARRAY<VECTOR_2D<int> >& ranges);
	static void Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<VECTOR_2D<int> >& rangesm,double& result);
	static void Scaled_Dot_Product_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TV>& vector_array_input_2, const ARRAY<TS>& scalar_array_input_3, const ARRAY<VECTOR_2D<int> >& ranges,double& result);
	static void Maximum_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<VECTOR_2D<int> >& ranges,double& result);
	static void Maximum_Scaled_Magnitude_Squared_Parallel (const ARRAY<TV>& vector_array_input_1, const ARRAY<TS>& scalar_array_input_2, const ARRAY<VECTOR_2D<int> >& ranges,double& result);
//#####################################################################
};
}
#endif

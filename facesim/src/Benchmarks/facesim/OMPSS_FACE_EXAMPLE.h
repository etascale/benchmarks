//#####################################################################
// Copyright 2004, Igor Neverov, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
// OmpSs/OpenMP 4.0 versions by Raul Vidal Ortiz - Barcelona Supercomputing Center
//#####################################################################
// Class FACE_EXAMPLE
//#####################################################################
#ifndef __FACE_EXAMPLE__
#define __FACE_EXAMPLE__

#include "QUASISTATICS_EXAMPLE.h"
#include "../../Public_Library/Collisions_And_Interactions/TETRAHEDRON_COLLISION_BODY.h"
#include "FACE_OPTIMIZATION.h"
#include "FACE_OPTIMIZATION_GOAL.h"
#include "ACTIVATION_CONTROL_SET.h"
#include "ATTACHMENT_FRAME_CONTROL_SET.h"
#include "LANDMARK_3D.h"
#include "../../Public_Library/Utilities/LOG.h"
#include "../../Public_Library/Thread_Utilities/OMPSS_POOL.h"
#ifdef ENABLE_OMPEXTRAE
	#include "../../Public_Library/Utilities/EXTRAE.h"
#endif

namespace PhysBAM
{

template <class T, class RW>
class FACE_EXAMPLE: public QUASISTATICS_EXAMPLE<T, RW>
{
public:
	using SOLIDS_FLUIDS_EXAMPLE<T>::output_directory;
	using SOLIDS_FLUIDS_EXAMPLE<T>::verbose;
	using SOLIDS_FLUIDS_EXAMPLE_3D<T, RW>::solids_parameters;
	FACE_CONTROL_PARAMETERS<T> control_parameters;
	FACE_OPTIMIZATION<T>* optimization;
	FACE_OPTIMIZATION_GOAL<T>* optimization_goal;
	int restart_step, write_last_step;
	int static_target_frame;

	std::string input_directory, model_directory;
	int number_of_muscles, number_of_attachments;
	LIST_ARRAY<LIST_ARRAY<int> > muscle_tets;
	LIST_ARRAY<LIST_ARRAY<VECTOR_3D<T> > > muscle_fibers;
	LIST_ARRAY<LIST_ARRAY<T> > muscle_densities;
	LIST_ARRAY<std::string> muscle_names;
	LIST_ARRAY<T> peak_isometric_stress;
	LIST_ARRAY<LIST_ARRAY<int> > attached_nodes;
	ARRAY<LIST_ARRAY<int> >* attached_nodes_parallel;
	int jaw_attachment_index;

	ACTIVATION_CONTROL_SET<T> *activation_controls;
	ATTACHMENT_FRAME_CONTROL_SET<T> *attachment_frame_controls;

	bool use_rigid_body_collision, use_self_collision;
	DIAGONALIZED_CONSTITUTIVE_MODEL_3D<T> *face_constitutive_model;
	T peak_isometric_stress_boosting;

	FACE_EXAMPLE()
		: optimization (0), optimization_goal (0), restart_step (0), write_last_step (true), static_target_frame (-1), peak_isometric_stress_boosting ( (T) 1), attached_nodes_parallel (0)
	{}

	void Write_Output_Files (const int frame) const
	{
		QUASISTATICS_EXAMPLE<T, RW>::Write_Output_Files (frame);

		if (optimization) optimization->template Write_Optimization_Data<RW> (output_directory + "/", frame);
		else Write_Controls (output_directory + "/", frame);
        //Write triangles of the face
        const DEFORMABLE_OBJECT_3D<T>& deformable_object = solids_parameters.deformable_body_parameters.list (1);
        const ARRAY<VECTOR_3D<T> >& vertices = deformable_object.particles.X.array;
        const TETRAHEDRON_MESH& tetrahedron_mesh = deformable_object.tetrahedralized_volume->tetrahedron_mesh;
        /*All triangles have the vertices in the same order, supposedly they go counter clock-wise, what OpenGL needs 
        for the normals*/
        std::ostream *output;
        std::string output_file;
        std::ostringstream ostrframe;
        ostrframe << frame;
        #ifdef ENABLE_OPENMP
        output_file = output_directory + "/openmp_final_boundary_mesh_" + ostrframe.str() + ".obj";
        #elif defined ENABLE_OMPSS
        output_file = output_directory + "/ompss_final_boundary_mesh_" + ostrframe.str() + ".obj";
        #endif
        output = FILE_UTILITIES::Safe_Open_Output (output_file, false);
        for (int i = 1; i <= vertices.m; i++) (*output) << "v " << vertices(i) << std::endl;
        (*output) << "usemtl (null)" << std::endl;
        for (int i = 1; i<= tetrahedron_mesh.boundary_mesh->triangles.m; i++) {
                (*output) << "f ";
                for (int j = 1; j <= 3; j++) {
                        (*output) << tetrahedron_mesh.boundary_mesh->triangles(j,i) << " ";
                }
                (*output) << std::endl;
        }
        delete output;
	}

	void Read_Output_Files_Solids (const int frame)
	{
		QUASISTATICS_EXAMPLE<T, RW>::Read_Output_Files_Solids (frame);

		if (optimization) optimization->template Read_Optimization_Data<RW> (output_directory + "/", frame);
		else Read_Controls (output_directory + "/", frame);
	}

	void Write_Controls (const std::string& prefix, const int frame) const
	{
		std::string f = STRING_UTILITIES::string_sprintf (".%d", frame);
		FILE_UTILITIES::Write_To_File<RW> (prefix + "controls" + f, control_parameters);
	}

	void Read_Controls (const std::string& prefix, const int frame)
	{
		std::string f = STRING_UTILITIES::string_sprintf (".%d", frame);
		FILE_UTILITIES::Read_From_File<RW> (prefix + "controls" + f, control_parameters);
	}

	void Default() const
	{
		std::cout << "THIS FACE_EXAMPLE FUNCTION IS NOT DEFINED!" << std::endl;
	}

	virtual void Set_External_Controls (const T time) {}

	virtual void Preprocess_Frame (const int frame) {}
	virtual void Postprocess_Frame (const int frame) {}
	virtual void Postprocess_Solids_Substep (const T time, const int substep) {}

//#####################################################################
// Function Initialize_Bodies
//#####################################################################
	void Initialize_Bodies()
	{
		Get_Initial_Data();

		if (use_self_collision) Initialize_Tetrahedron_Collisions();

		QUASISTATICS_EXAMPLE<T, RW>::Initialize_Bodies();

		DEFORMABLE_OBJECT_3D<T>& deformable_object = solids_parameters.deformable_body_parameters.list (1);
		activation_controls = new ACTIVATION_CONTROL_SET<T>;

		for (int i = 1; i <= number_of_muscles; i++) activation_controls->Add_Activation();

		control_parameters.list.Append_Element (activation_controls);
		attachment_frame_controls = new ATTACHMENT_FRAME_CONTROL_SET<T> (deformable_object.particles.X.array, attached_nodes, jaw_attachment_index);
		attachment_frame_controls->template Initialize_Jaw_Joint_From_File<RW> (input_directory + "/jaw_joint_parameters");
		control_parameters.list.Append_Element (attachment_frame_controls);

		if (use_rigid_body_collision)
		{
			assert (attachment_frame_controls);
			attachment_frame_controls->Set_Original_Attachment_Configuration (solids_parameters.rigid_body_parameters.list.rigid_bodies (1)->Frame(),
					solids_parameters.rigid_body_parameters.list.rigid_bodies (2)->Frame());
		}

		deformable_object.Add_Quasistatic_Diagonalized_Face (*deformable_object.tetrahedralized_volume, muscle_tets, muscle_fibers, muscle_densities,
				activation_controls->activations, &activation_controls->single_activation_used_for_force_derivative, &peak_isometric_stress, &face_constitutive_model, (T) 6e4, (T) 2e4, (T) 3e6, (T).1);
		activation_controls->muscle_force = deformable_object.diagonalized_finite_volume_3d (1);
		attachment_frame_controls->muscle_force = deformable_object.diagonalized_finite_volume_3d (1);

		// Collision settings
		if (solids_parameters.perform_collision_body_collisions)
		{
			COLLISION_PENALTY_FORCES<T> *penalty_force = new COLLISION_PENALTY_FORCES<T> (solids_parameters.deformable_body_parameters.list.deformable_objects (1)->tetrahedralized_volume->particles);
			penalty_force->Set_Stiffness ( (T) 1e3);
			penalty_force->Set_Separation_Parameter ( (T) 1e-4);
			penalty_force->Set_Collision_Body_List (solids_parameters.collision_body_list);

			if (use_self_collision) penalty_force->Set_Collision_Body_List_ID (solids_parameters.rigid_body_parameters.list.rigid_bodies.m + 1);

			penalty_force->Set_Boundary_Only_Collisions (solids_parameters.deformable_body_parameters.list.deformable_objects (1)->tetrahedralized_volume->tetrahedron_mesh);
			solids_parameters.deformable_body_parameters.list.deformable_objects (1)->collision_penalty_forces.Append_Element (penalty_force);
			solids_parameters.deformable_body_parameters.list.deformable_objects (1)->solids_forces.Append_Element (penalty_force);

			if (optimization) optimization->solids_evolution_callbacks = this;
		}
	}
//#####################################################################
// Function Get_Initial_Data
//#####################################################################
	void Get_Initial_Data()
	{
		solids_parameters.deformable_body_parameters.list.Add_Deformable_Object();
		solids_parameters.deformable_body_parameters.list (1).Allocate_Tetrahedralized_Volume();
		TETRAHEDRALIZED_VOLUME<T> &tetrahedralized_volume = *solids_parameters.deformable_body_parameters.list (1).tetrahedralized_volume;
		TETRAHEDRON_MESH& tetrahedron_mesh = tetrahedralized_volume.tetrahedron_mesh;
		SOLIDS_PARTICLE<T, VECTOR_3D<T> >& particles = tetrahedralized_volume.particles;
		std::string input_file;
		std::istream *input;
		int peak_isometric_stress_entries;
        
        //OMPSS
		OMPSS_POOL& pool = *OMPSS_POOL::Singleton();
		input_file = input_directory + STRING_UTILITIES::string_sprintf ("/face_simulation_%lu.tet", pool.Get_n_divisions());

		input = FILE_UTILITIES::Safe_Open_Input (input_file);
		std::cout << "Reading simulation model : " << input_file << std::endl;
		tetrahedralized_volume.template Read<RW> (*input);
		delete input;
		std::cout << "Total particles = " << particles.number << std::endl;
		std::cout << "Total tetrahedra = " << tetrahedron_mesh.tetrahedrons.m << std::endl;
        tetrahedron_mesh.Initialize_Boundary_Mesh();
		particles.Store_Velocity (false);
		particles.Store_Mass (false);
		particles.Store_Mass();
		tetrahedralized_volume.Set_Density (1000);
		tetrahedralized_volume.Set_Mass_Of_Particles (solids_parameters.use_constant_mass);
		tetrahedralized_volume.Update_Bounding_Box();
        
        //OMPSS
		input_file = input_directory + STRING_UTILITIES::string_sprintf ("/node_divisions_%lu.dat", pool.Get_n_divisions());
		tetrahedralized_volume.particles.particle_ranges = new ARRAY<VECTOR_2D<int> >;
		FILE_UTILITIES::Read_From_File<RW> (input_file, *tetrahedralized_volume.particles.particle_ranges);

		input_file = input_directory + "/muscle_list.txt";
		input = FILE_UTILITIES::Safe_Open_Input (input_file, false);
		(*input) >> number_of_muscles;
		std::cout << "muscles = " << number_of_muscles << std::endl;
		muscle_tets.Resize_Array (number_of_muscles);
		muscle_fibers.Resize_Array (number_of_muscles);
		muscle_densities.Resize_Array (number_of_muscles);
		muscle_names.Resize_Array (number_of_muscles), peak_isometric_stress.Resize_Array (number_of_muscles);

		for (int i = 1; i <= number_of_muscles; i++)
		{
			(*input) >> muscle_names (i);
			input_file = input_directory + "/" + muscle_names (i) + ".constitutive_data";

			if (verbose) std::cout << "Reading muscle data : " << muscle_names (i) << ".constitutive_data" << std::endl;

			std::istream *muscle_input = FILE_UTILITIES::Safe_Open_Input (input_file);
			muscle_tets (i).template Read<RW> (*muscle_input);
			muscle_fibers (i).template Read<RW> (*muscle_input);
			muscle_densities (i).template Read<RW> (*muscle_input);
			delete muscle_input;
		}

		delete input;

		LIST_ARRAY<T>::copy ( (T) 3e5, peak_isometric_stress);
		input_file = input_directory + "/peak_isometric_stress_list.txt";
		input = FILE_UTILITIES::Safe_Open_Input (input_file, false);
		(*input) >> peak_isometric_stress_entries;

		for (int i = 1; i <= peak_isometric_stress_entries; i++)
		{
			std::string muscle_name;
			T peak_isometric_stress_value;
			(*input) >> muscle_name >> peak_isometric_stress_value;
			int index;

			if (muscle_names.Find (muscle_name, index)) peak_isometric_stress (index) = peak_isometric_stress_value;
		}

		delete input;

		if (verbose) for (int i = 1; i <= number_of_muscles; i++) std::cout << muscle_names (i) << " : Peak isometric stress = " << peak_isometric_stress (i) << std::endl;

		for (int i = 1; i <= peak_isometric_stress.m; i++) peak_isometric_stress (i) *= peak_isometric_stress_boosting;
		
		//Cranium, jaw, flesh
		input_file = input_directory + "/attachment_list.txt";
		input = FILE_UTILITIES::Safe_Open_Input (input_file, false);
		(*input) >> number_of_attachments;
		std::cout << "attachments = " << number_of_attachments << std::endl;
		attached_nodes.Resize_Array (number_of_attachments);
		jaw_attachment_index = 0;

		for (int i = 1; i <= number_of_attachments; i++)
		{
			std::string attachment_name;
			(*input) >> attachment_name;
			input_file = input_directory + "/" + attachment_name + ".attached_nodes";

			if (verbose) std::cout << "Reading attachment data : " << attachment_name << ".attached_nodes" << std::endl;

			std::istream *attachment_input = FILE_UTILITIES::Safe_Open_Input (input_file);
			attached_nodes (i).template Read<RW> (*attachment_input);

			if (attachment_name == "jaw") jaw_attachment_index = i;
		}

		delete input;

		if (!jaw_attachment_index)
		{
			std::cerr << "ERROR: No jaw attachment specified" << std::endl;
			exit (1);
		}
        //OMPSS
		input_file = input_directory + STRING_UTILITIES::string_sprintf ("/particle_positions_%lu.dat", pool.Get_n_divisions());
		ARRAY<int> particle_positions;
		FILE_UTILITIES::Read_From_File<RW> (input_file, particle_positions);
			
		for (int i = 1; i <= attached_nodes.m; i++)
			for (int j = 1; j <= attached_nodes (i).m; j++)
				attached_nodes (i) (j) = particle_positions (attached_nodes (i) (j));

		attached_nodes_parallel = new ARRAY<LIST_ARRAY<int> > (particles.particle_ranges->m);
	    
        /*
		* We are spreading the attached_nodes particles to the different partitions.
		* 3 attached nodes exist:
		* Cranium
		* Jaw
		* Flesh
	    */
		for (int i = 1; i <= attached_nodes.m; i++)
			for (int j = 1; j <= attached_nodes (i).m; j++)
				for (int k = 1; k <= particles.particle_ranges->m; k++)
				{
					if (attached_nodes (i) (j) >= (*particles.particle_ranges) (k).x && attached_nodes (i) (j) <= (*particles.particle_ranges) (k).y)
						(*attached_nodes_parallel) (k).Append_Element (attached_nodes (i) (j));
				}
	
	if (use_rigid_body_collision)
		{
			solids_parameters.rigid_body_parameters.list.template Add_Rigid_Body<T> (model_directory + "/eftychis_cranium_collision_surface");
			solids_parameters.rigid_body_parameters.list.template Add_Rigid_Body<T> (model_directory + "/eftychis_jaw_collision_surface");
			solids_parameters.rigid_body_parameters.list.rigid_bodies (1)->Set_Coefficient_Of_Friction (0);
			solids_parameters.rigid_body_parameters.list.rigid_bodies (2)->Set_Coefficient_Of_Friction (0);
			LIST_ARRAY<RIGID_BODY_3D<T>*>& rigid_bodies = solids_parameters.rigid_body_parameters.list.rigid_bodies;
			solids_parameters.collision_body_list.Add_Bodies (solids_parameters.rigid_body_parameters.list);
		}
	}
//#####################################################################
// Function Initialize_Tetrahedron_Collisions
//#####################################################################
	void Initialize_Tetrahedron_Collisions()
	{
		std::cout << "Initializing geometric structures for self collision" << std::endl;
		TETRAHEDRALIZED_VOLUME<T> &tetrahedralized_volume = *solids_parameters.deformable_body_parameters.list (1).tetrahedralized_volume;
		TETRAHEDRON_COLLISION_BODY<T> *face_collision_body = new TETRAHEDRON_COLLISION_BODY<T> (tetrahedralized_volume);
		SOLIDS_PARTICLE<T, VECTOR_3D<T> >* undeformed_particles = new SOLIDS_PARTICLE<T, VECTOR_3D<T> > (tetrahedralized_volume.particles);
		TRIANGULATED_SURFACE<T> *undeformed_triangulated_surface = new TRIANGULATED_SURFACE<T> (tetrahedralized_volume.triangulated_surface->triangle_mesh, *undeformed_particles);
		undeformed_triangulated_surface->Update_Triangle_List();
		undeformed_triangulated_surface->Initialize_Triangle_Hierarchy();

		std::cout << "Loading implicit surface for self collision" << std::endl;
		LEVELSET_IMPLICIT_SURFACE<T>* undeformed_levelset;
		FILE_UTILITIES::template Create_From_File<RW> (model_directory + "/face_full.phi", undeformed_levelset);
		undeformed_levelset->Update_Box();
		face_collision_body->Set_Implicit_Surface (undeformed_levelset);
		face_collision_body->Set_Collision_Thickness ( (T) 1e-3);
		face_collision_body->Set_Undeformed_Triangulated_Surface (undeformed_triangulated_surface);
		solids_parameters.collision_body_list.Add_Body (face_collision_body);
	}
//#####################################################################
// Set_External_Positions
//#####################################################################
	void Set_External_Positions (ARRAY<VECTOR_3D<T> >& X, const T time, const int id_number)
	{
		//LOG::Time("Set external positions:");
		switch (id_number)
		{
		case 1:
			attachment_frame_controls->Set_Attachment_Positions (X);
			break;
		default:
			std::cout << "Unrecognized deformable object id number" << std::endl;
			exit (1);
		}
	}
//#####################################################################
// Zero_Out_Enslaved_Position_Nodes
//#####################################################################
    void Zero_Out_Enslaved_Position_Nodes_Helper(unsigned long i, ARRAY<VECTOR_3D<T> >& X)
    {
        #ifdef ENABLE_OMPEXTRAE
        Extrae_user_function(1);
        #endif
        
        for (unsigned long j = 1; j <= (*attached_nodes_parallel)(i).m; j++)
            X ((*attached_nodes_parallel)(i)(j)) = VECTOR_3D<T>();
        
        #ifdef ENABLE_OMPEXTRAE
        Extrae_user_function(0);
        #endif

    }
	void Zero_Out_Enslaved_Position_Nodes (ARRAY<VECTOR_3D<T> >& X, const T time, const int id_number)
	{
		//LOG::Time("Zero out enslaved position nodes:");
		OMPSS_POOL& pool = *OMPSS_POOL::Singleton();
        unsigned long xsize= X.m;
		switch (id_number)
		{
		case 1:
           #ifdef USE_TASKS
                #ifdef ENABLE_OPENMP
                    //ARRAY<VECTOR_3D<T> >* Xp = &X; VECTOR_3D<T>* Xbp = X.base_pointer;
                    //#pragma omp task depend(inout:Xp,Xbp[:xsize])
                    {
                      //  ARRAY<VECTOR_3D<T> >& X = *Xp;
                #elif defined ENABLE_OMPSS
                    //#pragma omp task inout(X) label(ZOEPN)
                    {
                #endif
                        for (unsigned long i = 1; i <= pool.Get_n_divisions(); i++) Zero_Out_Enslaved_Position_Nodes_Helper(i,X);
                    }
                    //#pragma omp taskwait
            #elif defined USE_WORKSHARING_FOR
                #ifdef ENABLE_OPENMP
                    //#pragma omp parallel for schedule(static,1)
                    for (unsigned long i = 1; i <= pool.Get_n_divisions(); i++)
                    {
                        if ((*attached_nodes_parallel)(i).m > 0) Zero_Out_Enslaved_Position_Nodes_Helper(i,X);
                    }
                #elif ENABLE_OMPSS
                    //#pragma omp for schedule(static,1) label (ZOEPN-FOR)
                    for (unsigned long i = 1; i <= pool.Get_n_divisions(); i++)
                    {
                        if ((*attached_nodes_parallel)(i).m > 0) Zero_Out_Enslaved_Position_Nodes_Helper(i,X);
                    }
                #endif
            #endif
			break;
		default:
			std::cout << "Unrecognized deformable object id number" << std::endl;
			exit (1);
		}
	}
	void Zero_Out_Enslaved_Position_Nodes (ARRAY<VECTOR_3D<T> >& X, const T time, const int id_number, const int partition_id)
	{
		assert (id_number == 1);
		LIST_ARRAY<int>const& partition_attached_nodes = (*attached_nodes_parallel) (partition_id);

		for (int i = 1; i <= partition_attached_nodes.m; i++) X (partition_attached_nodes (i)) = VECTOR_3D<T>();
	}
//#####################################################################
// Function Update_Collision_Body_Positions_And_Velocities
//#####################################################################
	void Update_Collision_Body_Positions_And_Velocities (const T time)
	{
		if (use_rigid_body_collision)
		{
			assert (attachment_frame_controls);
			solids_parameters.rigid_body_parameters.list.rigid_bodies (1)->Set_State (attachment_frame_controls->Cranium_Frame());
			solids_parameters.rigid_body_parameters.list.rigid_bodies (2)->Set_State (attachment_frame_controls->Jaw_Frame());
		}
	}
//#####################################################################
};
}
#endif

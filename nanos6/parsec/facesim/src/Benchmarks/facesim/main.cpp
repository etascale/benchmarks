//#####################################################################
// Copyright 2004, Igor Neverov, Eftychios Sifakis.
// This file is part of PhysBAM whose distribution is governed by the license contained in the accompanying file PHYSBAM_COPYRIGHT.txt.
//#####################################################################
#include "../../Public_Library/Utilities/PARSE_ARGS.h"
#include "../../Public_Library/Thread_Utilities/THREAD_POOL.h"
#include "../../Public_Library/Thread_Utilities/THREAD_DIVISION_PARAMETERS.h"
//OMPSS HEADER
#include "../../Public_Library/Thread_Utilities/OMPSS_POOL.h"


#include "FACE_DRIVER.h"
#include "Storytelling/STORYTELLING_EXAMPLE.h"
#include "../../Public_Library/Utilities/LOG.h"

#ifdef ENABLE_EXTRAE
    #include "../../Public_Library/Utilities/EXTRAE.h"
#endif

#ifdef ENABLE_PARSEC_HOOKS
    #include <hooks.h>
#endif

#ifdef ENABLE_PAPI
    #include <papi.h>
#endif





using namespace PhysBAM;

#ifdef ENABLE_PTHREADS
//Use serial code
bool PHYSBAM_THREADED_RUN = true;
# else
//Use multi-threaded code
bool PHYSBAM_THREADED_RUN = false;
#endif //ENABLE_PTHREADS

#if defined ENABLE_OMPSS || ENABLE_OPENMP
bool OMPSS_RUN = true;
#else
bool OMPSS_RUN = false;
#endif

#ifdef ENABLE_EXTRAE
bool USE_EXTRAE = true;
#else
bool USE_EXTRAE = false;
#endif

int main (int argc, char* argv[])
{
#ifdef ENABLE_PAPI
    int numevents=8;
    int eventsets=PAPI_NULL;
    long long values[numevents];
    int Events[numevents];
    Events[0]= PAPI_TOT_INS;
    Events[1]= PAPI_TOT_CYC;
    Events[2]= PAPI_L1_DCM;
    Events[3]= PAPI_L1_LDM;
    Events[4]= PAPI_L1_STM;
    Events[5]= PAPI_L2_DCM;
    Events[6]= PAPI_L2_STM;
    Events[7]= PAPI_L3_TCM;
    int retval=0;

    if (not PAPI_is_initialized()) {
        if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
            std::cout << "PAPI ERROR init bootthread.smpthread: " << retval << "\n";
        } else {
            std::cout << "PAPI ERROR no error init boothread.smpthread: " << retval << "\n";
        }
    } else {
        std::cout << "PAPI ERROR already initialized boothread.smpthread: \n";
    }
    

    if ((retval=PAPI_create_eventset(&eventsets)) != PAPI_OK) {
        std::cout << "PAPI ERROR create set (thread)" << retval << "\n";
    } else {
        std::cout<<"PAPI ERROR no error create set (thread): " << retval << "\n"; 
        if ((retval=PAPI_add_events(eventsets, Events, numevents)) != PAPI_OK){
            std::cout<<"PAPI ERROR adding event to set (thread): " << retval << "\n";
        } else {
            std::cout<<"PAPI ERROR no error adding set (thread): " << retval << "\n";
            if((retval=PAPI_start(eventsets)) != PAPI_OK){
                std::cout<<"PAPI ERROR starting set (thread): " << retval << "\n";
            } else { 
                std::cout<<"PAPI ERROR no error starting set (thread): " << retval << "\n";
            }
        }
    }

    if ((retval=PAPI_reset(eventsets)) != PAPI_OK) {
        std::cout<<"PAPI ERROR reseting: " <<retval<<" \n";
    }
        
#endif

    LOG::Initialize_Logging();


#ifdef PARSEC_VERSION
#define __PARSEC_STRING(x) #x
#define __PARSEC_XSTRING(x) __PARSEC_STRING(x)
	printf ("PARSEC Benchmark Suite Version "__PARSEC_XSTRING (PARSEC_VERSION) "\n");
	fflush (NULL);
#else
	printf ("PARSEC Benchmark Suite\n");
	fflush (NULL);
#endif //PARSEC_VERSION
#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_begin (__parsec_facesim);
#endif
	STORYTELLING_EXAMPLE<float, float> example;
	
	printf("parsing arguments\n");
	printf("Warning: Argument -threads is ignored in OmpSs and OpenMP 4.0! Use NX_ARGS or OMP_NUM_THREADS, respectively.\n");
    PARSE_ARGS parse_args;
	parse_args.Add_Integer_Argument ("-restart", 0);
	parse_args.Add_Integer_Argument ("-lastframe", 300);
	parse_args.Add_Integer_Argument ("-threads", 1);
	parse_args.Add_Option_Argument ("-timing");
    parse_args.Add_String_Argument("-inputdir",example.data_directory);
    parse_args.Add_String_Argument("-outputdir",example.output_directory);
	parse_args.Add_Integer_Argument("-ndivs", 8);
	parse_args.Parse (argc, argv);


	if (parse_args.Is_Value_Set ("-threads"))
	{
		static char tmp_buf[255];
		sprintf (tmp_buf, "PHYSBAM_THREADS=%d", parse_args.Get_Integer_Value ("-threads"));
		printf("%s\n",tmp_buf);
		if (putenv (tmp_buf) < 0) perror ("putenv");
	}
#if defined(ENABLE_OMPSS) || defined(ENABLE_OPENMP) 	
	if (parse_args.Is_Value_Set ("-ndivs"))
	{
		static char tmp_buf[255];
		sprintf (tmp_buf, "PHYSBAM_THREADS=%d", parse_args.Get_Integer_Value ("-ndivs"));
		printf("%s\n",tmp_buf);
		if (putenv (tmp_buf) < 0) perror ("putenv");
	}
#endif

	if (parse_args.Is_Value_Set ("-restart"))
	{
		example.restart = true;
		example.restart_frame = parse_args.Get_Integer_Value ("-restart");
	}

	if (parse_args.Is_Value_Set ("-lastframe"))
	{
		example.last_frame = parse_args.Get_Integer_Value ("-lastframe");
	}

	if (parse_args.Is_Value_Set ("-timing"))
	{
		example.write_output_files = false;
		example.verbose = false;
	}
    if (parse_args.Is_Value_Set("-inputdir"))
    {
        example.data_directory = parse_args.Get_String_Value("-inputdir");
		example.control_directory = example.data_directory + "/Face_Data/Motion_Data/Storytelling_Controls";

		// Simulation source
		example.model_directory = example.data_directory + "/Face_Data/Eftychis_840k";
		example.input_directory = example.model_directory + "/Front_370k";
    }
    if (parse_args.Is_Value_Set("-outputdir"))
    {
        example.output_directory = parse_args.Get_String_Value("-outputdir");
    }

	if (OMPSS_RUN == false) {
		if (PHYSBAM_THREADED_RUN == false && parse_args.Get_Integer_Value ("-threads") > 1)
		{
			printf ("Error: Number of threads cannot be greater than 1 for serial runs\n");
			exit (1);
		}
	} else {
	    //omp_set_num_threads(parse_args.Get_Integer_Value("-threads"));
	    OMPSS_POOL& pool = *OMPSS_POOL::Singleton();
	    if (!pool.Set_n_divisions(parse_args.Get_Integer_Value ("-ndivs")))
	    {
		printf ("Error: could not set the -threads value\n");
	    }
	    printf("Quantity of mesh partitions set to %d\n",pool.Get_n_divisions());
	}
	
	printf("Creating output directory\n");
    FILE_UTILITIES::Create_Directory (example.output_directory);
	LOG::Copy_Log_To_File (example.output_directory + "/log.txt", example.restart);
	
    THREAD_DIVISION_PARAMETERS<float>& parameters = *THREAD_DIVISION_PARAMETERS<float>::Singleton();
	parameters.grid_divisions_3d = VECTOR_3D<int> (5, 5, 5);

	FACE_DRIVER<float, float> driver (example);
    #ifdef ENABLE_OPENMP
        #if defined USE_TASKS
            #pragma omp parallel
            {
                #pragma omp single
                {
                    //#pragma omp task untied default(shared)
                    driver.Execute_Main_Program();
                }
            }
        #elif defined USE_WORKSHARING_FOR
            driver.Execute_Main_Program();
        #endif
    #else
	    driver.Execute_Main_Program();
    #endif
#ifdef ENABLE_PTHREADS
	delete (THREAD_POOL::Singleton());
#elif defined ENABLE_OMPSS || defined ENABLE_OPENMP
	delete (OMPSS_POOL::Singleton());
#endif

#ifdef ENABLE_PARSEC_HOOKS
	__parsec_bench_end();
#endif


    LOG::Pop_Scope();
    LOG::Dump_Log();


#ifdef ENABLE_PAPI
    if ((retval=PAPI_read( eventsets, values)) != PAPI_OK) {
        std::cout<<"PAPI ERROR read : "<< retval << "\n";
    } else {
        std::cout<<"PAPI ERROR no error read oldwd: " << retval << "\n";
    }
    std::cout<<"PAPI val: ";
    for (int i=0; i<numevents; i++){
        std::cout<<"value "<< i <<": " <<values[i]<< " ";
    }
    std::cout<<"\n";
    
    if (PAPI_stop(eventsets, values) != PAPI_OK) {
        std::cout<<"PAPI ERROR stop set (thread): \n";
    } else {
        std::cout<<"PAPI ERROR no error stop set (thread): \n";
    }
    

    if (PAPI_cleanup_eventset(eventsets) != PAPI_OK) {
        std::cout<<"PAPI ERROR cleanup set (thread): \n";
    } else {
        std::cout<<"PAPI ERROR no error cleanup set (thread): \n";
    }

    if (PAPI_destroy_eventset(&eventsets) != PAPI_OK) {
        std::cout<<"PAPI ERROR destroy set (thread): \n";
    } else {
        std::cout<<"PAPI ERROR no error destroy set (thread): \n";
    }
     
#endif
	return 0;
}

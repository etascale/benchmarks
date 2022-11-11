/**
 * NASA Advanced Supercomputing Parallel Benchmarks C++
 *
 * based on NPB 3.3.1
 *
 * original version and technical report:
 * http://www.nas.nasa.gov/Software/NPB/
 *
 * C++ version:
 *      Dalvan Griebler <dalvangriebler@gmail.com>
 *      Gabriell Alves de Araujo <hexenoften@gmail.com>
 *      Júnior Löff <loffjh@gmail.com>
 */

#include <cstdlib>
#include <cstdio>
#include <cmath>

/*****************************************************************/
/******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******/
/*****************************************************************/
void c_print_results(char* name,
		char class_npb,
		int n1, 
		int n2,
		int n3,
		int niter,
		double t,
		double mops,
		char* optype,
		int passed_verification,
		char* npbversion,
		char* compiletime,
		char* compilerversion,
		char* cc,
		char* clink,
		char* c_lib,
		char* c_inc,
		char* cflags,
		char* clinkflags,
		char* rand){
	printf("\n\n %s Benchmark Completed\n", name);
	printf(" class_npb       =                        %c\n", class_npb);
	if((name[0]=='I')&&(name[1]=='S')){
		if(n3==0){
			long nn = n1;
			if(n2!=0){nn*=n2;}
			printf(" Size            =             %12ld\n", nn); /* as in IS */
		}else{
			printf(" Size            =             %4dx%4dx%4d\n", n1,n2,n3);
		}
	}else{
		char size[16];
		int j;
		if((n2==0) && (n3==0)){
			if((name[0]=='E')&&(name[1]=='P')){
				sprintf(size, "%15.0lf", pow(2.0, n1));
				j = 14;
				if(size[j] == '.'){
					size[j] = ' '; 
					j--;
				}
				size[j+1] = '\0';
				printf(" Size            =          %15s\n", size);
			}else{
				printf(" Size            =             %12d\n", n1);
			}
		}else{
			printf(" Size            =           %4dx%4dx%4d\n", n1, n2, n3);
		}
	}	
	printf(" Iterations      =             %12d\n", niter); 
	printf(" Time in seconds =             %12.2f\n", t);
	printf(" Mop/s total     =             %12.2f\n", mops);
	printf(" Operation type  = %24s\n", optype);
	if(passed_verification < 0){
		printf( " Verification    =            NOT PERFORMED\n");
	}else if(passed_verification){
		printf(" Verification    =               SUCCESSFUL\n");
	}else{
		printf(" Verification    =             UNSUCCESSFUL\n");
	}
	printf(" Version         =             %12s\n", npbversion);
	printf(" Compiler ver    =             %12s\n", compilerversion);
	printf(" Compile date    =             %12s\n", compiletime);
	printf("\n Compile options:\n");
	printf("    CC           = %s\n", cc);
	printf("    CLINK        = %s\n", clink);
	printf("    C_LIB        = %s\n", c_lib);
	printf("    C_INC        = %s\n", c_inc);
	printf("    CFLAGS       = %s\n", cflags);
	printf("    CLINKFLAGS   = %s\n", clinkflags);
	printf("    RAND         = %s\n", rand);
#ifdef SMP
	evalue = getenv("MP_SET_NUMTHREADS");
	printf("   MULTICPUS = %s\n", evalue);
#endif    
	/* 
	 * printf(" Please send the results of this run to:\n\n");
	 * printf(" NPB Development Team\n");
	 * printf(" Internet: npb@nas.nasa.gov\n \n");
	 * printf(" If email is not available, send this to:\n\n");
	 * printf(" MS T27A-1\n");
	 * printf(" NASA Ames Research Center\n");
	 * printf(" Moffett Field, CA  94035-1000\n\n");
	 * printf(" Fax: 650-604-3957\n\n");
	 */
	printf("\n\n");
	
	printf("----------------------------------------------------------------------\n");
	printf("    NPB-CPP is developed by: \n");
	printf("        Dalvan Griebler\n");
	printf("        Gabriell Alves de Araujo (Sequential Porting)\n");
	printf("        Júnior Löff (Parallel Implementation)\n");
	printf("\n");
	printf("    In case of questions or problems, please send an e-mail to us:\n");	
	printf("        dalvan.griebler; gabriell.araujo; junior.loff@edu.pucrs.br\n");
	printf("----------------------------------------------------------------------\n");
	printf("\n");
}

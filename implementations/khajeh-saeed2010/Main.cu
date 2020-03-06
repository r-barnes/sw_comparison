#include <mpi.h>

#include <cstdlib>
#include <stdio.h>
#include <string.h>
#include <math.h>
//#include <cutil.h>
#include <limits.h>
#include <iostream>
#include <fstream>
#include <new>
// #include <cutil_inline.h>
//#include <Header.h>

using namespace std;

extern "C" 
unsigned int compare( const int* CPU, const int* GPU,const unsigned int len);


#include <Full_SW_CPU.cu>
#include <Kernel_1_CPU.cu>
#include <Kernel_1_Max_CPU.cu>
#include <Kernel_1_Max_CPU_MPI.cu>
#include <Kernel_2_CPU.cu>
#include <Kernel_2_Max_CPU_MPI.cu>
#include <Kernel_3_CPU.cu>
#include <Kernel_3_Max_CPU_MPI.cu>
#include <Kernel_4_CPU.cu>
#include <Kernel_5_CPU.cu>

#include <0_Data_Generator.cu>
#include <1_Pairwise_Alignment.cu>
#include <2_Sequence_Extraction.cu>
#include <3_Locating_Similar_Sequences.cu>
#include <4_Global_Alignment.cu>
#include <5_Multiple_Alignment.cu>

////////////////////////////////////////////////////////////////////////////////
const unsigned int   L_A1  = 4194304;
const unsigned int   L_B   = 64; 
const unsigned int   L_Gap = 512; 

const int   si			= 5;
const int   dis			=-3;
const int   Gex			= 1;
const int   Gop			= 8;
const int   K1R         = 256;

const int   Kerene2Max  = 128;
const int   K2R         = K1R;

const int   K4_S1		= 0;
const int   K4_S2		= 1;
const int   K4_S3		= 2;

const unsigned int        DATA_SZ_B   =  L_B   * sizeof(int);
const unsigned int        DATA_SZ_K1  =  K1R   * sizeof(int);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
 
	int NumProcs, MyProc, NameLength;
	char ProcessorName[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv );
	MPI_Comm_size( MPI_COMM_WORLD, &NumProcs );
	MPI_Comm_rank( MPI_COMM_WORLD, &MyProc );
	MPI_Get_processor_name(ProcessorName,&NameLength);
	CUDA_SAFE_CALL(cudaSetDevice(MyProc));
 
	int *h_B, *h_A_Location, *h_B_Location, *h_Max_CPU,*h_A_Location_All, *h_B_Location_All, *h_Max_CPU_All;
	int *h_A;
	int *d_A, *d_B; 
	int L_A, Start_A, L_M, DATA_SZ_L, DATA_SZ_A, DATA_SZ_H, DATA_SZ_M;
	
	dim3 BlockSize(64, 1);         //64
	dim3 GridSize (1024, 1);        // 512
  	int Threads_N = GridSize.x*GridSize.y*BlockSize.x*BlockSize.y;

	h_A               = (int *)malloc(sizeof(int)*L_A1);
	h_B               = (int *)malloc(sizeof(int)*L_B );
	h_A_Location_All  = (int *)malloc(DATA_SZ_K1);
	h_B_Location_All  = (int *)malloc(DATA_SZ_K1);
	h_Max_CPU_All     = (int *)malloc(DATA_SZ_K1);
	h_A_Location      = (int *)malloc(DATA_SZ_K1);
	h_B_Location      = (int *)malloc(DATA_SZ_K1);
	h_Max_CPU         = (int *)malloc(DATA_SZ_K1);

	CUDA_SAFE_CALL( cudaMalloc((void **)&d_B,       DATA_SZ_B) );

	//---------------------------------------------------------------- Data Generation -------------------------------------------------------------------------------
	
	Data_Generator (h_A, h_B, 
	  			    MyProc, NumProcs, L_B, L_A, L_A1, L_Gap, L_M, Threads_N, Start_A, GridSize.x, BlockSize.x,
				    DATA_SZ_A, DATA_SZ_H, DATA_SZ_M, DATA_SZ_L);
	
	MPI_Barrier(MPI_COMM_WORLD);
	//---------------------------------------------------------------- Kernel 1 Start -------------------------------------------------------------------------------
        
	Pairwise_Alignment (h_A, d_B, h_B, h_Max_CPU_All, h_A_Location_All, h_B_Location_All,
						h_Max_CPU, h_A_Location, h_B_Location,
						K1R, MyProc, NumProcs, L_B, L_A, L_A1,  si, dis, Gop, Gex, L_M, Threads_N, Start_A,
						DATA_SZ_K1, DATA_SZ_A, DATA_SZ_H, DATA_SZ_B, DATA_SZ_M, DATA_SZ_L);
	MPI_Barrier(MPI_COMM_WORLD);
	//-------------------------------------------------------------------- Kernel 2 Start ------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_A,       DATA_SZ_A) );                
	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, DATA_SZ_A, cudaMemcpyHostToDevice) );   
	int *h_Kernel_2_output_B = new int [K2R*(Kerene2Max+1)];

	Sequence_Extraction (h_A, d_A, d_B, h_Kernel_2_output_B, h_B, h_Max_CPU_All,h_A_Location_All, h_B_Location_All,
						 h_Max_CPU,h_A_Location, h_B_Location,
						 DATA_SZ_K1,  K1R,  MyProc, NumProcs,L_B, L_A, K2R, Kerene2Max, si,  dis, Gop, Gex, Start_A);
	
	MPI_Barrier(MPI_COMM_WORLD);
	//---------------------------------------------------------------- MPI Starts for Kernel 3 ------------------------------------------------------------------------------------------

	int K3_Length = 32;    //K3_L_K2+1;
	int K3R = 20;              // Number of Maximum Similarity
	int K3_Safety = 2;
	int K_3_R = 128;   //cnt+1;            // Number of Reports form Kernel 3
	int K3_Report = 128;
  	int K3_Timed_Out = 2000;

	int BlockSize_K3_X = 64;
	int GridSize_K3_X = 128;

	int *h_Val_K3_K4_B_All, *h_Length_Seq_K4_All;
	int *d_Val_K3_K4_B_All, *d_Length_Seq_K4_All;

	int DATA_SZ_K31_MPI     = K3_Safety *  K3_Length  *  K3_Report * K_3_R * sizeof(int);
	int DATA_SZ_K32_MPI     =                            K3_Report * K_3_R * sizeof(int);
	
	h_Val_K3_K4_B_All = (int *)malloc(DATA_SZ_K31_MPI);
	h_Length_Seq_K4_All= (int *)malloc(DATA_SZ_K32_MPI);

	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Val_K3_K4_B_All,	  DATA_SZ_K31_MPI) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Length_Seq_K4_All,	  DATA_SZ_K32_MPI) );

	Locating_Similar_Sequences (h_A, d_A, h_Kernel_2_output_B, h_B, 
								h_Val_K3_K4_B_All, h_Length_Seq_K4_All, d_Val_K3_K4_B_All, d_Length_Seq_K4_All,
								 L_A, L_A1, K2R, Kerene2Max,
								 si,  dis,  Gop, Gex, K3_Timed_Out,Start_A,
								 K_3_R,  MyProc, K3_Report, K3_Safety, NumProcs, K3_Length, K3R, BlockSize_K3_X, GridSize_K3_X);

	//-------------------------------------------------------------------- Kernel 4 Start  ----------------------------------------------------------------------------------------
	int *h_Kernel_4_output, *Kernel_4_output_CPU;
	int *d_Kernel_4_output;

	int DATA_SZ_K4 = K3_Report * K3_Report * K_3_R * sizeof(int);

	h_Kernel_4_output  = (int *)malloc(DATA_SZ_K4);
	Kernel_4_output_CPU	= (int *)malloc(DATA_SZ_K4);

	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Kernel_4_output,	 DATA_SZ_K4) );

	Global_Alignment (h_Val_K3_K4_B_All, h_Length_Seq_K4_All, d_Val_K3_K4_B_All, d_Length_Seq_K4_All,
					  h_Kernel_4_output, Kernel_4_output_CPU, d_Kernel_4_output, 
	   			      K_3_R/NumProcs, MyProc, K3_Report, K3_Safety, NumProcs,  K3_Length, BlockSize_K3_X,
					  K4_S1,  K4_S2, K4_S3);
	MPI_Barrier(MPI_COMM_WORLD);	
	//-------------------------------------------------------------------- Kernel 5 Start  ----------------------------------------------------------------------------------------
	
	Multiple_Alignment (h_Val_K3_K4_B_All, h_Length_Seq_K4_All, d_Val_K3_K4_B_All, 
						d_Length_Seq_K4_All, Kernel_4_output_CPU, d_Kernel_4_output,
						K_3_R/NumProcs, MyProc, K3_Report, K3_Safety, NumProcs,  K3_Length, BlockSize_K3_X,
						K4_S1,  K4_S2, K4_S3);

	//-------------------------------------------------------------------- SSCA # 1 ------------------------------------------------------------------------------------------
/**/
	MPI_Barrier(MPI_COMM_WORLD);	

	// cleanup memory

    free(h_A);
    free(h_B);
	free(h_Max_CPU);

	free(h_A_Location);
	free(h_B_Location);
delete [] h_Kernel_2_output_B;

	
    CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_B));

	
	MPI_Finalize();
	if (MyProc==0)
	{
		printf("       \n");
		printf("Shutting down...\n");
		// CUT_EXIT(argc, argv);
	}
}
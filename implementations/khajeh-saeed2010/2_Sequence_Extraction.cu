#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>

#include <SW_kernel_2.cu>
#include <Kernel_2_CPU.cu>
#include <Kernel_2_Max_CPU_MPI.cu>

void Sequence_Extraction (int *h_A, int *d_A, int *d_B, int *h_Kernel_2_output_B, int *h_B, int *h_Max_CPU_All,
						  int *h_A_Location_All, int *h_B_Location_All,
						  int *h_Max_CPU, int *h_A_Location, int *h_B_Location,
						  int DATA_SZ_K1, int K1R, int MyProc, int NumProcs, int L_B,  int L_A, int K2R, int Kerene2Max,
						  int si, int dis, int Gop, int Gex, int Start_A)
{

	unsigned int hTimer;
	dim3 BlockSize(64, 1);         //64
	dim3 GridSize (200, 1);        // 512
	
	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
	CUT_SAFE_CALL ( cutStartTimer(hTimer)   );

	float Start_Kernel_2_CPU, End_Kernel_2_CPU, Start_Kernel_2_GPU, End_Kernel_2_GPU;
	int *h_F2;  
	int *d_F2, *d_H2, *d_E2, *d_FLoc2, *d_ELoc2, *d_L2;
	int *d_Kernel_2_output_A, *d_Kernel_2_output_B, *d_Max_Kernel_2, *d_Loc_A_Kernel_2, *d_Loc_B_Kernel_2; 
	int *h_Kernel_2_A, *h_Kernel_2_B;
	int *Kernel_2_output_A_CPU, *Kernel_2_output_B_CPU;
	int *h_Kernel_2_A_All, *h_Kernel_2_B_All, *h_Max_CPU_output;
	
	int   DATA_SZ_K2  =  K2R*(Kerene2Max+1)*(Kerene2Max+1) * sizeof(int);
	int   DATA_GATHER_K2_MPI = K2R*(Kerene2Max);
	int   DATA_SZ_K21  =  K2R*(Kerene2Max+1) * sizeof(int);
	    
   	Kernel_2_output_A_CPU   = (int *)malloc(DATA_SZ_K2*NumProcs);
	Kernel_2_output_B_CPU   = (int *)malloc(DATA_SZ_K2*NumProcs);
	h_Max_CPU_output        = (int *)malloc(K1R*NumProcs*sizeof(int));
	h_F2			        = (int *)malloc(DATA_SZ_K2);
	h_Kernel_2_A_All = (int *)malloc(DATA_SZ_K21*NumProcs);
	h_Kernel_2_B_All = (int *)malloc(DATA_SZ_K21*NumProcs);
	h_Kernel_2_B     = (int *)malloc(DATA_SZ_K21);
	h_Kernel_2_A     = (int *)malloc(DATA_SZ_K21);


	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_Kernel_2  ,   DATA_SZ_K1) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_A_Kernel_2,   DATA_SZ_K1) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_B_Kernel_2,   DATA_SZ_K1) );


	CUDA_SAFE_CALL( cudaMalloc((void **)&d_F2,				 DATA_SZ_K2) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_E2,				 DATA_SZ_K2) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_H2,				 DATA_SZ_K2) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_FLoc2,			 DATA_SZ_K2) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_ELoc2,		     DATA_SZ_K2) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_L2,			     DATA_SZ_K2) );
    
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Kernel_2_output_A,DATA_SZ_K21) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Kernel_2_output_B,DATA_SZ_K21) );

    for(int i = 0; i < K2R*(Kerene2Max+1)*(Kerene2Max+1); ++i)     
    {
		Kernel_2_output_A_CPU[i]=0;
		Kernel_2_output_B_CPU[i]=0;
    } 

	for(int i = 0; i < K2R*(Kerene2Max+1)/NumProcs; ++i)     
    {
   		h_Kernel_2_A[i]=0;
   		h_F2[i]=0;
    } 
	for(int i = 0; i < K2R*(Kerene2Max+1)*NumProcs; ++i)     
		h_Kernel_2_A_All[i]=0;


    CUDA_SAFE_CALL( cudaMemcpy(d_F2,				h_F2,		      DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_H2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_E2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_FLoc2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_ELoc2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_L2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );                        

    CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_2_output_A, h_Kernel_2_A,  DATA_SZ_K21, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_2_output_B, h_Kernel_2_A,  DATA_SZ_K21, cudaMemcpyHostToDevice) );
    
	CUDA_SAFE_CALL( cudaMemcpy(d_Max_Kernel_2  , h_Max_CPU,    K1R*sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Loc_A_Kernel_2, h_A_Location, K1R*sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Loc_B_Kernel_2, h_B_Location, K1R*sizeof(int), cudaMemcpyHostToDevice) );
    
	
	if (MyProc==0)
	{
		Start_Kernel_2_CPU =	cutGetTimerValue(hTimer);
		Kernel_2_CPU(h_A,h_B,h_Max_CPU_All,h_A_Location_All, h_B_Location_All, 
					Kernel_2_output_A_CPU, Kernel_2_output_B_CPU, 
			         si, dis, Gop, Gex, Kerene2Max, K2R);
		End_Kernel_2_CPU   =	cutGetTimerValue(hTimer);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	Start_Kernel_2_GPU =	cutGetTimerValue(hTimer);
	Kernel_2 <<<GridSize,BlockSize>>> (d_F2,d_H2,d_E2,d_FLoc2,d_ELoc2,d_L2,d_A, d_B, d_Max_Kernel_2, d_Loc_A_Kernel_2, d_Loc_B_Kernel_2,
									   d_Kernel_2_output_A,d_Kernel_2_output_B,
									   K2R, Kerene2Max, si, dis, Gop, Gex, Start_A);
	CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	CUT_CHECK_ERROR("Kernel 2");
	MPI_Barrier(MPI_COMM_WORLD);
	End_Kernel_2_GPU =	cutGetTimerValue(hTimer);

	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_2_A, d_Kernel_2_output_A,   DATA_SZ_K21, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_2_B, d_Kernel_2_output_B,   DATA_SZ_K21, cudaMemcpyDeviceToHost) );

	
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather( h_Max_CPU   , K1R               , MPI_INT, h_Max_CPU_output, K1R               , MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_Kernel_2_A, DATA_GATHER_K2_MPI, MPI_INT, h_Kernel_2_A_All, DATA_GATHER_K2_MPI, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_Kernel_2_B, DATA_GATHER_K2_MPI, MPI_INT, h_Kernel_2_B_All, DATA_GATHER_K2_MPI, MPI_INT, 0, MPI_COMM_WORLD); 
	

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (MyProc ==0)
	{
		printf("CPU execution time for Kernel 2 :   %f ms\n", End_Kernel_2_CPU -Start_Kernel_2_CPU);
		printf("GPU execution time for Kernel 2 :   %f ms\n", End_Kernel_2_GPU -Start_Kernel_2_GPU);
		printf("Speedup                         :   %f   \n", (End_Kernel_2_CPU -Start_Kernel_2_CPU)/(End_Kernel_2_GPU -Start_Kernel_2_GPU));
		
		Kernel_2_Max_CPU_MPI(h_Max_CPU_output, h_Kernel_2_A_All, h_Kernel_2_B_All, 
						     h_Max_CPU       , h_Kernel_2_A    , h_Kernel_2_output_B    ,
						     K1R, K1R*NumProcs, Kerene2Max);

		unsigned int result_regtest_K2A = cutComparei( Kernel_2_output_A_CPU, h_Kernel_2_A       , K2R*(Kerene2Max));
		unsigned int result_regtest_K2B = cutComparei( Kernel_2_output_B_CPU, h_Kernel_2_output_B, K2R*(Kerene2Max));

		printf("Verification of Kernel 2        :   %s   \n", ((1 == result_regtest_K2A) && (1 == result_regtest_K2B)) ? "PASSED" : "FAILED");
		printf("-------------------------------------------------------------\n");
	}

	free(h_F2);
	free(h_Kernel_2_A);
	free(h_Kernel_2_B);
	free(Kernel_2_output_A_CPU);
	free(Kernel_2_output_B_CPU);
	free(h_Kernel_2_A_All);
	free(h_Kernel_2_B_All);
	free(h_Max_CPU_output);

	CUDA_SAFE_CALL(cudaFree(d_F2));
	CUDA_SAFE_CALL(cudaFree(d_H2));
	CUDA_SAFE_CALL(cudaFree(d_E2));
	CUDA_SAFE_CALL(cudaFree(d_FLoc2));
	CUDA_SAFE_CALL(cudaFree(d_ELoc2));
	CUDA_SAFE_CALL(cudaFree(d_L2));
	CUDA_SAFE_CALL(cudaFree(d_Max_Kernel_2));
	CUDA_SAFE_CALL(cudaFree(d_Loc_A_Kernel_2));
	CUDA_SAFE_CALL(cudaFree(d_Loc_B_Kernel_2));
	CUDA_SAFE_CALL(cudaFree(d_Kernel_2_output_A));
	CUDA_SAFE_CALL(cudaFree(d_Kernel_2_output_B));
    
    CUT_CHECK_ERROR("Kernel 2");
}
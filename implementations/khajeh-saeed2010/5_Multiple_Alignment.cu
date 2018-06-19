#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>


#include <SW_kernel_5.cu>

void Multiple_Alignment (int *h_Val_K3_K4_B_All, int *h_Length_Seq_K4_All, int *d_Val_K3_K4_B_All, 
						 int *d_Length_Seq_K4_All, int *Kernel_4_output_CPU, int *d_Kernel_4_output,
					     int K_3_R, int MyProc, int K3_Report, int K3_Safety, int NumProcs, int K3_Length, int BlockSize_K3,
					     int K4_S1, int K4_S2, int K4_S3)
{
    unsigned int hTimer;
	dim3 BlockSize_K4(64, 1);         //64
    dim3 GridSize_K4 (128, 1);        // 512
    CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );

	int *d_Kernel_5_Temp, *d_Kernel_5_output;
	int *h_Kernel_5_Temp, *h_Kernel_5_output;
	int *Kernel_5_output_CPU;
	int *h_Kernel_5_output_All;

	int K5_Length = K3_Length;    //Max Number 40-42;
	int K5R = K_3_R * NumProcs;

	int DATA_SZ_K51 = K5_Length * K3_Report * K5R * sizeof(int);
	int DATA_SZ_K52 = K3_Report * K3_Report * K5R * sizeof(int);

  	h_Kernel_5_output     = (int *)malloc(DATA_SZ_K51);
	Kernel_5_output_CPU   = (int *)malloc(DATA_SZ_K51);

	h_Kernel_5_output_All = (int *)malloc(DATA_SZ_K51);

	h_Kernel_5_Temp   = (int *)malloc(DATA_SZ_K52);


	for( int i = 0; i < DATA_SZ_K51/4 ; i++) 
	{
		h_Kernel_5_output[i]=0;
		h_Kernel_5_output_All[i]=0;
		Kernel_5_output_CPU[i]=0;
	}

	for( int i = 0; i < DATA_SZ_K52/4 ; i++) 
   		h_Kernel_5_Temp[i]=0;

	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Kernel_5_output,	 DATA_SZ_K51) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Kernel_5_Temp,	 DATA_SZ_K52) );

	CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_5_output,	h_Kernel_5_output,	DATA_SZ_K51,	cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_5_Temp  ,	h_Kernel_5_Temp,	DATA_SZ_K52,	cudaMemcpyHostToDevice) );
	
	float Start_Kernel_5_CPU, End_Kernel_5_CPU;
	if (MyProc==0)
	{
  		Start_Kernel_5_CPU =	cutGetTimerValue(hTimer);  

		Kernel_5_CPU (h_Val_K3_K4_B_All, Kernel_5_output_CPU, Kernel_4_output_CPU, h_Length_Seq_K4_All, 
					 K3_Length, K3_Report, K3_Safety, K5R, K5_Length, K4_S1, K4_S2, K4_S3);
		End_Kernel_5_CPU =	cutGetTimerValue(hTimer);
	}
   	MPI_Barrier(MPI_COMM_WORLD);

	float Start_Kernel_5_GPU =	cutGetTimerValue(hTimer);  

	Kernel_51 <<<GridSize_K4,BlockSize_K4>>> (d_Kernel_4_output, d_Kernel_5_Temp, K3_Report, K5R);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	Kernel_52 <<<GridSize_K4,BlockSize_K4>>> (d_Val_K3_K4_B_All, d_Kernel_5_output, d_Length_Seq_K4_All, d_Kernel_5_Temp,
											 K3_Length, K3_Report, K3_Safety, K5R/NumProcs,MyProc,
											 K5_Length, K4_S1, K4_S2, K4_S3);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );				   
 	MPI_Barrier(MPI_COMM_WORLD);
	float End_Kernel_5_GPU =	cutGetTimerValue(hTimer);
	CUT_CHECK_ERROR("Kernel 5");

	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_5_output, d_Kernel_5_output,   DATA_SZ_K51,    cudaMemcpyDeviceToHost) );

	int MPI_Reduce_Size = K5_Length * K3_Report * K5R;
	MPI_Allreduce( h_Kernel_5_output, h_Kernel_5_output_All, MPI_Reduce_Size, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

	if (MyProc==0)
	{
		unsigned int result_regtest_K5 = cutComparei( Kernel_5_output_CPU, h_Kernel_5_output_All, K5_Length * K3_Report * K5R);
	    
		printf("CPU execution time for Kernel 5 :   %f ms \n", End_Kernel_5_CPU - Start_Kernel_5_CPU);
		printf("GPU execution time for Kernel 5 :   %f ms \n", End_Kernel_5_GPU - Start_Kernel_5_GPU);
		printf("Speedup:                        :   %f    \n",(End_Kernel_5_CPU - Start_Kernel_5_CPU)/(End_Kernel_5_GPU - Start_Kernel_5_GPU));
		printf("Verification of Kernel 5        :   %s    \n", ((1 == result_regtest_K5) ? "PASSED" : "FAILED"));
		printf("-------------------------------------------------------------\n");
	} 
}
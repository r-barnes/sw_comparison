#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>

#include <SW_kernel_4.cu>


void Global_Alignment (int *h_Val_K3_K4_B_All, int *h_Length_Seq_K4_All, int *d_Val_K3_K4_B_All, int *d_Length_Seq_K4_All,
					   int *h_Kernel_4_output, int *Kernel_4_output_CPU, int *d_Kernel_4_output,
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

	int *d_D, *d_Start_A, *d_Start_B;
	int *h_D, *h_Start_A, *h_Start_B;
	int *h_Kernel_4_output_All;

	int DATA_SZ_K4 = K3_Report * K3_Report * K_3_R * sizeof(int) * NumProcs;
	int DATA_SZ_K42 = (K3_Length+1)* (K3_Length+1)* K_3_R * (BlockSize_K3+1)  * sizeof(int) * NumProcs;
	int DATA_SZ_K43 = K3_Report * K_3_R * sizeof(int) * NumProcs;

	h_Kernel_4_output_All  = (int *)malloc(DATA_SZ_K4);
	h_D  = (int *)malloc(DATA_SZ_K42);
	h_Start_A = (int *)malloc(DATA_SZ_K43);
	h_Start_B = (int *)malloc(DATA_SZ_K43);
	    
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Start_A,	 DATA_SZ_K43) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Start_B,	 DATA_SZ_K43) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_D,	     DATA_SZ_K42) );

	for( int i = 0; i < DATA_SZ_K4/4 ; i++) 
	{	
		h_Kernel_4_output[i]=0;
		Kernel_4_output_CPU[i]=0;
	}

	for( int i = 0; i < DATA_SZ_K42/4 ; i++) 
	{
   		h_D[i]=0;
	}

	for (int k_Bl = 0; k_Bl<(K_3_R * NumProcs); k_Bl++)
	{
		for (int j_th = 0; j_th<(BlockSize_K3+1); j_th++)
		{
			for( int i = 0; i < (K3_Length+1) ; i++)
			{
   				int index1 = i              + j_th*(K3_Length+1)*(K3_Length+1) + k_Bl * (K3_Length+1)*(K3_Length+1)*(BlockSize_K3+1);
				int index2 = i*(K3_Length+1)+ j_th*(K3_Length+1)*(K3_Length+1) + k_Bl * (K3_Length+1)*(K3_Length+1)*(BlockSize_K3+1);
				h_D[index1] = i * K4_S3;
				h_D[index2] = i * K4_S3;
			}
		}
	}
	int counter = 0;
	for (int i = 0; i<(K_3_R * NumProcs); i++)
	{
		for (int j=i+1; j<(K_3_R * NumProcs); j++)
		{
			h_Start_A[counter]= j;
			h_Start_B[counter]= i;
			counter++;
		}
	}

	CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_4_output,	h_Kernel_4_output,	DATA_SZ_K4 ,	cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_D              ,	h_D              ,	DATA_SZ_K42,	cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Start_A        ,	h_Start_A        ,	DATA_SZ_K43,	cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Start_B        ,	h_Start_B        ,	DATA_SZ_K43,	cudaMemcpyHostToDevice) );

	int Start_Th1 =0;
	int End_Th1 = (K3_Report*K3_Report-K3_Report)/2;
	float Start_Kernel_4_CPU,End_Kernel_4_CPU; 

	if (MyProc==0)
	{
		Start_Kernel_4_CPU =	cutGetTimerValue(hTimer);  

	   Kernel_4_CPU (h_Val_K3_K4_B_All, Kernel_4_output_CPU, h_Start_A, h_Start_B, h_Length_Seq_K4_All,
								K3_Length, K3_Report, K3_Safety, K_3_R*NumProcs,
								Start_Th1,  End_Th1,K4_S1, K4_S2, K4_S3);
		End_Kernel_4_CPU =	cutGetTimerValue(hTimer);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	float Start_Kernel_4_GPU =	cutGetTimerValue(hTimer);  

	Kernel_4 <<<GridSize_K4,BlockSize_K4>>> (d_Val_K3_K4_B_All,d_D, d_Kernel_4_output, d_Start_A, d_Start_B,d_Length_Seq_K4_All,
							K3_Length, K3_Report, K3_Safety, K_3_R, MyProc,
							Start_Th1,  End_Th1,K4_S1, K4_S2, K4_S3);
		
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
   	MPI_Barrier(MPI_COMM_WORLD);
	float End_Kernel_4_GPU =	cutGetTimerValue(hTimer);
	CUT_CHECK_ERROR("Kernel 4");

	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_4_output, d_Kernel_4_output,   DATA_SZ_K4,    cudaMemcpyDeviceToHost) );
	int MPI_Reduce_Size = K3_Report * K3_Report * K_3_R * NumProcs;
	MPI_Allreduce( h_Kernel_4_output, h_Kernel_4_output_All, MPI_Reduce_Size, MPI_INT, MPI_SUM, MPI_COMM_WORLD );

	int *Temp;
	Temp = h_Kernel_4_output;
	h_Kernel_4_output = h_Kernel_4_output_All;
	h_Kernel_4_output_All = Temp;

	CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_4_output,	h_Kernel_4_output,	DATA_SZ_K4 ,	cudaMemcpyHostToDevice) );
 
	if (MyProc==0)
	{
		unsigned int result_regtest_K4    = cutComparei( Kernel_4_output_CPU, h_Kernel_4_output, K3_Report * K3_Report * K_3_R * NumProcs);

		printf("CPU execution time for Kernel 4 :   %f ms \n", End_Kernel_4_CPU - Start_Kernel_4_CPU);
		printf("GPU execution time for Kernel 4 :   %f ms \n", End_Kernel_4_GPU - Start_Kernel_4_GPU);
		printf("Speedup:                        :   %f    \n",(End_Kernel_4_CPU - Start_Kernel_4_CPU)/(End_Kernel_4_GPU - Start_Kernel_4_GPU));
		printf("Verification of Kernel 4        :   %s    \n", ((1 == result_regtest_K4) ? "PASSED" : "FAILED"));
		printf("-------------------------------------------------------------\n");
	}
	free(h_Kernel_4_output_All);
}
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>

#include <SW_kernel_2.cu>

void Sequence_Extraction (int *h_A, int *d_A, int *d_B, int *h_Kernel_2_output_B_All, int *h_B, int *h_Max_CPU_All,
						  int *h_A_Location_All, int *h_B_Location_All,
						  int DATA_SZ_K1, int K1R, int MyProc, int NumProcs, int L_B,  int L_A, int K2R, int Kerene2Max,
						  int si, int dis, int Gop, int Gex)
{
    
	unsigned int hTimer;
	dim3 BlockSize(64, 1);         //64
    dim3 GridSize (200, 1);        // 512

	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );

    int *h_F2;  
    int *d_F2, *d_H2, *d_E2, *d_FLoc2, *d_ELoc2, *d_L2;
	int *d_Kernel_2_output_A, *d_Kernel_2_output_B, *d_Max_Kernel_2, *d_Loc_A_Kernel_2, *d_Loc_B_Kernel_2; 
	int *h_Kernel_2_output_A, *h_Kernel_2_output_B;
	int *Kernel_2_output_A_CPU, *Kernel_2_output_B_CPU;
	int *h_Max_CPU, *h_A_Location, *h_B_Location, *h_Kernel_2_output_A_All;
	
	const unsigned int   DATA_SZ_K2  =  K2R*(Kerene2Max+1)*(Kerene2Max+1) * sizeof(int)/NumProcs;
	const unsigned int   DATA_GATHER_K2_MPI = K2R*(Kerene2Max) / NumProcs;
	const unsigned int   DATA_SZ_K21  =  K2R*(Kerene2Max+1) * sizeof(int)/NumProcs;
	    
   	Kernel_2_output_A_CPU   = (int *)malloc(DATA_SZ_K2*NumProcs);
	Kernel_2_output_B_CPU   = (int *)malloc(DATA_SZ_K2*NumProcs);
	h_F2			        = (int *)malloc(DATA_SZ_K2);
	h_Kernel_2_output_A_All = (int *)malloc(DATA_SZ_K21*NumProcs);
	h_Kernel_2_output_B     = (int *)malloc(DATA_SZ_K21);
	h_Kernel_2_output_A     = (int *)malloc(DATA_SZ_K21);

	h_Max_CPU             = (int *)malloc(DATA_SZ_K1/NumProcs);
	h_A_Location          = (int *)malloc(DATA_SZ_K1/NumProcs);
	h_B_Location          = (int *)malloc(DATA_SZ_K1/NumProcs);

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_Kernel_2  ,   DATA_SZ_K1/NumProcs) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_A_Kernel_2,   DATA_SZ_K1/NumProcs) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_B_Kernel_2,   DATA_SZ_K1/NumProcs) );


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
   		h_Kernel_2_output_A[i]=0;
   		h_F2[i]=0;
    } 
	for(int i = 0; i < K2R*(Kerene2Max+1); ++i)     
		h_Kernel_2_output_A_All[i]=0;


    CUDA_SAFE_CALL( cudaMemcpy(d_F2,				h_F2,		      DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_H2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_E2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_FLoc2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_ELoc2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_L2,				h_F2,			  DATA_SZ_K2, cudaMemcpyHostToDevice) );                        

    CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_2_output_A, h_Kernel_2_output_A,  DATA_SZ_K21, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Kernel_2_output_B, h_Kernel_2_output_A,  DATA_SZ_K21, cudaMemcpyHostToDevice) );
    
	MPI_Scatter( h_Max_CPU_All   , K1R/NumProcs, MPI_INT, h_Max_CPU   , K1R/NumProcs, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Scatter( h_A_Location_All, K1R/NumProcs, MPI_INT, h_A_Location, K1R/NumProcs, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Scatter( h_B_Location_All, K1R/NumProcs, MPI_INT, h_B_Location, K1R/NumProcs, MPI_INT, 0, MPI_COMM_WORLD); 

	
	CUDA_SAFE_CALL( cudaMemcpy(d_Max_Kernel_2,   h_Max_CPU,    K1R*sizeof(int)/NumProcs, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_Loc_A_Kernel_2, h_A_Location, K1R*sizeof(int)/NumProcs, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Loc_B_Kernel_2, h_B_Location, K1R*sizeof(int)/NumProcs, cudaMemcpyHostToDevice) );
    
	float Start_Kernel_2_CPU, End_Kernel_2_CPU;
	if (MyProc==0)
	{
		Start_Kernel_2_CPU =	cutGetTimerValue(hTimer);
		Kernel_2_CPU(h_A,h_B,h_Max_CPU_All,h_A_Location_All, h_B_Location_All, 
					Kernel_2_output_A_CPU, Kernel_2_output_B_CPU, 
			         si, dis, Gop, Gex, Kerene2Max, K2R);
		End_Kernel_2_CPU   =	cutGetTimerValue(hTimer);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	float Start_Kernel_2_GPU =	cutGetTimerValue(hTimer);
	Kernel_2 <<<GridSize,BlockSize>>> (d_F2,d_H2,d_E2,d_FLoc2,d_ELoc2,d_L2,d_A, d_B, d_Max_Kernel_2, d_Loc_A_Kernel_2, d_Loc_B_Kernel_2,
									   d_Kernel_2_output_A,d_Kernel_2_output_B,
									   K2R/NumProcs, Kerene2Max, si, dis, Gop, Gex);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	CUT_CHECK_ERROR("Kernel 2");
	MPI_Barrier(MPI_COMM_WORLD);
	float End_Kernel_2_GPU =	cutGetTimerValue(hTimer);

	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_2_output_A, d_Kernel_2_output_A,   DATA_SZ_K21, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_Kernel_2_output_B, d_Kernel_2_output_B,   DATA_SZ_K21, cudaMemcpyDeviceToHost) );

	
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather( h_Kernel_2_output_A, DATA_GATHER_K2_MPI, MPI_INT, h_Kernel_2_output_A_All, DATA_GATHER_K2_MPI, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_Kernel_2_output_B, DATA_GATHER_K2_MPI, MPI_INT, h_Kernel_2_output_B_All, DATA_GATHER_K2_MPI, MPI_INT, 0, MPI_COMM_WORLD); 
	

	MPI_Barrier(MPI_COMM_WORLD);
	
	if (MyProc ==0)
	{
		printf("CPU execution time for Kernel 2 :   %f ms\n", End_Kernel_2_CPU -Start_Kernel_2_CPU);
		printf("GPU execution time for Kernel 2 :   %f ms\n", End_Kernel_2_GPU -Start_Kernel_2_GPU);
		printf("Speedup:                        :   %f   \n", (End_Kernel_2_CPU -Start_Kernel_2_CPU)/(End_Kernel_2_GPU -Start_Kernel_2_GPU));
		
		unsigned int result_regtest_K2A = cutComparei( Kernel_2_output_A_CPU, h_Kernel_2_output_A_All, K2R*(Kerene2Max+1));
		unsigned int result_regtest_K2B = cutComparei( Kernel_2_output_B_CPU, h_Kernel_2_output_B_All, K2R*(Kerene2Max+1));

		printf("Verification of Kernel 2        :   %s   \n", ((1 == result_regtest_K2A) && (1 == result_regtest_K2B)) ? "PASSED" : "FAILED");
		printf("-------------------------------------------------------------\n");
	
	    for( int i = 0; i < K2R*(Kerene2Max+1); i+=Kerene2Max)
		{
			 if ((Kernel_2_output_A_CPU[i]!=h_Kernel_2_output_A_All[i]) || (Kernel_2_output_B_CPU[i]!=h_Kernel_2_output_B_All[i]))
		//	 if ((0!=h_Kernel_2_output_A_All[i]) || (0!=h_Kernel_2_output_B_All[i]))
			{
				 printf(" %i           %i         %i     %i       %i      %i   \n", i/Kerene2Max, i%128, Kernel_2_output_A_CPU[i], h_Kernel_2_output_A_All[i], Kernel_2_output_B_CPU[i],h_Kernel_2_output_B_All[i]);

   			 }
		}
/*
		 int cnt = 0;
		 int K3_L_K2 = 0;
		 for( int i = 0; i < K2R*(Kerene2Max+1)*(Kerene2Max+1); i+=Kerene2Max)
		 {
			 if ((h_Kernel_2_output_A[i]!=0) || (h_Kernel_2_output_A[i+1]!=0))
			 {
				 cnt++;
				 int Temp = 1;
				 int k=i;
 //  				 printf(" No.  Seq. Len.      Seq. A       Seq. B     A     B    Location \n");

				 while (Temp!=0)     
				{
 // 					 printf(" %i      %i           %i         %i       %i    %i      %i   \n", cnt,k-i+1, h_Kernel_2_output_A[k], h_Kernel_2_output_B[k], h_A[h_Kernel_2_output_A[k]],h_B[h_Kernel_2_output_B[k]],i);
	//   				 printf(" %i      %i           %i         %i       %i    %i      %i   \n", cnt,k-i+1, h_Kernel_2_output_A[k], h_Kernel_2_output_B[k], Kernel_2_output_A_CPU[k],Kernel_2_output_B_CPU[k],i);
					k++;
   					 Temp=h_Kernel_2_output_A[k];
				} 
				K3_L_K2 = max(K3_L_K2,k-i+1); 
 //  				 printf("--------------------------------------------------------- \n");
   			 }
		}
*/
	}
	free(h_F2);
	free(h_Kernel_2_output_A);
	free(h_Kernel_2_output_B);
	free(h_Max_CPU);
	free(h_A_Location);
	free(h_B_Location);

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
    
    CUT_CHECK_ERROR("Kernel 2");
}
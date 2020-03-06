#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>


#include <Scan_SW.cu>  
#include <ss.cu>
#include <SW_kernel_1.cu>
#include <Kernel_1_CPU.cu>
#include <Kernel_1_Max_CPU.cu>
#include <Kernel_1_Max_CPU_MPI.cu>

void Pairwise_Alignment (int *h_A, int *d_B, int *h_B, int *h_Max_CPU, int *h_A_Location, int *h_B_Location,
						 int K1R, int MyProc, int NumProcs, int L_B, int L_A, int L_A1,  
						 int si, int dis, int Gop, int Gex, int L_M, int Threads_N, int Start_A,
						 int DATA_SZ_K1, int DATA_SZ_A, int DATA_SZ_H, int DATA_SZ_B, int DATA_SZ_M, int DATA_SZ_L)
{
	int	*h_H, *h_Max_GPU, *h_Loc_GPU, *h_Con_New, *h_Loc_GPU2, *h_Max_GPU2,  *Max_CPU;
	int *d_H, *d_F, *d_E_til, *d_H_til, *d_Max_H, *d_Loc_H, *d_Con_Old,  *d_Con_New, *d_Max_H2, *d_Loc_H2;
	int *d_A;
	int *A_Location_CPU, *B_Location_CPU;
	float Total_CPU, Total_GPU, Total_Copy;
	Total_CPU = 0;
	Total_GPU = 0;
	Total_Copy = 0;

	A_Location_CPU    = (int *)malloc(DATA_SZ_K1); 
	B_Location_CPU    = (int *)malloc(DATA_SZ_K1); 

    unsigned int timerGPU;
    CUT_SAFE_CALL(cutCreateTimer(&timerGPU));

	if (MyProc == 0)
	{
		printf("...allocating CPU memory.\n");
	}
	h_H               = (int *)malloc(DATA_SZ_H);
	Max_CPU           = (int *)malloc(DATA_SZ_K1);
	h_Max_GPU         = (int *)malloc(DATA_SZ_A);
	h_Max_GPU2        = (int *)malloc(DATA_SZ_A);
	h_Loc_GPU2        = (int *)malloc(DATA_SZ_A);
	h_Loc_GPU         = (int *)malloc(DATA_SZ_A);
	h_Con_New         = (int *)malloc(sizeof(int));

	if (MyProc == 0)
	{
		printf("...allocating GPU memory.\n");
	}
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_A,       DATA_SZ_A) );                //////////////
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_H,       DATA_SZ_H) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_F,       DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_E_til,   DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_H_til,   DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Con_Old, DATA_SZ_M) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Con_New, DATA_SZ_M) );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_H,   DATA_SZ_L) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_H,   DATA_SZ_L) );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_H2,  DATA_SZ_A/4) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_H2,  DATA_SZ_A/4) );
    

/*	h_A[0]=2;       h_A[1]=1;        h_A[2]=3;        h_A[3]=2;       h_A[4]=2;
	h_A[5]=4;       h_A[6]=2;        h_A[7]=3;        h_A[8]=2;       h_A[9]=4;
	h_A[10]=4;      h_A[11]=1;       h_A[12]=3;

	h_B[0]=1;       h_B[1]=1;        h_B[2]=4;        h_B[3]=3;       h_B[4]=2;
	h_B[5]=2;       h_B[6]=1;        h_B[7]=4;        h_B[8]=4;       h_B[9]=3;
	h_B[10]=1;      h_B[11]=2;       h_B[12]=3;       h_B[13]=3;                       
*/                            
    for(int i = 0; i < 2*(L_A+1) ; i++)
    {
		h_H[i]=0;
    }  
    for(int i = 0; i < (L_A) ; i++)
    {
		h_Max_GPU[i]=0;
		h_Loc_GPU[i]=0;
		h_Max_GPU2[i]=0;
    }  
 
    for(int i = 0; i < K1R ; i++)
    {
		h_A_Location[i]=0;
		h_B_Location[i]=0;
		A_Location_CPU[i]=0;
		B_Location_CPU[i]=0;
		Max_CPU[i]=1;
		h_Max_CPU[i]=1;
	}

	if (MyProc == 0)
	{
		printf("...copying input data to GPU memory \n");
	}
    
//	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, DATA_SZ_A1, cudaMemcpyHostToDevice) );                   //////////////////

	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, DATA_SZ_A, cudaMemcpyHostToDevice) );                   //////////////////
    CUDA_SAFE_CALL( cudaMemcpy(d_F, h_H, DATA_SZ_A, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_H, h_H, DATA_SZ_H, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, DATA_SZ_B, cudaMemcpyHostToDevice) );

	if (MyProc == 0)
	{
		printf("Data initialization done.\n");
		printf("-------------------------------------------------------------\n");
	}
	
	unsigned int hTimer;
	dim3 BlockSize(64, 1);         //64
    dim3 GridSize (512, 1);        // 512

	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );

	//************************************* SW in CPU ***********************************
	float Start_CPU_SW_Time=0;
	float End_CPU_SW_Time=0;
	if ((MyProc == 0) )
	{
//	printf("------------------------------------- \n");
//	printf("CPU execution started...\n");

	Start_CPU_SW_Time =	cutGetTimerValue(hTimer);
	Kernel_1_CPU(h_A,h_B,Max_CPU,A_Location_CPU, B_Location_CPU, K1R,L_A1,L_B,si, dis, Gop, Gex);  					  
	End_CPU_SW_Time =	cutGetTimerValue(hTimer);
	
//	printf("CPU execution finished...\n");
//   	printf("------------------------------------- \n");
	}
//************************************************************************************      
/*
             int *H_Full_CPU;
			H_Full_CPU               = (int *)malloc((L_A+1)*(L_B+1)*sizeof(int));
	float Start_CPU_Full_SW_Time =	cutGetTimerValue(hTimer);
	Full_SW_CPU(h_A, h_B, H_Full_CPU, L_A, L_B, si,dis, Gop, Gex);
	float End_CPU_Full_SW_Time =	cutGetTimerValue(hTimer);
	printf("CPU execution time:   %f ms\n", End_CPU_Full_SW_Time-Start_CPU_Full_SW_Time);

    for( unsigned int i = 0; i < (L_A+1)*(L_B+1); i++)
     {
     	 if (H_Full_CPU[i]>8)
     	 printf(" %i     %i      \n", i, H_Full_CPU[i]);
     } 
   	 printf("-------------------------------- \n");

					  

	if (MyProc == 0)
	{
		printf("GPU execution started...\n");
	}
*/	
	MPI_Barrier(MPI_COMM_WORLD);

	preallocBlockSums(L_A);
	preallocBlockSums_ss(L_M);
	float Start_Time_GPU =	cutGetTimerValue(hTimer);

//------------------------------------------------------------------------- Kernel 1 START -------------------------------------------------------------------------------------------
	for (int i = 0; i < L_B; i++)
	{
   		float Start_Time_Step =	cutGetTimerValue(hTimer);
			
		float Start_Time_H_Tilda =	cutGetTimerValue(hTimer);
//        H_tilda<<<GridSize,BlockSize>>>(d_H, d_A, d_F, d_H_til,d_E_til,h_B[i], L_A,si,dis, Gop, Gex, Threads_N, Start_A);
		H_tilda<<<GridSize,BlockSize>>>(d_H, d_A, d_F, d_H_til,d_E_til,h_B[i], L_A,si,dis, Gop, Gex, Threads_N, 0);
	
//		CUDA_SAFE_CALL( cudaThreadSynchronize() );                       
		float End_Time_H_Tilda=	cutGetTimerValue(hTimer);


	//******* SCAN START ********
	    float Start_Time_Scan = cutGetTimerValue(hTimer);
		// run once to remove startup overhead
		prescanArray(d_E_til, d_H_til,L_A);

		// Run the prescan
		prescanArray(d_E_til, d_H_til,L_A);
	
//	    CUDA_SAFE_CALL( cudaThreadSynchronize() ); 
	    float End_Time_Scan = cutGetTimerValue(hTimer);																	
	//******* SCAN FINISH ********
 
	    
	    int Mimimum_Kernel1=h_Max_CPU[0];

	    float Start_Time_H_Final = cutGetTimerValue(hTimer);
//		Final_H<<<GridSize,BlockSize>>> (d_A, d_H_til, d_E_til, d_H, d_Max_H, d_Loc_H ,d_Con_Old, d_Con_New, h_B[i], L_A, L_M, Gop, Gex,Threads_N, Mimimum_Kernel1,Start_A);
		Final_H<<<GridSize,BlockSize>>> (d_A, d_H_til, d_E_til, d_H, d_Max_H, d_Loc_H ,d_Con_Old, d_Con_New, h_B[i], L_A, L_M, Gop, Gex,Threads_N, Mimimum_Kernel1,0);
	
//		CUDA_SAFE_CALL( cudaThreadSynchronize() );                       
	    float End_Time_H_Final = cutGetTimerValue(hTimer);
//			MPI_Barrier(MPI_COMM_WORLD);



	    float Start_Time_Shrink_H =	cutGetTimerValue(hTimer);
		sum_scan(d_Con_New,d_Con_Old,L_M);

		Shrink_H <<<GridSize,BlockSize>>>(d_Max_H2, d_Loc_H2, d_Max_H, d_Loc_H,d_Con_Old,d_Con_New, L_A, L_M, Threads_N, i);

//		CUDA_SAFE_CALL( cudaThreadSynchronize() );                       
		float End_Time_Shrink_H = cutGetTimerValue(hTimer);
//		MPI_Barrier(MPI_COMM_WORLD);

		CUDA_SAFE_CALL( cudaMemcpy(h_Con_New, d_Con_New + L_M - 1, sizeof(int), cudaMemcpyDeviceToHost) );
		int number=h_Con_New[0];
		int DATA_SZ_COPY = number * sizeof(int);
		CUDA_SAFE_CALL( cudaMemcpy(h_Max_GPU2, d_Max_H2  , DATA_SZ_COPY, cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaMemcpy(h_Loc_GPU2, d_Loc_H2,   DATA_SZ_COPY, cudaMemcpyDeviceToHost) );

     // 		if (i==3) for( int qw = 0; qw < number; ++qw)     {	 printf(" %i     %i             %i \n", i, qw, h_Max_GPU2[qw]);	} 

	    float Start_Time_200_Best_Match =	cutGetTimerValue(hTimer);
		Kernel_1_Max_CPU(h_Max_GPU2,h_Max_CPU, h_A_Location, h_B_Location, h_Loc_GPU2, K1R,  number, i);  
	    float End_Time_200_Best_Match =	cutGetTimerValue(hTimer);

  	//	if (i==3) for( int qw = 0; qw < K1R; ++qw)     {	 printf(" %i     %i             %i \n", i, qw, h_Max_CPU[qw]);	} 
		
   		float End_Time_Step =	cutGetTimerValue(hTimer);
		
		float H_Tilda_Time = End_Time_H_Tilda-Start_Time_H_Tilda;
		float Scan_Time = End_Time_Scan-Start_Time_Scan;
		float H_Final_Time = End_Time_H_Final-Start_Time_H_Final;
		float H_Shrink_Time = End_Time_Shrink_H-Start_Time_Shrink_H;
		float CPU_200_Time = End_Time_200_Best_Match-Start_Time_200_Best_Match;
		float Copy_Time = End_Time_Step-Start_Time_Step-(End_Time_200_Best_Match-Start_Time_200_Best_Match)-
   			   		     (End_Time_Shrink_H-Start_Time_Shrink_H)-(End_Time_H_Final-Start_Time_H_Final)-
   						 (End_Time_Scan-Start_Time_Scan)-(End_Time_H_Tilda-Start_Time_H_Tilda);
//		float Step_Time = End_Time_Step-Start_Time_Step;
		Total_CPU = Total_CPU + CPU_200_Time;
		Total_GPU = Total_GPU + H_Tilda_Time + Scan_Time + H_Final_Time + H_Shrink_Time;
		Total_Copy = Total_Copy + Copy_Time;

/*	    printf(" %i  \n", i);
		printf("Time for H-Tilda:      %f ms,  %i percent\n", H_Tilda_Time, int((H_Tilda_Time/Step_Time)*100));
	    printf("Scan Time:             %f ms,  %i percent\n", Scan_Time   , int((Scan_Time/Step_Time)*100));
	    printf("Time for Final H:      %f ms,  %i percent\n", H_Final_Time, int((H_Final_Time/Step_Time)*100)); 	    
   	    printf("Time for Shrink H:     %f ms,  %i percent\n", H_Shrink_Time, int((H_Shrink_Time/Step_Time)*100)); 	    
	    printf("CPU 200 Best Match:    %f ms,  %i percent\n", CPU_200_Time, int((CPU_200_Time/Step_Time)*100)); 
   		printf("Copy Time from H-D-H:  %f ms,  %i percent\n", Copy_Time,int((Copy_Time/Step_Time)*100)); 
   																		   
	    printf("Time for Each Step:    %f ms\n", Step_Time); 
    	printf("---------------------------------------- \n");		                     */
	
	}
//	printf("CPU or GPU ID:   %i, CPU Time:  %f ms,  GPU Time:  %f ms, CPU & GPU Time:  %f ms\n", MyProc, Total_CPU, Total_GPU, Total_Copy); 
	for (int i=0; i<K1R; i++)
		h_A_Location[i] +=Start_A;
// ---------------------------------  gathering Data form other PCs ---------------------------------------------------------	 
	MPI_Barrier(MPI_COMM_WORLD);
	
	int *h_Max_All_CPU, *h_A_Location_All_CPU, *h_B_Location_All_CPU;

	int DATA_SZ_MPI_K1   =  NumProcs*K1R*sizeof(int);
	int Number = NumProcs*K1R;

	h_Max_All_CPU        = (int *)malloc(DATA_SZ_MPI_K1);
	h_A_Location_All_CPU = (int *)malloc(DATA_SZ_MPI_K1);
	h_B_Location_All_CPU = (int *)malloc(DATA_SZ_MPI_K1);

	MPI_Gather( h_Max_CPU   , K1R, MPI_INT, h_Max_All_CPU       , K1R, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_A_Location, K1R, MPI_INT, h_A_Location_All_CPU, K1R, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_B_Location, K1R, MPI_INT, h_B_Location_All_CPU, K1R, MPI_INT, 0, MPI_COMM_WORLD); 
	if (MyProc==0)
	{
		Kernel_1_Max_CPU_MPI(h_Max_All_CPU, h_A_Location_All_CPU, h_B_Location_All_CPU, 
							 h_Max_CPU    , h_A_Location        , h_B_Location,
							 K1R          , Number);

	//	for( int i = 0; i < K1R; ++i)     {	 printf(" %i     %i       %i      %i \n", i, Max_CPU[i],h_Max_CPU[i],h_B_Location[i]);	} 


	 
	//	 printf("GPU execution finished...\n");
		printf("CPU execution time for Kernel 1 :   %f ms \n", End_CPU_SW_Time-Start_CPU_SW_Time);
		printf("GPU execution time for Kernel 1 :   %f ms \n", cutGetTimerValue(hTimer)-Start_Time_GPU);
		printf("Speedup:                        :   %f    \n", (End_CPU_SW_Time-Start_CPU_SW_Time)/(cutGetTimerValue(hTimer)-Start_Time_GPU));

		unsigned int result_regtest_K1 = cutComparei( Max_CPU, h_Max_CPU, K1R);
		printf("Verification of Kernel 1        :   %s    \n", (1 == result_regtest_K1) ? "PASSED" : "FAILED");

		printf("-------------------------------------------------------------\n");
	}	
	
	CUDA_SAFE_CALL(cudaFree(d_H));
	CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_F));
    CUDA_SAFE_CALL(cudaFree(d_E_til));
    CUDA_SAFE_CALL(cudaFree(d_H_til));
    CUDA_SAFE_CALL(cudaFree(d_Max_H));
    CUDA_SAFE_CALL(cudaFree(d_Loc_H));
	CUDA_SAFE_CALL(cudaFree(d_Con_Old));
	CUDA_SAFE_CALL(cudaFree(d_Con_New));
 	CUDA_SAFE_CALL(cudaFree(d_Max_H2));
	CUDA_SAFE_CALL(cudaFree(d_Loc_H2));


	deallocBlockSums();    
	deallocBlockSums_ss();    
  

	free(h_H);
	free(h_Max_GPU);
	free(h_Loc_GPU);   
	free(h_Con_New);
	free(h_Loc_GPU2);
	free(h_Max_GPU2);
	free(Max_CPU);
	free(A_Location_CPU); 
	free(B_Location_CPU);

}
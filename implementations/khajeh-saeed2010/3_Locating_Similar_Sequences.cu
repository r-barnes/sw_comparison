#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>


#include <SW_kernel_3.cu>

void Locating_Similar_Sequences (int *h_A, int *d_A, int *h_Kernel_2_output_B, int *h_B, 
								 int *h_Val_K3_K4_B_All, int *h_Length_Seq_K4_All, int *d_Val_K3_K4_B_All, int *d_Length_Seq_K4_All,
								 int L_A, int L_A1, int K2R, int Kerene2Max,
								 int si, int dis, int Gop, int Gex, int K3_Timed_Out, int Start_A,
								 int K_3_R, int MyProc, int K3_Report, int K3_Safety, int NumProcs, int K3_Length, int K3R, int BlockSize_K3_X, int GridSize_K3_X)
{
	unsigned int hTimer;
	dim3 BlockSize_K3(BlockSize_K3_X, 1);         //64
	dim3 GridSize_K3 (GridSize_K3_X, 1);        // 512

	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );


	int *d_F3, *d_H3, *d_E3, *d_FLoc3, *d_ELoc3, *d_L3, *d_Max_Kernel_3, *d_Loc_Kernel_3_A, *d_Loc_Kernel_3_B, *d_B_K3, *d_Length_B_K3, *d_Num_Resize, *d_Min_Val_K3, *d_Min_Loc_K3;
	int *h_Max_Kernel_3, *h_Loc_Kernel_3_A, *h_Loc_Kernel_3_B, *h_B_K3, *h_Length_B_K3, *h_HEF_Kernel_3, *h_jEnd;
	int *d_jStart, *d_jEnd;

	int *Max_Kernel_3_CPU, *Loc_Kernel_3_A_CPU, *Loc_Kernel_3_B_CPU, *Max_K3_K4_CPU, *Loc_K3_K4_A_CPU , *Loc_K3_K4_B_CPU, *Length_Seq_K4_CPU;

    int DATA_SZ_K31 = K3_Safety * (BlockSize_K3.x+1) * (BlockSize_K3.x+1)       * K_3_R * sizeof(int);
    int DATA_SZ_K32 = K3_Safety *  K3_Length         *  BlockSize_K3.x    * K3R * K_3_R * sizeof(int);
	int DATA_SZ_K33 =                                   BlockSize_K3.x    * K3R * K_3_R * sizeof(int);
	int DATA_SZ_K34 =              K3_Length                                    * K_3_R * sizeof(int);
	int DATA_SZ_K35 =                                                             K_3_R * sizeof(int);
	
	int DATA_SZ_K31_CPU = K3_Safety *  K3_Length  * K3_Report * K_3_R * NumProcs * sizeof(int);
	int DATA_SZ_K32_CPU =                           K3_Report * K_3_R * NumProcs * sizeof(int);

	h_HEF_Kernel_3   = (int *)malloc(DATA_SZ_K31);
	h_Loc_Kernel_3_A = (int *)malloc(DATA_SZ_K32);
	h_Loc_Kernel_3_B = (int *)malloc(DATA_SZ_K32);
	h_Max_Kernel_3   = (int *)malloc(DATA_SZ_K33);
	h_B_K3           = (int *)malloc(DATA_SZ_K34);
	h_Length_B_K3    = (int *)malloc(DATA_SZ_K35);
	h_jEnd           = (int *)malloc(DATA_SZ_K35);
	
	
	Loc_Kernel_3_A_CPU = (int *)malloc(DATA_SZ_K31_CPU); 
	Loc_Kernel_3_B_CPU = (int *)malloc(DATA_SZ_K31_CPU);
	Loc_K3_K4_A_CPU    = (int *)malloc(DATA_SZ_K31_CPU); 
	Loc_K3_K4_B_CPU    = (int *)malloc(DATA_SZ_K31_CPU);

	Max_Kernel_3_CPU   = (int *)malloc(DATA_SZ_K32_CPU);
	Max_K3_K4_CPU      = (int *)malloc(DATA_SZ_K32_CPU);
	Length_Seq_K4_CPU  = (int *)malloc(DATA_SZ_K32_CPU);

	
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_F3,				 DATA_SZ_K31) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_E3,				 DATA_SZ_K31) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_H3,				 DATA_SZ_K31) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_FLoc3,			 DATA_SZ_K31) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_ELoc3,		     DATA_SZ_K31) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_L3,			     DATA_SZ_K31) );
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_Kernel_3_A,     DATA_SZ_K32) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_Kernel_3_B,     DATA_SZ_K32) );
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_Kernel_3,     DATA_SZ_K33) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Min_Val_K3 ,      DATA_SZ_K33) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Min_Loc_K3 ,      DATA_SZ_K33) );
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_B_K3,             DATA_SZ_K34) );
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Length_B_K3,      DATA_SZ_K35) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Num_Resize ,      DATA_SZ_K35) );
	
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_jStart ,      DATA_SZ_K35) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_jEnd ,      DATA_SZ_K35) );
	

	for( int i = 0; i < DATA_SZ_K31/4 ; i++)     
		h_HEF_Kernel_3[i]=0;

    for( int i = 0; i < DATA_SZ_K32/4 ; i++)     
   		h_Loc_Kernel_3_A[i]=0;

	for( int i = 0; i < DATA_SZ_K33/4 ; i++)     
   		h_Max_Kernel_3[i]=0;

    for( int i = 0; i < DATA_SZ_K34/4 ; i++)     
   		h_B_K3[i]=0;

    for( int i = 0; i < DATA_SZ_K35/4 ; i++)     
   		h_Length_B_K3[i]=0;
  
  	CUDA_SAFE_CALL( cudaMemcpy(d_Num_Resize ,   h_Length_B_K3,     DATA_SZ_K35,  cudaMemcpyHostToDevice) );

  
	// Preparing small Sequences for Kernel 3
	int K2_K3 = 0;
	int *h_B_K3_All, *h_Length_B_K3_All;
	
	h_B_K3_All           = (int *)malloc(DATA_SZ_K34*NumProcs);
	h_Length_B_K3_All    = (int *)malloc(DATA_SZ_K35*NumProcs);

	for( int i = 0; i < DATA_SZ_K34/4 ; i++)     
   		h_B_K3_All[i]=0;
	for( int i = 0; i < DATA_SZ_K35/4 ; i++)     
   		h_Length_B_K3_All[i]=0;
	

	if (MyProc == 0)
	{
		for( int i = 0; i < K2R*(Kerene2Max); i+=Kerene2Max)    ///////////
		{

			if (((h_Kernel_2_output_B[i]!=0) || ((h_Kernel_2_output_B[i+1]!=0))) && (K2_K3<(K_3_R)))
			{
				 int Temp = 1;
				 int kk=i;
   				 int Num = 0;
				 while (Temp!=0)     
				{
   					 int index1 = h_Kernel_2_output_B[kk]  ;
   					 int index2 = h_Kernel_2_output_B[kk+1];
   					 h_B_K3_All[Num+K2_K3*K3_Length]=h_B[index1];  
   					 Num++;
   					 kk++;
					 while (index1==index2)      // Remove Gap
					{
						index2 = h_Kernel_2_output_B[kk+1];
						kk++;
//						printf("************** Gap ******************   %i \n",  K2_K3+1);

  					 } 
   					 Temp=h_Kernel_2_output_B[kk];
				} 
					
				h_Length_B_K3_All[K2_K3]=Num;
				
				K2_K3++;
				
				if (Num>=K3_Length)
				{
					 printf("************** WARNING ****************** \n");
					 printf("The size of K3_Length should be increased \n");
					 printf(" %i     %i      \n", Num,K3_Length);
 					 printf("************** WARNING ****************** \n");
 				}
  			}
  		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

 

	MPI_Bcast( h_B_K3_All       , K3_Length * K_3_R , MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast( h_Length_B_K3_All,             K_3_R , MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast( &K2_K3, 1, MPI_INT, 0, MPI_COMM_WORLD);

	CUDA_SAFE_CALL( cudaMemcpy(d_F3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_H3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_E3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_FLoc3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_ELoc3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_L3,			h_HEF_Kernel_3,	   DATA_SZ_K31,  cudaMemcpyHostToDevice) );
				
	CUDA_SAFE_CALL( cudaMemcpy(d_Loc_Kernel_3_A,  h_Loc_Kernel_3_A,    DATA_SZ_K32,  cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_Loc_Kernel_3_B,  h_Loc_Kernel_3_A,    DATA_SZ_K32,  cudaMemcpyHostToDevice) );
	
	CUDA_SAFE_CALL( cudaMemcpy(d_Max_Kernel_3,  h_Max_Kernel_3,    DATA_SZ_K33,  cudaMemcpyHostToDevice) );
   	CUDA_SAFE_CALL( cudaMemcpy(d_Min_Val_K3 ,   h_Max_Kernel_3,    DATA_SZ_K33,  cudaMemcpyHostToDevice) );
  	CUDA_SAFE_CALL( cudaMemcpy(d_Min_Loc_K3 ,   h_Max_Kernel_3,    DATA_SZ_K33,  cudaMemcpyHostToDevice) );

	CUDA_SAFE_CALL( cudaMemcpy(d_B_K3,          h_B_K3_All,            DATA_SZ_K34,  cudaMemcpyHostToDevice) );
	
  	CUDA_SAFE_CALL( cudaMemcpy(d_Length_B_K3,   h_Length_B_K3_All,     DATA_SZ_K35,  cudaMemcpyHostToDevice) );

	CUT_CHECK_ERROR("Kernel 3");
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
    
 //   printf("************ Start Kernel 3 ************* \n");
 
 
  	for( int i = 0; i < DATA_SZ_K35/4 ; i++)     
   		h_jEnd[i]= min(BlockSize_K3.x,L_A);

   	CUDA_SAFE_CALL( cudaMemcpy(d_jStart ,   h_HEF_Kernel_3,     DATA_SZ_K35,  cudaMemcpyHostToDevice) );        ///////////
  	CUDA_SAFE_CALL( cudaMemcpy(d_jEnd ,   h_jEnd,     DATA_SZ_K35,  cudaMemcpyHostToDevice) );        ///////////

 	int j_Start_Section=0;
  	int j_End_Section = K3_Timed_Out;

	float Start_Kernel_3_CPU, End_Kernel_3_CPU;

	if ((MyProc == 0))
	{
		Start_Kernel_3_CPU =	cutGetTimerValue(hTimer);

		Kernel_3_CPU(h_A, h_B_K3_All, Max_Kernel_3_CPU, Loc_Kernel_3_A_CPU, Loc_Kernel_3_B_CPU, h_Length_B_K3_All,
				 Max_K3_K4_CPU, Loc_K3_K4_A_CPU , Loc_K3_K4_B_CPU, Length_Seq_K4_CPU,
		         K3_Length, K3_Report, K3_Safety,K_3_R,L_A1, si, dis, Gop, Gex,BlockSize_K3.x);   	      
		End_Kernel_3_CPU =	cutGetTimerValue(hTimer);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	
	

  	float Start_Kernel_3 =	cutGetTimerValue(hTimer);
  	
	int K3_End = ((L_A-1)/(BlockSize_K3.x)+1)/K3_Timed_Out + 1;          
	
  	for (int iii = 0; iii<K3_End; iii++ )
	{
		float S_Kernel_3 =	cutGetTimerValue(hTimer);

		Kernel_3 <<<GridSize_K3,BlockSize_K3>>> (d_A, d_B_K3, d_F3, d_H3, d_E3, d_FLoc3, d_ELoc3, d_L3, d_Max_Kernel_3,d_Loc_Kernel_3_A, d_Loc_Kernel_3_B, d_Length_B_K3,
							   d_Num_Resize,d_Min_Val_K3,d_Min_Loc_K3, d_jStart, d_jEnd,
							   K3_Length, K3R, K3_Safety, K_3_R,L_A, si, dis, Gop, Gex, 15,j_Start_Section,j_End_Section);      
						
		CUDA_SAFE_CALL( cudaThreadSynchronize() );
    	float E_Kernel_3 =	cutGetTimerValue(hTimer);

		j_Start_Section+=K3_Timed_Out;
		j_End_Section +=K3_Timed_Out;    					  	
							
	}
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
	MPI_Barrier(MPI_COMM_WORLD);
	float End_Kernel_3 =	cutGetTimerValue(hTimer);
	CUT_CHECK_ERROR("Kernel 3");
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int *d_Loc_K3_K4_A,*d_Loc_K3_K4_B, *d_Max_K3_K4, *d_Length_Seq_K4;
	int *h_Loc_K3_K4_A,*h_Loc_K3_K4_B, *h_Max_K3_K4, *h_Length_Seq_K4;

	int *h_Loc_K3_K4_A_MPI, *h_Loc_K3_K4_B_MPI, *h_Max_K3_K4_MPI, *h_Length_Seq_K4_MPI;

	int DATA_SZ_K37 = K3_Safety *  K3_Length  *  K3_Report * K_3_R * sizeof(int);
	int DATA_SZ_K38 = K3_Report * K_3_R * sizeof(int);
    
    
	h_Loc_K3_K4_A       = (int *)malloc(DATA_SZ_K37);
	h_Loc_K3_K4_B       = (int *)malloc(DATA_SZ_K37);

	h_Max_K3_K4         = (int *)malloc(DATA_SZ_K38);
	h_Length_Seq_K4     = (int *)malloc(DATA_SZ_K38);
    
	h_Loc_K3_K4_A_MPI   = (int *)malloc(DATA_SZ_K37*NumProcs);
	h_Loc_K3_K4_B_MPI   = (int *)malloc(DATA_SZ_K37*NumProcs);

	h_Max_K3_K4_MPI     = (int *)malloc(DATA_SZ_K38*NumProcs);
	h_Length_Seq_K4_MPI = (int *)malloc(DATA_SZ_K38*NumProcs);

	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_K3_K4_A,	  DATA_SZ_K37) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_K3_K4_B,	  DATA_SZ_K37) );
	
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_K3_K4,	  DATA_SZ_K38) );
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_Length_Seq_K4, DATA_SZ_K38) );

	MPI_Barrier(MPI_COMM_WORLD);
  	float Start_Kernel_3_4 =	cutGetTimerValue(hTimer);

 	Kernel_3_4 <<<GridSize_K3,BlockSize_K3>>> (d_Loc_K3_K4_A, d_Loc_K3_K4_B, d_Max_K3_K4,d_Max_Kernel_3, d_Loc_Kernel_3_A, d_Loc_Kernel_3_B,d_Length_Seq_K4,
        					   K3_Length, K3R, K3_Safety, K_3_R, K3_Report, Start_A);
	CUDA_SAFE_CALL( cudaThreadSynchronize() );

	CUDA_SAFE_CALL( cudaMemcpy(h_Loc_K3_K4_A,    d_Loc_K3_K4_A,   DATA_SZ_K37, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_Loc_K3_K4_B,   d_Loc_K3_K4_B,   DATA_SZ_K37, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_Max_K3_K4,      d_Max_K3_K4,     DATA_SZ_K38, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_Length_Seq_K4,   d_Length_Seq_K4,     DATA_SZ_K38, cudaMemcpyDeviceToHost) );

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gather( h_Loc_K3_K4_A     , DATA_SZ_K37/4, MPI_INT, h_Loc_K3_K4_A_MPI       , DATA_SZ_K37/4, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather( h_Loc_K3_K4_B     , DATA_SZ_K37/4, MPI_INT, h_Loc_K3_K4_B_MPI       , DATA_SZ_K37/4, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather( h_Max_K3_K4       , DATA_SZ_K38/4, MPI_INT, h_Max_K3_K4_MPI         , DATA_SZ_K38/4, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather( h_Length_Seq_K4   , DATA_SZ_K38/4, MPI_INT, h_Length_Seq_K4_MPI     , DATA_SZ_K38/4, MPI_INT, 0, MPI_COMM_WORLD);


	int *h_Loc_K3_K4_A_All, *h_Loc_K3_K4_B_All, *h_Max_K3_K4_All;

	int DATA_SZ_K31_MPI     = K3_Safety *  K3_Length  *  K3_Report * K_3_R * sizeof(int);
	int DATA_SZ_K32_MPI     =                            K3_Report * K_3_R * sizeof(int);
	int DATA_GATHER_K32_MPI =                            K3_Report * K_3_R;

	h_Loc_K3_K4_A_All         = (int *)malloc(DATA_SZ_K31_MPI);
	h_Loc_K3_K4_B_All         = (int *)malloc(DATA_SZ_K31_MPI);
	h_Max_K3_K4_All           = (int *)malloc(DATA_SZ_K32_MPI);


	if (MyProc==0)
	{
		Kernel_3_Max_CPU_MPI(	h_Max_K3_K4_MPI, h_Length_Seq_K4_MPI, h_Loc_K3_K4_A_MPI, h_Loc_K3_K4_B_MPI,
 					h_Max_K3_K4_All, h_Length_Seq_K4_All, h_Loc_K3_K4_A_All, h_Loc_K3_K4_B_All,
 					K3_Report, K_3_R, K3_Length, K3_Safety, NumProcs);
	}
					   
   	float End_Kernel_3_4 =	cutGetTimerValue(hTimer);
	CUT_CHECK_ERROR("Kernel 3_4");
	MPI_Barrier(MPI_COMM_WORLD);
	if (MyProc==0)
	{
	//   printf("------------- Kernel 3-4 ------------------- \n");

		for ( int i = 0; i <DATA_GATHER_K32_MPI  ; i++)   //
		{
			for ( int j = i*K3_Safety*K3_Length; j<(i+1)*K3_Safety*K3_Length ; j++)
			{
				if (h_Loc_K3_K4_A_All[j]!=0)
				{
					 h_Val_K3_K4_B_All [j]=  h_B_K3_All[(i/K3_Report) * K3_Length + h_Loc_K3_K4_B_All[j]];
					 
					 //------------------- If someone needs to add gap to the sequence, this part should be changed 
					 // Ex. if h_Loc_K3_K4_B_All[j] = h_Loc_K3_K4_B_All[j+1] then do something
					 
					if ((h_Loc_K3_K4_B_All[j]==h_Loc_K3_K4_B_All[j-1]) && (j>(i*K3_Safety*K3_Length) ))
						h_Val_K3_K4_B_All [j-1] =666;
				}
				else
				{
					j = (i+1)*K3_Safety*K3_Length;
				}
   			} 	
   		}
	}
  
	MPI_Bcast( h_Val_K3_K4_B_All  , DATA_SZ_K31_MPI/sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);
  	MPI_Bcast( h_Length_Seq_K4_All, DATA_SZ_K32_MPI/sizeof(int), MPI_INT, 0, MPI_COMM_WORLD);

  	CUDA_SAFE_CALL( cudaMemcpy(d_Val_K3_K4_B_All   ,   h_Val_K3_K4_B_All,       DATA_SZ_K31_MPI,  cudaMemcpyHostToDevice) );        ///////////
  	CUDA_SAFE_CALL( cudaMemcpy(d_Length_Seq_K4_All ,   h_Length_Seq_K4_All,     DATA_SZ_K32_MPI,  cudaMemcpyHostToDevice) );        ///////////

	if (MyProc==0)
	{
		unsigned int result_regtest_K3    = cutComparei( Max_K3_K4_CPU, h_Max_K3_K4_All, K3_Report * K_3_R);

		printf("CPU execution time for Kernel 3 :   %f ms \n", End_Kernel_3_CPU -Start_Kernel_3_CPU);
		printf("GPU execution time for Kernel 3 :   %f ms \n", (End_Kernel_3 -Start_Kernel_3)+(End_Kernel_3_4 -Start_Kernel_3_4));
		printf("Speedup:                        :   %f    \n", (End_Kernel_3_CPU -Start_Kernel_3_CPU)/((End_Kernel_3 -Start_Kernel_3)+(End_Kernel_3_4 -Start_Kernel_3_4)));
		printf("Verification of Kernel 3        :   %s    \n", ((1 == result_regtest_K3) ? "PASSED" : "FAILED"));
		printf("-------------------------------------------------------------\n");
	}
	MPI_Barrier(MPI_COMM_WORLD);

	free(h_Max_Kernel_3);
	free(h_Loc_Kernel_3_A);
	free(h_Loc_Kernel_3_B);
	free(h_B_K3);
	free(h_Length_B_K3);
	free(h_HEF_Kernel_3);
	free(h_jEnd);
	free(h_Loc_K3_K4_A_MPI), 
	free(h_Loc_K3_K4_B_MPI); 
	free(h_Max_K3_K4_MPI); 
	free(h_Length_Seq_K4_MPI);
  	
  	CUDA_SAFE_CALL(cudaFree(d_F3));
	CUDA_SAFE_CALL(cudaFree(d_H3));
	CUDA_SAFE_CALL(cudaFree(d_E3));
	CUDA_SAFE_CALL(cudaFree(d_FLoc3));
	CUDA_SAFE_CALL(cudaFree(d_ELoc3));
	CUDA_SAFE_CALL(cudaFree(d_L3));
	CUDA_SAFE_CALL(cudaFree(d_Max_Kernel_3));
	CUDA_SAFE_CALL(cudaFree(d_Length_B_K3));
	CUDA_SAFE_CALL(cudaFree(d_Num_Resize));
	CUDA_SAFE_CALL(cudaFree(d_Min_Val_K3));
	CUDA_SAFE_CALL(cudaFree(d_Min_Loc_K3));
	CUDA_SAFE_CALL(cudaFree(d_jStart));
	CUDA_SAFE_CALL(cudaFree(d_jEnd));

}
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>




void Data_Generator (int *h_A, int *h_B, 
					 int MyProc, int NumProcs, int L_B, int& L_A, int L_A1, int L_Gap, int& L_M, int Threads_N, int& Start_A, int GridSize, int BlockSize,
					 int& DATA_SZ_A, int& DATA_SZ_H,  int& DATA_SZ_M, int& DATA_SZ_L)
{
	MPI_Status status;
	if (MyProc == 0)
	{
		printf("Input Data ...\n");
		printf("Number of Processors is    %i \n", NumProcs);
		printf("Length of Sequence of A is %i \n", L_A1);
		printf("Length of Sequence of B is %i \n", L_B);
//		printf("Block Per Grid is          %i \n", GridSize);
//		printf("Threads Per Block is       %i \n", BlockSize);
		printf("Maximum GPU Memory is      %i MB \n", int((10.5*((L_A1-2*L_Gap)/NumProcs+2*L_Gap)/1024. + L_B/1024.)/1024.0*sizeof(int)));
		printf("-------------------------------------------------------------\n");
		printf("Initializing data...\n");
		printf("...Generating input data in CPU memory\n");

		for(int i = 0; i < L_A1; i++)
		{
			h_A[i] = int ((rand() % 20)+1);
		}
			
		for(int i = 0; i < L_B; i++)
		{
			h_B[i] = (rand() % 20)+1;
		} 
	 }

	int Length = (L_A1-2*L_Gap)/NumProcs;
	L_A = Length + 2*L_Gap;
	Start_A = MyProc*Length;

	MPI_Bcast( h_B, L_B , MPI_INT, 0, MPI_COMM_WORLD);

	if (MyProc == 0)
	{
		for (int i=1; i<NumProcs; i++)
		{
			MPI_Send( h_A + i*Length, L_A, MPI_INT, i, 111, MPI_COMM_WORLD );
		}
	}
	else
	{
		MPI_Recv( h_A     , L_A, MPI_INT, 0, 111, MPI_COMM_WORLD, &status );		
	}
 
	DATA_SZ_A  =  L_A            * sizeof(int);
	DATA_SZ_H  = 2*(L_A+1)       * sizeof(int);

	if (L_A>Threads_N)
	{
		L_M = Threads_N + 1;
		DATA_SZ_L = (int ((L_A-1)/Threads_N+1))*Threads_N*sizeof(int); 
	}
	else
	{
		L_M = L_A +1 ; 
	    DATA_SZ_L =DATA_SZ_A; 
	}
	DATA_SZ_M = L_M * sizeof(int);
	
}


//	ofstream tests;
//	tests.open ("times.txt");

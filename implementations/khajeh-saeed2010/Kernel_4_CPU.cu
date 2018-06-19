///////	/////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>
#include <cmath>
using namespace std;


void Kernel_4_CPU (int* B, int* Kernel_4_output, int* Start_A, int* Start_B, int* Length_Seq_K4, 
							int K3_Length, int K3_Report, int K3_Safety, int K_3_R,
							int Start_Th1, int End_Th1,int K4_S1, int K4_S2,int K4_S3)
{
   	int *D;
	
	D = new int [K3_Safety*(K3_Length+1)*(K3_Length+1)];
	
	for (int i=0; i<(K3_Safety*(K3_Length+1)*(K3_Length+1)); i++)
		D[i]=0;

	for (int i=0; i<(K3_Length+1); i++)
	{
		D[i]=i*K4_S3;
		D[i*(K3_Length+1)]=i*K4_S3;
	}

	for(int Sub_Block =0; Sub_Block < K_3_R; Sub_Block ++)
	{
		for (int Sub_Thread=Start_Th1; Sub_Thread<End_Th1; Sub_Thread++)
		{			
			int A_Loc = Start_A[Sub_Thread] * K3_Safety * K3_Length + Sub_Block * K3_Safety * K3_Length * K3_Report;
			int B_Loc = Start_B[Sub_Thread] * K3_Safety * K3_Length + Sub_Block * K3_Safety * K3_Length * K3_Report;
			int End_A = Length_Seq_K4[Start_A[Sub_Thread] + Sub_Block * K3_Report];
			int End_B = Length_Seq_K4[Start_B[Sub_Thread] + Sub_Block * K3_Report];

			for (int i = 0; i<End_B; i++ )
			{
				for (int j = 0; j<End_A; j++) 
				{
					int D_Sim;
					int Num  =  i   *(K3_Length+1) + j ;
					int Num1 = (i+1)*(K3_Length+1) + j ;
											
					if (B[A_Loc + j]==B[B_Loc + i])
						D_Sim = D[Num]+K4_S1;
					else
						D_Sim=D[Num]+K4_S2;
					
					int F = D[Num+1] + K4_S3;
					int E = D[Num1]  + K4_S3; 
					
					D[Num1+1] = min(min(E,F),D_Sim);             
				}
			}
			int Index_1 = Sub_Thread + Sub_Block *  K3_Report * K3_Report;
			int Num1 = (End_B)*(K3_Length+1) + End_A-1 ; 
			Kernel_4_output[Index_1] = D[Num1+1];
		}
	}
}
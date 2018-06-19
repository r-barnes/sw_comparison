////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>
#include <cmath>
using namespace std;


void Kernel_1_CPU(int *A, int *B, int *Max_CPU, int *A_Location_CPU, int *B_Location_CPU, 
			int K1_Max_Report, int LA, int LB, int sim, int dissim, int Gop, int Gex)					  

{
	int i,j,S1,H1,EF, H_Max	;

	int *F, *E, *H;
	F = new int [LA];
	E = new int [LA+1];
	H = new int [(LA+1)*2];
	//Max_CPU = new int [K1_Max_Report];

	for (j=0; j<LA; j++){
	F[j]=0;
	E[j]=0;
	}
	E[LA+1]=0;


	for (j=0; j<(2*(LA+1)); j++)
	H[j]=0;

	for (i=0; i<LB; i++)
	{
		for (j=0; j<LA; j++)
		{
			if (A[j]==B[i])
			S1=H[j]+sim;
			else 
			S1=H[j]+dissim;
			H1=max(S1,0);
			F[j]=max(F[j]-Gex,H[j+1]-Gop);
			E[j+1]=max(E[j]-Gex,H[(LA+1)+j]-Gop);
			EF=max(F[j],E[j+1]);
			H[(LA+1)+j+1]=max(EF,H1);
		}
		for (j=1; j<(LA+1); j++)
		{
			H[j]=H[(LA+1)+j];
			H_Max = H[j];

			if ((H_Max>1) && (H_Max>Max_CPU[0]))
			{
				if (A[j-1]==B[i])
				{
					Max_CPU[0]=H_Max;
					A_Location_CPU[0]=j-1;
					B_Location_CPU[0]=i;

					for (int k=0; k<K1_Max_Report-1; k++)
					{
						if (Max_CPU[k]>Max_CPU[k+1])
						{		
							int temp1=Max_CPU[k+1];
							int temp2=A_Location_CPU[k+1];
							int temp3=B_Location_CPU[k+1];	
							Max_CPU[k+1]=Max_CPU[k];
							A_Location_CPU[k+1]=A_Location_CPU[k];
							B_Location_CPU[k+1]=B_Location_CPU[k];
							Max_CPU[k]=temp1;
							A_Location_CPU[k]=temp2;
							B_Location_CPU[k]=temp3;
						}
						else
						{
							k=K1_Max_Report;
						}
					}
				}
			}
		}
	}
	for (int i=0; i<(K1_Max_Report-1)/2 + 1 ; i++)
	{
		int Temp = Max_CPU[K1_Max_Report-i-1];
		Max_CPU[K1_Max_Report-i-1]=Max_CPU[i];
		Max_CPU[i]=Temp;
	}
}

/*	 printf("No. Sim. Val.  Seq. A    Seq. B \n");
	 printf("-------------------------------- \n");
    for( i = 1; i < (K1_Max_Report+1); ++i)     
    {
   		 printf(" %i     %i       %i       %i \n", i, Max_CPU[K1_Max_Report-i],int(fmod(1.0*End_Point[K1_Max_Report-i],(LA+1))), int(1.0*(End_Point[K1_Max_Report-i])/(LA+1)));
    }  */



//		cout<<A[j]<<"-----"<<B[i]<<"-----"<<F[j]<<"------"<<E[j]<<endl;	
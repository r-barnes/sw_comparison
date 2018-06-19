////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>
#include <cmath>
using namespace std;



void Full_SW_CPU(int *A, int *B, int *H, int LA, int LB, int sim, int dissim, int Gop, int Gex)
{
int S1,H1,EF;

int *F, *E;
F = new int [(LA+1)*(LB+1)];
E = new int [(LA+1)*(LB+1)];
//H = new int [(LA+1)*(LB+1)];
//Max_CPU = new int [K1_Max_Report];


for (int j=0; j<LA; j++){
F[j]=0;
E[j]=0;

}
E[LA+1]=0;

for (int j=0; j<((LA+1)*(LB+1)); j++)
H[j]=0;


int iStart,jStart;
int End_K=LA+LB-1;
int End_B=LB;
int End_A=LA;

for (int k = 0; k<End_K; ++k )
{
	if (k<End_A)
	{
		iStart=0;
		jStart=k;
	}
	else
	{
		iStart=iStart+1;
		jStart=jStart+1;
	}
	int iEnd = min(k+1,End_B);

	for (int i=iStart; i<iEnd; i++)
	{
			
		int j=jStart-i;
		if (A[j]==B[i])
			S1=H[(LA+1)*i+j]+sim;
		else 
			S1=H[(LA+1)*i+j]+dissim;
		
		H1=max(S1,0);
		F[(i+1)*(End_A+1)+j+1]=max(F[(i)*(End_A+1)+j+1]-Gex,H[(LA+1)*i+j+1]-Gop);
		E[(i+1)*(End_A+1)+j+1]=max(E[(i+1)*(End_A+1)+j]-Gex,H[(LA+1)*(i+1)+j]-Gop);
		EF=max(F[(i+1)*(End_A+1)+j+1],E[(i+1)*(End_A+1)+j+1]);
		H[(LA+1)*(i+1)+j+1]=max(EF,H1);
	}
}
	/*for (j=0; j<(LA+1); j++)
	{
		H[j]=H[(LA+1)+j];
//		cout<<i<<"-----"<<j<<"-----"<<H[j]<<endl;	
		H_Max[j]=H[j];

		if ((H_Max[j]>1) && (H_Max[j]>Max_CPU[0]))
		{
			Max_CPU[0]=H_Max[j];
			End_Point[0]=(LA+1)*(i+1)+j;
			
			for (int k=0; k<K1_Max_Report-1; k++)
			{
				if (Max_CPU[k]>Max_CPU[k+1])
				{		
					int temp1=Max_CPU[k+1];
					int temp2=End_Point[k+1];
					Max_CPU[k+1]=Max_CPU[k];
					End_Point[k+1]=End_Point[k];
					Max_CPU[k]=temp1;
					End_Point[k]=temp2;
				}
				else
				{
					k=K1_Max_Report;
				}
			}
		}
	}
}*/

/*	 printf("No. Sim. Val.  Seq. A    Seq. B \n");
	 printf("-------------------------------- \n");
    for( i = 1; i < (K1_Max_Report+1); ++i)     
    {
   		 printf(" %i     %i       %i       %i \n", i, Max_CPU[K1_Max_Report-i],int(fmod(1.0*End_Point[K1_Max_Report-i],(LA+1))), int(1.0*(End_Point[K1_Max_Report-i])/(LA+1)));
    }  */

}

//		cout<<A[j]<<"-----"<<B[i]<<"-----"<<F[j]<<"------"<<E[j]<<endl;	
////////////////////////////////////////////////////////////////////////////
// 
// 
////////////////////////////////////////////////////////////////////////////
 #include <iostream>
#include <cmath>
using namespace std;



void Kernel_2_CPU(int *A, int *B, int *Max_CPU, int *A_Location_CPU, int *B_Location_CPU,int *Kernel_2_output_A_CPU, int*Kernel_2_output_B_CPU, 
			int Sim_Exact, int Sim_Dissim, int GAP_START, int GAP_EXTEND, int Kerene2Max, int Kernel_2_Report)
{
	int *F, *E, *H, *L, *F_Loc, *E_Loc;
	
	F     = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	E     = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	H     = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	L     = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	F_Loc = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	E_Loc = new int [(Kerene2Max+1)*(Kerene2Max+1)];
	

	for(int Sub_Seq = 0; Sub_Seq < Kernel_2_Report; Sub_Seq++)
    {
		for (int j=0; j<(Kerene2Max+1)*(Kerene2Max+1); j++)
		{
			F[j]=0;
			E[j]=0;
			H[j]=0;
			L[j]=0;
			F_Loc[j]=0;
			E_Loc[j]=0;
		}
		int Sim_Val = Max_CPU[Sub_Seq];
        int Start_A = A_Location_CPU[Sub_Seq];
		int Start_B = B_Location_CPU[Sub_Seq];

		int End_A   = min  (Start_A+1  ,  Kerene2Max );
		int End_B   = min  (Start_B+1  ,  Kerene2Max );
		int End_K   = End_A + End_B -1;

		int iStart,k;
		
		for (k = 0; k<End_K; ++k)
		{
			if (k<End_A)
			{
				iStart=0;
			}
			else
			{
				iStart=iStart+1;
			}
			
			int iEnd = min(k+1,End_B);

			for (int i=iStart; i<iEnd; i++)
			{
				int j=k-i;
				int H_Sim=0;
				int Num  =  i   *(End_A+1)+j;
				int Num1 = (i+1)*(End_A+1)+j;
				int Temp[10];
				
				for (int n=0; n<9; n=n+2)
				{
					Temp[n] = -2;
				}
				
				if (A[Start_A-j]==B[Start_B-i])
				{
					H_Sim = H[Num]+Sim_Exact;
					Temp[0]= H_Sim;
					Temp[1]= Num;
				}
				else
				{
					H_Sim=H[Num]+Sim_Dissim;
					Temp[8]= H_Sim;
					Temp[9]= Num;
				}
				
				Temp[2]=0;
				Temp[3]= Num1+1;
				
				if ((H[Num+1]-GAP_START)>(F[Num+1]-GAP_EXTEND))
				{
					F[Num1+1] = H[Num+1]-GAP_START;
					Temp[4] = F[Num1+1];
					Temp[5] = Num+1;
					F_Loc[Num1+1] = Num+1;
				}
				else
				{
					F[Num1+1] = F[Num+1]-GAP_EXTEND;
					Temp[4]       = F[Num1+1];
					Temp[5]       = F_Loc[Num1+1-(End_A+1)]; 
					F_Loc[Num1+1] = F_Loc[Num1+1-(End_A+1)];
				}
				
				if ((H[Num1] -GAP_START)>(E[Num1] -GAP_EXTEND))
				{
					E[Num1+1] = H[Num1] -GAP_START;
					Temp[6] = E[Num1+1]; 
					Temp[7] = Num1;
					E_Loc[Num1+1] = Num1;
				}
				else
				{
					E[Num1+1]     = E[Num1] -GAP_EXTEND;
					Temp[6]       = E[Num1+1]; 
					Temp[7]       = E_Loc[Num1]; 
					E_Loc[Num1+1] = E_Loc[Num1];
				}	
				int maxH =-1;
				int MaxL = 0;
				for (int n=0; n<9; n=n+2)
				{
					if (Temp[n]>maxH)
					{
						maxH = Temp[n];
						MaxL = Temp[n+1];
					}
				}	
				H[Num1+1] = maxH;
				L[Num1+1] = MaxL;
				

				if (maxH==Sim_Val)
				{				
				//  stop the program and start to find the track back
					int Loc_Path = 0;
					int cnt = 0;
					int Loc_Temp = Num1+1;

					while   ((H[Loc_Temp] != 0) && (Loc_Temp != Loc_Path) && (cnt<Kerene2Max)) 
					{	
						int Loc_Temp1=Loc_Temp;
						int Remind = Loc_Temp1 - int (Loc_Temp1/(End_A+1))*(End_A+1)-1;
						Kernel_2_output_A_CPU[cnt+Kerene2Max*Sub_Seq] = A_Location_CPU[Sub_Seq]- Remind;
						Kernel_2_output_B_CPU[cnt+Kerene2Max*Sub_Seq] = B_Location_CPU[Sub_Seq]-int(Loc_Temp1/(End_A+1)-1);
						
						Loc_Path = Loc_Temp;
						Loc_Temp = L[Loc_Path];
						cnt++;
					}
			
					// Check for Start point & End Point
					if ((Kernel_2_output_A_CPU[cnt-1+Kerene2Max*Sub_Seq]==A_Location_CPU[Sub_Seq]) && 
						(Kernel_2_output_B_CPU[cnt-1+Kerene2Max*Sub_Seq]==B_Location_CPU[Sub_Seq]))
					{
						k=End_K;
						i=iEnd;
					}
					else
					{
						// Delete the results that is not correct
						for (int del=cnt-1; del>=0; del--)
						{
							Kernel_2_output_A_CPU[del+Kerene2Max*Sub_Seq] =0;
							Kernel_2_output_B_CPU[del+Kerene2Max*Sub_Seq] =0;
						}
					}
				}
			} // End i
		} // diagonal
	} // block
}
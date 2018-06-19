////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>
#include <cmath>
using namespace std;


void Kernel_3_CPU(int *A, int *B, int *Max_Kernel_3, int *Loc_Kernel_3_A_CPU, int *Loc_Kernel_3_B_CPU,int *L_B, 
				  int *Max_K3_K4, int *Loc_K3_K4_A , int *Loc_K3_K4_B, int *Length_Seq_K4,
			      int K3_Length,  int K3_Report, int K3_Safety, int K_3_R,
				  int LA, int Sim_Exact, int Sim_Dissim, int GAP_START, int GAP_EXTEND, int BlockDim)
{
	int *F, *E, *H, *L, *F_Loc, *E_Loc;
	
	F     = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];
	E     = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];
	H     = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];
	L     = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];
	F_Loc = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];
	E_Loc = new int [K3_Safety*(BlockDim+1)*(BlockDim+1)];

	for (int i = 0; i<(K3_Report*K_3_R); i++ )
		Max_Kernel_3[i]=0;

	for(int Sub_Seq = 0; Sub_Seq < K_3_R; Sub_Seq ++)
	{
		
		int LB = L_B[Sub_Seq];
		int End_B = LB;
		int jStart =0;
		int jEnd = BlockDim;
	    int Num_Resize=0;
		int Min_Val_K3 = 0;
		int Min_Loc_K3 = 0;

		for (int i = 0; i<(K3_Safety*(BlockDim+1)*(BlockDim+1)); i++ )
		{
			F[i]     = 0;
			E[i]     = 0;
			H[i]     = 0;
			L[i]     = 0;
			F_Loc[i] = 0;
			E_Loc[i] = 0;
		}
		for (int jj=0; jj<(((LA-1)/BlockDim)+1); jj++)
		{
			for (int i = 0; i<End_B; i++ )
			{
				for (int j=jStart; j<jEnd; j++)
				{
					int H_Sim = 0;
					int Num11  =  i   *(2*BlockDim) + j    ;
					int Num22  =  i   *(2*BlockDim) + j + 1;
					int Num33  = (i+1)*(2*BlockDim) + j    ;
					int Num44  = (i+1)*(2*BlockDim) + j + 1;
					
					if (j == 0)
					{
						Num11  =  i   *(2*BlockDim) + 2*BlockDim;
						Num22  =  i   *(2*BlockDim) +       1   ;
						Num33  = (i+1)*(2*BlockDim) + 2*BlockDim;
						Num44  = (i+1)*(2*BlockDim) +       1   ;
					}
					int Temp[10];
					for (int n=0; n<9; n=n+2)
					{
						Temp[n] = -2;
					}
					
					if (A[j+((Num_Resize/2)*2*BlockDim)]==B[i+Sub_Seq*K3_Length])
					{
						H_Sim = H[Num11]+Sim_Exact;
						Temp[0]= H_Sim;
						Temp[1]= Num11;
					}
					else
					{
						H_Sim=H[Num11]+Sim_Dissim;
						Temp[6]= H_Sim;
						Temp[7]= Num11;
					}
					
					Temp[2]=0;
					Temp[3]= Num44;
					
					if ((H[Num22]-GAP_START)>(F[Num22]-GAP_EXTEND))
					{
						F[Num44]     = H[Num22]-GAP_START;
						Temp[4]      = F[Num44];
						Temp[5]      = Num22;
						F_Loc[Num44] = Num22;
					}
					else
					{
						F[Num44]      = F[Num22]-GAP_EXTEND;
						Temp[4]       = F[Num44];
						Temp[5]       = F_Loc[Num44-(2*BlockDim)]; 
						F_Loc[Num44]  = F_Loc[Num44-(2*BlockDim)];
					}
					if ((H[Num33]-GAP_START)>(E[Num33]-GAP_EXTEND))
					{
						E[Num44]     = H[Num33]-GAP_START;
						Temp[8]      = E[Num44];
						Temp[9]      = Num33;
						E_Loc[Num44] = Num33;
					}
					else 
					{
						E[Num44]      = E[Num33]-GAP_EXTEND;
						Temp[8]       = E[Num44];
						Temp[9]       = E_Loc[Num33];
						E_Loc[Num44]  = E_Loc[Num33];
					}
					///////////////////////////
/*					int K3_Gap_Length = 8;
					int Max_Thread = -10;
					int End_K = j-K3_Gap_Length;
                           
					for (int k=j; k>End_K; k--)
					{
						int k1=k;
						if (k1<1)
							k1 = 2*BlockDim + k ;                    ///////////////////////////////////////////////
						
						int Num33 = (i+1)*(2*BlockDim) + k1;
//int fllkfmj = (i+1)*(2*blockDim.x) + k1;
						int Ga_Ex = GAP_EXTEND*(j-k+1)+(GAP_START-GAP_EXTEND);
						
						if ((H[Num33]-Ga_Ex)>Max_Thread)
						//if ((H_Hat[fllkfmj]-Ga_Ex)>Max_Thread)
						{
							Max_Thread    = H [Num33]-Ga_Ex;
							E	  [Num44] = Max_Thread;
							E_Loc [Num44] = Num33;
						}
					}
						Temp[8]       = E[Num44];
						Temp[9]       = E_Loc[Num33];

*/
					int maxH =-1;
					int maxL = 0;
					for (int n=0; n<9; n=n+2)
					{
						if (Temp[n]>maxH)
						{
							maxH = Temp[n];
							maxL = Temp[n+1];
						}
					}	
					H	 [Num44] = maxH;
					L	 [Num44] = maxL;

					if ((maxH>Min_Val_K3)  && (A[j+((Num_Resize/2)*2*BlockDim)]==B[i+Sub_Seq*K3_Length]))      /// 2 * Sub_Seq*BlockDim
					{
						Max_Kernel_3[Min_Loc_K3 + Sub_Seq*K3_Report]=maxH;
						int Loc_Path = 0;
						int cnt = 0;
						int corection = 0;
						int condition = 0;
						int Loc_Temp = Num44;
						while   ((H[Loc_Temp] != 0)  && (Loc_Temp != Loc_Path))  
						{						   		
							int Loc_Temp1=Loc_Temp;
							int Remind = Loc_Temp1 % (2*BlockDim) - 1;
							int Index = cnt+K3_Safety*K3_Length*Min_Loc_K3 + K3_Safety*K3_Length*K3_Report*Sub_Seq;
							Loc_Kernel_3_A_CPU[Index] = Remind+BlockDim*(Num_Resize/2)*2  - corection;
							Loc_Kernel_3_B_CPU[Index] = int(Loc_Temp1/(2*BlockDim)) - 1;
							
	//						if (condition ==1  && Sub_Seq==0)
	//							printf("   %i   %i    %i       %i   %i    %i\n", Loc_Temp1, Remind, Index, Loc_Kernel_3_A_CPU[Index], Loc_Kernel_3_B_CPU[Index], corection);
							if ((cnt>1) && ((Loc_Kernel_3_A_CPU[Index]-Loc_Kernel_3_A_CPU[Index-1])>40) && (condition ==0))    // 50 because of first row and second row
							{
								corection = 2*BlockDim;
								Loc_Kernel_3_A_CPU[Index] = Loc_Kernel_3_A_CPU[Index] - corection;
								Loc_Kernel_3_B_CPU[Index-1] = Loc_Kernel_3_B_CPU[Index-1] - 1;
								condition =1;
							}
							Loc_Path = Loc_Temp;
							Loc_Temp = L[Loc_Path];
							cnt++;
						}
						for (int m = cnt; m<(K3_Safety*K3_Length); m++)
						{
							int Index = m+K3_Safety*K3_Length*Min_Loc_K3+ K3_Safety*K3_Length*K3_Report*Sub_Seq;
							Loc_Kernel_3_A_CPU[Index] = 0;   
							Loc_Kernel_3_B_CPU[Index] = 0;   			
						 }         
				
						int Min_Local = 10000;
						for (int ij = 0; ij<K3_Report; ij++)
						{				
							if (Max_Kernel_3[ij + Sub_Seq*K3_Report]<Min_Local)
							{
								Min_Val_K3 = Max_Kernel_3[ij + Sub_Seq*K3_Report];
								Min_Loc_K3 = ij;
								Min_Local = Min_Val_K3;
							}
						}
					}
				}
			}
			if (Num_Resize % 2 ==0)
			{
				jStart = BlockDim;
				jEnd   = min(2*BlockDim,LA-(Num_Resize*BlockDim));
			}
			else
			{
				jStart = 0;
				jEnd   = min(  BlockDim,LA-(Num_Resize*BlockDim));
			}
			Num_Resize++;
		}
// sort the results for comparison with GPU results
		int Max_Thread = 0;
		int Max_Loc_Th;
		for (int k = 0; k<K3_Report; k++)
		{
			for (int j = Sub_Seq*K3_Report; j<((Sub_Seq+1)*K3_Report); j++)
			{
				if (Max_Kernel_3[j]>Max_Thread)
				{
					Max_Thread = Max_Kernel_3[j];
					Max_Loc_Th = j;
				}
			}
			Max_K3_K4[k + Sub_Seq*K3_Report] = Max_Thread;
			Max_Thread = 0;
			Max_Kernel_3[Max_Loc_Th] = 0;
			int cnt =0;
			int Length = 0;
			for (int i=Max_Loc_Th*K3_Length*K3_Safety;  i<((Max_Loc_Th+1)*K3_Length*K3_Safety);  i++ )
			{
				Loc_K3_K4_A[cnt + k * K3_Length * K3_Safety + Sub_Seq * K3_Length * K3_Safety * K3_Report]= Loc_Kernel_3_A_CPU[i];
				Loc_K3_K4_B[cnt + k * K3_Length * K3_Safety + Sub_Seq * K3_Length * K3_Safety * K3_Report]= Loc_Kernel_3_B_CPU[i];
				if ((Loc_Kernel_3_B_CPU[i]==0) && (Loc_Kernel_3_A_CPU[i]==0) && (Length ==0) )    //// Check IF Loc_Kernel_3_B[i]==0) && (Loc_Kernel_3_A[i]==0 and still one point of seq.
					Length = cnt;

				cnt++;
			}	
			Length_Seq_K4[k + Sub_Seq*K3_Report] = Length;
		}	// End Sort
	}
}
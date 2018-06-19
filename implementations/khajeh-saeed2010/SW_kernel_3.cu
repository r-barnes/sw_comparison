
__global__ void Kernel_3 (int* A, int* B,int* F,int* H,int* E,int* F_Loc,int* E_Loc,int* L,int* Max_Kernel_3, int* Loc_Kernel_3_A,int* Loc_Kernel_3_B,int* L_B,
						  int* Num_Resize1, int* Min_Val_K3,int* Min_Loc_K3, int* jStart1, int* jEnd1, 
						  int K3_Length, int K3R, int K3_Safety, int K_3_R,
                          int LA, int Sim_Exact, int Sim_Dissim, int GAP_START, int GAP_EXTEND,int K3_Gap_Length, int jj_Start, int jjEnd )
{
   #define Local_Size_3 512
    
	__shared__ int H_Hat		[Local_Size_3];
	__shared__ int jStart		[Local_Size_3];
	__shared__ int jEnd		    [Local_Size_3]; 
	__shared__ int Num_Resize	[Local_Size_3];

	for(int Sub_Seq = blockIdx.x; Sub_Seq < K_3_R; Sub_Seq += gridDim.x)
	{
		
	
		int LB = L_B[Sub_Seq];
		int End_B = LB;
		
		int jj_End = min(jjEnd, (LA-1)/(blockDim.x)+1);               ///////////////

		jStart[Sub_Seq] = jStart1[Sub_Seq];
		jEnd [Sub_Seq] = jEnd1 [Sub_Seq];		
		Num_Resize[Sub_Seq] = Num_Resize1[Sub_Seq];

		for (int jj=jj_Start; jj<jj_End; jj++)
		{	
			for (int i = 0; i<End_B; i++ )
			{
				for (int j=jStart[Sub_Seq]+threadIdx.x; j<jEnd[Sub_Seq]; j+=blockDim.x)
				{
					int H_Sim = 0;
					int Num11  =  i   *(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
					int Num22  =  i   *(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1) + 1;
					int Num44  = (i+1)*(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1) + 1;
					
					if (j == 0)
					{
						Num11  =  i   *(2*blockDim.x) + 2*blockDim.x  + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
						Num22  =  i   *(2*blockDim.x) +       1       + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
						Num44  = (i+1)*(2*blockDim.x) +       1       + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
					}
					int Temp[8];
//					for (int n=0; n<7; n=n+2)
					{
						Temp[0] = -2;
						Temp[6] = -2;
					}
					
					if (A[j+((Num_Resize[Sub_Seq]/2)*2*blockDim.x)]==B[i+Sub_Seq*K3_Length])
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
						Temp[5]       = F_Loc[Num44-(2*blockDim.x)]; 
						F_Loc[Num44]  = F_Loc[Num44-(2*blockDim.x)];
					}
					int maxH =-1;
					int maxL = 0;
					for (int n=0; n<7; n=n+2)
					{
						if (Temp[n]>maxH)
						{
							maxH = Temp[n];
							maxL = Temp[n+1];
						}
					}	
					H_Hat[j] = maxH;
					H[Num44] = maxH;
					L[Num44] = maxL;
				}
				__syncthreads();
                    
				for (int j=jStart[Sub_Seq]+threadIdx.x; j<jEnd[Sub_Seq]; j+=blockDim.x)
				{
					int Num44 = (i+1)*(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1) + 1;
					
					int Max_Thread = -100;
					int End_K = j-K3_Gap_Length;
                           
					for (int k=j; k>End_K; k--)
					{
						int k1=k;
						if (k1<1)
							k1 = 2*blockDim.x + k ;                    ///////////////////////////////////////////////
						
						int Num33 = (i+1)*(2*blockDim.x) + k1 + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
						int Ga_Ex = GAP_EXTEND*(j-k+1)+(GAP_START-GAP_EXTEND);
						
						if ((H[Num33]-Ga_Ex) >= Max_Thread)
						{
							Max_Thread    = H [Num33]-Ga_Ex;
							E	  [Num44] = Max_Thread;
							E_Loc [Num44] = Num33;
						}
					}
				}
				__syncthreads();           
				
				for (int j=jStart[Sub_Seq]+threadIdx.x; j<jEnd[Sub_Seq]; j+=blockDim.x)
				{
					int Num44 = (i+1)*(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1) + 1;					
					if (E[Num44]>H [Num44])
					{
						H [Num44] = E     [Num44];
						L [Num44] = E_Loc [Num44];
						H_Hat[j]  = H     [Num44];
					}
				}
				__syncthreads();

				for (int j=jStart[Sub_Seq]+threadIdx.x; j<jEnd[Sub_Seq]; j+=blockDim.x)
				{
					if ((H_Hat[j]>Min_Val_K3[(j % blockDim.x) + Sub_Seq*blockDim.x])  && (A[j+((Num_Resize[Sub_Seq]/2)*2*blockDim.x)]==B[i+Sub_Seq*K3_Length]))      /// 2 * Sub_Seq*blockDim.x
					{
						Max_Kernel_3[Min_Loc_K3[(j % blockDim.x) + Sub_Seq*blockDim.x]+ (j % blockDim.x)*K3R + Sub_Seq*blockDim.x*K3R]=H_Hat[j];
						int Num44 = (i+1)*(2*blockDim.x) + j + Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1) + 1;
						int Loc_Path = 0;
						int cnt = 0;
						int corection = 0;
						int condition = 0;
						int Loc_Temp = Num44;
						while   ((H[Loc_Temp] != 0)  && (Loc_Temp != Loc_Path))  
						{						   		
							int Loc_Temp1=Loc_Temp-Sub_Seq*K3_Safety*(blockDim.x+1)*(blockDim.x+1);
							int Remind = Loc_Temp1 % (2*blockDim.x) - 1;
							int Index = cnt+K3_Safety*K3_Length*(Min_Loc_K3[(j % blockDim.x) + Sub_Seq*blockDim.x])+K3_Safety*K3_Length*K3R*(j % blockDim.x) + K3_Safety*K3_Length*K3R*blockDim.x*Sub_Seq;
							Loc_Kernel_3_A[Index] = Remind+blockDim.x*(Num_Resize[Sub_Seq]/2)*2  - corection;
							Loc_Kernel_3_B[Index] = int(Loc_Temp1/(2*blockDim.x)) - 1;
							
							if ((cnt>1) && ((Loc_Kernel_3_A[Index]-Loc_Kernel_3_A[Index-1])>50) && (condition ==0))
							{
								corection = 2*blockDim.x;
								Loc_Kernel_3_A[Index] = Loc_Kernel_3_A[Index] - corection;
								Loc_Kernel_3_B[Index-1] = Loc_Kernel_3_B[Index-1] - 1;
								condition =1;
							}
							Loc_Path = Loc_Temp;
							Loc_Temp = L[Loc_Path];
							cnt++;
						}
						for (int m = cnt; m<(K3_Safety*K3_Length); m++)
						{
							int Index = m+K3_Safety*K3_Length*(Min_Loc_K3[(j % blockDim.x) + Sub_Seq*blockDim.x])+K3_Safety*K3_Length*K3R*(j % blockDim.x) + K3_Safety*K3_Length*K3R*blockDim.x*Sub_Seq;
							Loc_Kernel_3_A[Index] = 0;   
							Loc_Kernel_3_B[Index] = 0;   			
						}
			
						int Min_Local = 10000;
						for (int ij = 0; ij<K3R; ij++)
						{
							if (Max_Kernel_3[ij+(j % blockDim.x)*K3R + Sub_Seq*blockDim.x*K3R]<Min_Local)
							{
								Min_Val_K3[(j % blockDim.x) + Sub_Seq*blockDim.x] = Max_Kernel_3[ij+(j % blockDim.x)*K3R + Sub_Seq*blockDim.x*K3R];
								Min_Loc_K3[(j % blockDim.x) + Sub_Seq*blockDim.x] = ij;
								Min_Local = Max_Kernel_3[ij+(j % blockDim.x)*K3R + Sub_Seq*blockDim.x*K3R];
							}
						}
					}
				}
				__syncthreads();       
			}
			__syncthreads();
			
			if (threadIdx.x == 0)
			{
				if (Num_Resize[Sub_Seq] % 2 ==0)
				{
					jStart[Sub_Seq] = blockDim.x;
					jEnd[Sub_Seq]   = min(2*blockDim.x,LA-(Num_Resize[Sub_Seq]*blockDim.x));
				}
				else
				{
					jStart[Sub_Seq] = 0;
					jEnd[Sub_Seq]   = min(  blockDim.x,LA-(Num_Resize[Sub_Seq]*blockDim.x));
				}
				Num_Resize[Sub_Seq]=Num_Resize[Sub_Seq]+1;
			}
			__syncthreads();
		}
		__syncthreads();
	
		Num_Resize1[Sub_Seq] = Num_Resize[Sub_Seq];
		jStart1[Sub_Seq] =jStart[Sub_Seq];
		jEnd1[Sub_Seq]   = jEnd[Sub_Seq];
	}
	__syncthreads();
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


__global__ void Kernel_3_4 (int* Loc_K3_K4_A, int* Loc_K3_K4_B, int* Max_K3_K4,
						   int* Max_Kernel_3, int* Loc_Kernel_3_A,int* Loc_Kernel_3_B, int* Length_Seq_K4,
						   int K3_Length, int K3R, int K3_Safety, int K_3_R, int K3_Report, int Start_A)
{
	for(int Sub_Seq = blockIdx.x; Sub_Seq < K_3_R; Sub_Seq += gridDim.x)
	{	
		int Start_thread =  Sub_Seq    * blockDim.x * K3R;
		int End_Thread   = (Sub_Seq+1) * blockDim.x * K3R;
		int Max_Thread = 0;
		int Max_Loc_Th = 0;
		
		for (int k = 0; k<K3_Report; k++)
		{
			if (threadIdx.x == 0 )
			{
				for (int j=Start_thread; j<End_Thread; j++)
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
					Loc_K3_K4_A[cnt + k * K3_Length * K3_Safety + Sub_Seq * K3_Length * K3_Safety * K3_Report]= Loc_Kernel_3_A[i]+Start_A;
					Loc_K3_K4_B[cnt + k * K3_Length * K3_Safety + Sub_Seq * K3_Length * K3_Safety * K3_Report]= Loc_Kernel_3_B[i];
					if ((Loc_Kernel_3_B[i]==0) && (Loc_Kernel_3_A[i]==0) && (Length ==0) )    //// Check IF Loc_Kernel_3_B[i]==0) && (Loc_Kernel_3_A[i]==0 and still one point of seq.
						Length = cnt;

					cnt++;
				}	
				Length_Seq_K4[k + Sub_Seq*K3_Report] = Length;
			}
		}
	}
}
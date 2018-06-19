
__global__ void Kernel_2 (int* F,int* H,int* E,int* F_Loc,int* E_Loc,int* L,int* A, int* B, int* Max_H, int* Max_Loc_A, int* Max_Loc_B, 
						  int* Kernel_2_output_A, int* Kernel_2_output_B,
                          int Kernel_2_Report, int Kerene2Max, int Sim_Exact, int Sim_Dissim, int GAP_START, int GAP_EXTEND, int Start_A_Seq)

{
	#define Local_Size_2 128
	__shared__ int H_Max_Block  [Local_Size_2];
	__shared__ int H_Loc_Block  [Local_Size_2];

    for(int Sub_Seq = blockIdx.x; Sub_Seq < Kernel_2_Report; Sub_Seq += gridDim.x)
    {
   		int Sim_Val = Max_H[Sub_Seq];
        	int Start_A = Max_Loc_A[Sub_Seq]-Start_A_Seq;
		int Start_B = Max_Loc_B[Sub_Seq];
		int End_A   = min  (Start_A+1  ,  Kerene2Max );
		int End_B   = min  (Start_B+1  ,  Kerene2Max );
		int End_K   = End_A + End_B -1;

		int iStart,k;
		

		for (k = 0; k<End_K; ++k )
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
			
			__syncthreads();
			
			for (int i=iStart+threadIdx.x; i<iEnd; i+=blockDim.x)
			{
				__syncthreads();
				int j=k-i;
				int H_Sim=0;
				int Num  =  i   *(End_A+1)+j+Sub_Seq*(Kerene2Max+1)*(Kerene2Max+1);
				int Num1 = (i+1)*(End_A+1)+j+Sub_Seq*(Kerene2Max+1)*(Kerene2Max+1);
				int Temp[10];
				
				for (int n=0; n<9; n=n+2)
				{
					Temp[n] = -2;
				}
				
				if (A[Start_A-j]==B[Start_B-i])
		//						if (A[j]==B[i])
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
//				Kernel_2_output[Num1+1] = maxH;	
				
				if (maxH==Sim_Val)
				{
					H_Max_Block[i] = maxH;
					H_Loc_Block[i] = Num1+1;
				}
				else
				{
					H_Max_Block[i] = 0;
					H_Loc_Block[i] = 0;			
				}			
			}				
			__syncthreads();
			
			if (threadIdx.x == 0)
			{
				for (int n=iStart; n<iEnd; n++)
				{
					if (H_Max_Block[n]==Sim_Val) 
					{
					//  stop the program and start to find the track back
						int Loc_Path = 0;
						int cnt = 0;
						int Loc_Temp = H_Loc_Block[n];

						while   ((H[Loc_Temp] != 0) && (Loc_Temp != Loc_Path) && (cnt<Kerene2Max)) 
						{
							int Loc_Temp1=Loc_Temp-Sub_Seq*(Kerene2Max+1)*(Kerene2Max+1);
							int Remind = Loc_Temp1 - int (Loc_Temp1/(End_A+1))*(End_A+1)-1;
//							Kernel_2_output[cnt+Kerene2Max*Sub_Seq] = Max_Loc[K1R-Sub_Seq-1]-int(Loc_Temp1/(End_A+1)-1)*(LA+1)- Remind;
							Kernel_2_output_A[cnt+Kerene2Max*Sub_Seq] = Max_Loc_A[Sub_Seq]- Remind;
							Kernel_2_output_B[cnt+Kerene2Max*Sub_Seq] = Max_Loc_B[Sub_Seq]-int(Loc_Temp1/(End_A+1)-1);
							
							Loc_Path = Loc_Temp;
							Loc_Temp = L[Loc_Path];
							cnt++;
						}
						// Check for Start point & End Point
						if (((Kernel_2_output_A[cnt-1+Kerene2Max*Sub_Seq])==Max_Loc_A[Sub_Seq]) && (Kernel_2_output_B[cnt-1+Kerene2Max*Sub_Seq]==Max_Loc_B[Sub_Seq]))
						{
							k=End_K;
							n=iEnd;
						}
						else
						{
							// Delete the results that is not correct
							for (int del=cnt-1; del>=0; del--)
							{
								Kernel_2_output_A[del+Kerene2Max*Sub_Seq] =0;
								Kernel_2_output_B[del+Kerene2Max*Sub_Seq] =0;
							}
						}
					}
				}
			}
			__syncthreads();
			
		} // diagonal
		__syncthreads();

	} // block
	__syncthreads();
}




/*
		Kernel_2_output[0]=End_K;
		Kernel_2_output[1]=End_A;
		Kernel_2_output[2]=End_B;
		Kernel_2_output[3]=Sim_Val;
		Kernel_2_output[4]=Start_A;
		Kernel_2_output[5]=LA;
		Kernel_2_output[6]=Max_Loc[K1R-Sub_Seq-1] ;
		Kernel_2_output[7]= 377344547 - int (377344547/(LA+1))*(LA+1) ;
		
		
				Kernel_2_output[0]=End_K;
		Kernel_2_output[1]=End_A;
		Kernel_2_output[2]=End_B;
		Kernel_2_output[3]=Sim_Val;
		Kernel_2_output[4]=Start_A;
		Kernel_2_output[5]=Start_B;
		Kernel_2_output[6]=Sub_Seq ;
		Kernel_2_output[7]= Max_Loc[K1R-Sub_Seq-1] ;
		Kernel_2_output[8]=0   *(End_A+1)+0+Sub_Seq*Kerene2Max*Kerene2Max;


		Kernel_2_output[0]=End_K;
		Kernel_2_output[1]=End_A;
		Kernel_2_output[2]=End_B;
		Kernel_2_output[3]=Sim_Val;
		Kernel_2_output[4]=Start_A;
		Kernel_2_output[5]=Start_B;
		Kernel_2_output[6]=Sub_Seq ;
		Kernel_2_output[7]= Max_Loc[K1R-Sub_Seq-1] ;
		Kernel_2_output[8]=0   *(End_A+1)+0+Sub_Seq*Kerene2Max*Kerene2Max;
		Kernel_2_output[9]=Kernel_2_Report;


*/                          
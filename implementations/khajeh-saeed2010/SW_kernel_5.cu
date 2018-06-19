//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------- Kernel 51 ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void Kernel_51 (int* Kernel_4_output, int* Kernel_5_Temp, int K3_Report, int K5R)
{
   
    if (blockIdx.x == 0 )   // Converting symetric matrix to full matrix to find Center
	{
		for (int Sub_Thread=threadIdx.x; Sub_Thread<K5R; Sub_Thread+=blockDim.x)
		{
			
			int counter = Sub_Thread*K3_Report*K3_Report;
			int Loc = counter;
			for (int i=0; i<K3_Report; i++)
			{
				for (int j=i+1; j<K3_Report; j++)
				{
					Kernel_5_Temp [i*K3_Report + j + Loc] = Kernel_4_output[counter];
					Kernel_5_Temp [j*K3_Report + i + Loc] = Kernel_4_output[counter];
					counter++;
				}
			}
		}
	}
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------- Kernel 52 ---------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------
__global__ void Kernel_52 (int* B, int* Kernel_5_output, int* Length_Seq_K4, int* Kernel_5_Temp,
							int K3_Length, int K3_Report, int K3_Safety, int K5R, int MyProc,
							int K5_Length, int K4_S1, int K4_S2,int K4_S3)
{
   
   #define Local_Size_51 128     // K5_Report = 128 = K3_Report
   #define Local_Size_52 32      // K5_Length = 32  = K3_Length       Max Number is 40-42
    
	__shared__ int Sum   			   [Local_Size_51];     // OK
	__shared__ int Center_Loc		   [1];
	__shared__ int Center_Seq		   [Local_Size_52];
	__shared__ int Center_Rev		   [Local_Size_52];
	__shared__ int Kernel_5_out_shared [Local_Size_52];
	__shared__ int D    			   [(Local_Size_52+1)*(Local_Size_52+1)];
	__shared__ int L    			   [(Local_Size_52+1)*(Local_Size_52+1)];
 
	
	// ---------------------------- finding summation of each row of matrix (minimum is a Center) --------------------------------
	for(int Sub_Block = (MyProc*K5R + blockIdx.x); Sub_Block < ((MyProc+1)*K5R); Sub_Block += gridDim.x)
	{
		for (int Sub_Thread=threadIdx.x; Sub_Thread<K3_Report; Sub_Thread+=blockDim.x)
		{
			int iStart = Sub_Thread * K3_Report + Sub_Block * K3_Report * K3_Report;
			Sum[Sub_Thread] = 0;
			for (int j=0; j<K3_Report; j++)
			{
				Sum[Sub_Thread] = Sum[Sub_Thread] + Kernel_5_Temp [j + iStart] ;
			}
		}
		__syncthreads();

		if (threadIdx.x == 0)    // Finding Minimum (Center Sequence)
		{
			int Minimum_Sum = 10000;
			for (int i=0; i<K3_Report; i++)
			{
				if (Sum[i]<Minimum_Sum)
				{
					Minimum_Sum   = Sum[i];
					Center_Loc[0] = i;
				}
			}
		}
		__syncthreads();
		//--------------------------------------------- Extract Center Sequence ----------------------------------
		for (int i=threadIdx.x; i<K3_Length; i+=blockDim.x)          
		{
			Center_Seq [i]= B[i + Center_Loc[0] * K3_Length * K3_Safety + Sub_Block * K3_Length * K3_Safety * K3_Report];
			Center_Rev[i]=0;
		}
		//--------------------------------------------- Initialize D & L arrays ----------------------------------
		L[0]=0;
		for (int i=threadIdx.x; i<K5_Length+1; i+=blockDim.x)
		{
			D[i] = i * K4_S3;
			D[i*(K5_Length+1)] = i * K4_S3;
			if (i>0)
				{
					L[i]           = i - 1;
					L[i*(K5_Length+1)] = (i-1)*(K5_Length+1);
				}
		}
		__syncthreads();
		//--------------------------------------------- Start Multiple Alignment  ----------------------------------
		if (threadIdx.x ==0)
		{
			int Length_Center_Seq = Length_Seq_K4[Center_Loc[0] + Sub_Block * K3_Report];
			int Temp[6];
			for (int k=0; k<K3_Report; k++)
			{
				int B_Loc = k * K3_Safety * K3_Length + Sub_Block * K3_Safety * K3_Length * K3_Report;
				int End_A = Length_Center_Seq;
				int End_B = Length_Seq_K4[k + Sub_Block * K3_Report];

				for (int i = 0; i<End_B; i++ )
				{
					for (int j = 0; j<End_A; j++) 
					{
						int D_Sim;
						int Num  =  i   *(K5_Length+1) + j ;
						int Num1 = (i+1)*(K5_Length+1) + j ;

						// First E then F then Similarity 	
						if (Center_Seq [j]==B[B_Loc + i])
							D_Sim = D[Num]+K4_S1;
						else
							D_Sim=D[Num]+K4_S2;
						
						Temp[4]= D_Sim;
						Temp[5]= Num;
						
						Temp[0] = D[Num1]  + K4_S3; 
						Temp[1] = Num1;

						Temp[2] = D[Num+1] + K4_S3;
						Temp[3] = Num+1;
							
						int minD =1000;
						int minL = 0;
						for (int n=0; n<6; n=n+2)
						{
							if (Temp[n]<minD)
							{
								minD = Temp[n];
								minL = Temp[n+1];
							}
						}	
						D[Num1+1] = minD;             
						L[Num1+1] = minL;
					}
				}

				//------------------------------------------------ Trace Back -----------------------------------------------------------------------
				int Loc_Temp = (End_B)*(K5_Length+1) + End_A; 
				int Loc_Path = Loc_Temp;
				int Check = 0;

				int Index_A = Length_Center_Seq-1;
				int Index_B = B_Loc + End_B-1;
				int cnt = 0;
				int update = 0;

				while   (Loc_Path != Check) 
				{						   
					Loc_Path = L[Loc_Temp];
					int Dif = Loc_Temp - Loc_Path;
					if (Dif==(K5_Length+1+1))
					{
						Center_Rev [cnt] = Center_Seq [Index_A];
						Kernel_5_output[cnt + K5_Length*k + K5_Length*K3_Report*Sub_Block] = B[Index_B];
						
						Index_B -=1;
						Index_A -=1;
					}
					else
					{				
						if (Dif==(K5_Length+1))
						{
							Center_Rev [cnt] = 666;
							Kernel_5_output[cnt + K5_Length*k + K5_Length*K3_Report*Sub_Block] = B[Index_B];
							Index_B -=1;
							update=1;
							// update previous sequences
							for (int j=0; j<k; j++)
							{
								for (int i=cnt; i<K5_Length; i++)
								{
									Kernel_5_out_shared[i] = Kernel_5_output[i + K5_Length*j + K5_Length*K3_Report*Sub_Block]; 
								}
								Kernel_5_output[cnt + K5_Length*j + K5_Length*K3_Report*Sub_Block] = 666;
								for (int i=cnt+1; i<K5_Length; i++)
								{
									Kernel_5_output[i + K5_Length*j + K5_Length*K3_Report*Sub_Block]  = Kernel_5_out_shared[i-1] ;
								}

							}
						}
						else
						{
							Center_Rev [cnt] = Center_Seq [Index_A];
							Kernel_5_output[cnt + K5_Length*k + K5_Length*K3_Report*Sub_Block] = 666;
							Index_A -=1;
						}
					}
					Loc_Temp = Loc_Path;
					cnt++;
				}       // While End
				if (update == 1)
				{
					Length_Center_Seq = cnt;
					for (int i=0; i<Length_Center_Seq; i++)
					{
						Center_Seq [i] = Center_Rev [Length_Center_Seq-i-1];
					}
				}
			} // End K (Report)
			//  Bring Center Sequence to the first of array
			for (int i=0; i<K5_Length; i++)
			{
				int Tmp = Kernel_5_output[i + K5_Length*0   + K5_Length*K3_Report*Sub_Block]; 
				Kernel_5_output[i + K5_Length*0             + K5_Length*K3_Report*Sub_Block] = Center_Rev [i]; 
				Kernel_5_output[i + K5_Length*Center_Loc[0] + K5_Length*K3_Report*Sub_Block] = Tmp;
			}

		} // End threadIdx.x 
		__syncthreads();
	}
}
//	__syncthreads();


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

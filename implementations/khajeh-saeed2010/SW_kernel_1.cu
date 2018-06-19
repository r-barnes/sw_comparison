__global__ void H_tilda (int* H, int* A,int* F,int* H_Til, int* E_Til,
                         int B, int LA, int Sim_Exact, int Sim_Dissim,
                         int GAP_START, int GAP_EXTEND, int threadsN, int Start_A)
{

	 #define Local_Size_1 256

	__shared__ int H_Sh		 [Local_Size_1+1];
	__shared__ int F_Sh		 [Local_Size_1+1];
	__shared__ int H_Til_Sh  [Local_Size_1+1];
	__shared__ int A_Sh      [Local_Size_1+1];


	for(int Sub_Seq = blockIdx.x; Sub_Seq < (int((LA-1)/Local_Size_1) +1); Sub_Seq += gridDim.x)   //(int((LA-1)/Local_Size_1) +1)
	{
		int Block_Start = Sub_Seq*Local_Size_1;
		int	Block_End = min ((Sub_Seq+1)*Local_Size_1 +1, LA);
		int Length = Block_End-Block_Start;

		// Copy from Global Memory to the Shared Memory
		for (int i=threadIdx.x; i<Length; i+=blockDim.x)
		{
			H_Sh[i]=H[i+Block_Start];
			F_Sh[i]=F[i+Block_Start];
			A_Sh[i]=A[i+Block_Start+Start_A];
		}
		__syncthreads();

		// Calculate H tilda and F values
		for (int i=threadIdx.x; i<Length-1; i+=blockDim.x)
		{
			int S;
			if (A_Sh[i]==B)  S = Sim_Exact;
			else  S = Sim_Dissim;

			F_Sh[i] = max(F_Sh[i]-GAP_EXTEND, H_Sh[i+1]-GAP_START);
			H_Til_Sh[i]=max(F_Sh[i],max(H_Sh[i]+S,0));
		}
		__syncthreads();

		// Write back to the Global memory
		for (int i=threadIdx.x; i<Length-1; i+=blockDim.x)
		{
			F    [i+Block_Start] = F_Sh[i];
			H_Til[i+Block_Start] = H_Til_Sh[i];
	//		E_Til[i+Block_Start] = H_Til_Sh[i];
		}
		__syncthreads();
	}
}

//------------------------------------------------------------------------------------------------------------------------------
__global__ void Final_H (int* A,int* H_tilda, int* E_tilda, int* H_Final, int* Max_H, int* Max_Loc,int* Max_Num1,int* Max_Num2, 
                          int B, int LA, int LM, int GAP_START, int GAP_EXTEND, int threadsN, int Minimum_H_Max, int Start_A)
{
 
 	int i, Jmax;
	Jmax=int((LA-1)/threadsN)+1;

	int myid = gridDim.x*blockIdx.y*blockDim.x*blockDim.y +
			   blockIdx.x*blockDim.x*blockDim.y +
			   blockDim.x*threadIdx.y +
			   threadIdx.x;
	
	int H_Loc;
	if (myid<LM-1)	
	{
		int cnt=myid*Jmax;
		for (i=myid; i<LA; i+=threadsN)
		{   
			H_Loc = max(E_tilda[i]-(GAP_START-GAP_EXTEND), H_tilda[i]);
			H_Final[i+1] =H_Loc; 
			if (H_Loc > Minimum_H_Max)
			{
				if (A[i+Start_A]==B)
				{
					Max_H[cnt] = H_Loc;
					Max_Loc[cnt] = i+1;
					cnt++; 
				}
			}
		}   
		cnt -= myid*Jmax;  //number of finds  
		Max_Num1[myid]=cnt;
		Max_Num2[myid]=cnt;
	}
}
//------------------------------------------------------------------------------------------------------------------------------
__global__ void Shrink_H (int* Max_H_New, int* Max_Loc_New, int* Max_H, int* Max_Loc,int* Max_Num1,int* Max_Num2, 
                          int LA, int LM, int threadsN, int row)
{
	int Jmax,i;
	Jmax=int((LA-1)/threadsN)+1;

	int myid = gridDim.x*blockIdx.y*blockDim.x*blockDim.y +
			   blockIdx.x*blockDim.x*blockDim.y +
			   blockDim.x*threadIdx.y +
			   threadIdx.x;
			   			   
	if (myid<LM-1)	
	{
		for (i=0; i<Max_Num1[myid]; ++i)
		{   
			Max_H_New  [i+Max_Num2[myid]] = Max_H[i+myid*Jmax];
//			Max_Loc_New[i+Max_Num2[myid]] = (LA+1)*(row+1)+Max_Loc[i+myid*Jmax];
			Max_Loc_New[i+Max_Num2[myid]] = Max_Loc[i+myid*Jmax]-1;
		}
	}
__syncthreads();

}                      
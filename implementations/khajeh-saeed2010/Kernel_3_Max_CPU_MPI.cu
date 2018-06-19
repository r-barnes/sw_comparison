////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>

//extern "C"
void Kernel_3_Max_CPU_MPI(int *Max_All, int *Length_Seq_K4_All, int *A_Location_All, int *B_Location_All, 
 			int *Max_K3 , int *Length_Seq_K4    , int *A_Location    , int *B_Location,
 			int K3_Report, int K_3_R, int K3_Length, int K3_Safety, int NumProcs)
{

	
	for (int k=0; k<K_3_R; k++)
	{
	//	if (k==2 || k==12 || k==86) for (int j=0; j<K3_Report; j++) for (int i=j+1; i<K3_Report; i++) if (A_Location_All[j*K3_Length*K3_Safety+k*K3_Report*K3_Length*K3_Safety]==A_Location_All[i*K3_Length*K3_Safety+k*K3_Report*K3_Length*K3_Safety]) printf("  %i    %i     %i     %i     %i   %i \n", k, j, i, Max_All[j+k*K3_Report],Length_Seq_K4_All[j+k*K3_Report], 
	//		A_Location_All[j*K3_Length*K3_Safety+k*K3_Report*K3_Length*K3_Safety] );
		int n=0;
		do 
		{
			int Max_Val_MPI = 0;
			int Max_Loc_MPI = 0;
			for (int j=0; j<NumProcs; j++)
			{
				int Start = j*K3_Report*K_3_R + k * K3_Report;
				int End   = Start + K3_Report;
				for (int i=Start; i<End; i++)
				{
					if (Max_Val_MPI<Max_All[i])
					{
						Max_Val_MPI = Max_All[i];
						Max_Loc_MPI = i;
					} 
				}
			}

			for (int m=0; m<n; m++)
			{
				if ((Max_K3[m+k*K3_Report]==Max_Val_MPI) && (Length_Seq_K4_All[Max_Loc_MPI]==Length_Seq_K4[m+k*K3_Report]) && 
					(A_Location_All[Max_Loc_MPI*K3_Length*K3_Safety]==A_Location[m*K3_Length*K3_Safety + k*K3_Report*K3_Length*K3_Safety]))
				{
					m=n;
					Max_All[Max_Loc_MPI]=0;
					Max_Val_MPI=0;
					n--;
	//				printf("-------------------------\n");
				}
			}
			if (Max_Val_MPI!=0)
			{
				Max_K3[n + k*K3_Report]=Max_Val_MPI;
				Max_All[Max_Loc_MPI]=0;
				Length_Seq_K4[n + k*K3_Report]=Length_Seq_K4_All[Max_Loc_MPI];
				for (int i=0; i<(K3_Length*K3_Safety); i++)
				{
					A_Location[i + n*K3_Length*K3_Safety + k*K3_Report*K3_Length*K3_Safety]=A_Location_All[i + Max_Loc_MPI*K3_Length*K3_Safety];
					B_Location[i + n*K3_Length*K3_Safety + k*K3_Report*K3_Length*K3_Safety]=B_Location_All[i + Max_Loc_MPI*K3_Length*K3_Safety];
				}
			}
			n++;
	//		printf("   %i     %i    %i    \n", n, k, Max_Val_MPI);
		} while (n<K3_Report);
	//	printf("-----------------------------------------------------------\n");
	}
}

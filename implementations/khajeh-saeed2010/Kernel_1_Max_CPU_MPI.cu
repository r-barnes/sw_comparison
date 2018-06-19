////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>


void Kernel_1_Max_CPU_MPI(int *Max_All_CPU, int *A_Location_All_CPU, int *B_Location_All_CPU, 
						  int *Max_K1     , int *A_Location_K1     , int *B_Location_K1,
						  int K1_Max_Report, int Number)
{
	for (int k=0; k<K1_Max_Report; k++)
	{
		int check=0;
		do
		{
		check=0;
			int Max_Val_MPI = 0;
		int Max_Loc_MPI = 0;
		for (int i=0; i<Number; i++)
		{
			if (Max_Val_MPI<Max_All_CPU[i])
			{
				Max_Val_MPI = Max_All_CPU[i];
				Max_Loc_MPI = i;
			}
		}
		
//		printf("-----  Ata ------    %i       %i       %i \n", k, Max_Val_MPI,Max_Loc_MPI);
		for (int i=0; i<k; i++)
		{
			if ((A_Location_K1[i]==A_Location_All_CPU[Max_Loc_MPI])  &&  (B_Location_K1 [i]==B_Location_All_CPU[Max_Loc_MPI]))
			{
				
				check=1;
				Max_All_CPU[Max_Loc_MPI]=0;
				
//				printf("-----     %i    %i   %i    %i   %i    %i \n", k, i, A_Location_K1[i],A_Location_All_CPU[Max_Loc_MPI], Max_K1[i],Max_Val_MPI);
				i=k+2*Number;
			}
		}
		if (check==0)
		{
			Max_K1[k]        = Max_Val_MPI;
	//		printf("------  %i\n", Max_Val_MPI);
			A_Location_K1 [k]= A_Location_All_CPU[Max_Loc_MPI];
			B_Location_K1 [k]= B_Location_All_CPU[Max_Loc_MPI];
			Max_All_CPU[Max_Loc_MPI]=0;
			Max_Val_MPI = 0;
		}
		}while (check!=0);
	}
}

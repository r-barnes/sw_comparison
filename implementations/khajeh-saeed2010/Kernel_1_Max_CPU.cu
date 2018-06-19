////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU.
// Straight accumulation in double precision.
////////////////////////////////////////////////////////////////////////////
 #include <iostream>


void Kernel_1_Max_CPU(int *Max_GPU, int *Max_CPU, int *A_Location, int *B_Location, int *Location,
					  int K1_Max_Report, int Number, int Row)
{
//	int ii=0;
	for (int i=0; i<Number; ++i)
//	do 
	{
//		int i=Location[ii];
		if ((Max_GPU[i]>1) && (Max_GPU[i]>Max_CPU[0]))
		{
			Max_CPU[0]=Max_GPU[i];

//			End_Point[0]=i+(LA+1)*(row+1)+1;
			A_Location[0]=Location[i];
			B_Location[0] = Row;
			for (int j=0; j<K1_Max_Report-1; j++)
			{
				if (Max_CPU[j]>Max_CPU[j+1])
				{		
//							printf("C++++ %i, %i \n", j, End_Point[j] );
					
					int temp1=Max_CPU[j+1];
					int temp2=A_Location[j+1];
					int temp3=B_Location[j+1];

					Max_CPU[j+1]    = Max_CPU[j];
					A_Location[j+1] = A_Location[j];
					B_Location[j+1] = B_Location[j];

					Max_CPU[j]    = temp1;
					A_Location[j] = temp2;
					B_Location[j] = temp3;
				}
				else
				{
					j=K1_Max_Report;
				}
			}
		}
//		 ii++ ;
//	} while (Location[ii]>0);
	}
}
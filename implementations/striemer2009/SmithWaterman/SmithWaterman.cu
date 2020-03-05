/*******************************************************************************/
/* Copyright 2008 by Gregory Striemer.  All rights reserved. This program      */
/* may not be sold or incorporated into a commercial product in whole or in    */
/* part, without written consent of Gregory Striemer.  For further information */
/* regarding permission for use or reproduction, please                        */
/* contact: Gregory Striemer at gmstrie@email.ece.arizona.edu                  */
/*******************************************************************************/



/*_____________________________________________________________________________________________________________*/
/*                                                                                                             */
/*                                     Main CUDA Smith-Waterman Program                                        */
/*_____________________________________________________________________________________________________________*/



// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes cuda libraries
#include <cutil.h>

// includes kernel
#include <SmithWaterman_kernel.cuh>



void runSW( int argc, char** argv);                                  //Function called from main


//All of the following functions are located in SmithWaterman_gold.cpp
extern "C"
void computeNumSeq( int*, char*);
extern "C"
void get_protein_lengths( int*, int*, char*, int*, unsigned int*);
extern "C"
void create_Host_Array(char*, char**);
extern "C"
void Host_Array_2Dto1D(int*, int*, char**,char*);
extern "C"
void Offset(int*, int*, int*, int*);
extern "C"
void zeros_to_array(int*, int*);
extern "C"
void write_Results(int*, char**, char*, int*, int*);



/*______________________________________________Program Main___________________________________________________*/


int main( int argc, char** argv){
  runSW( argc, argv);
  return 0;
}




/*______________________________________________runSW function_________________________________________________*/

void runSW( int argc, char** argv){
  //Searches for first available CUDA device
  CUT_DEVICE_INIT(argc, argv);

  if(argc!=4){
    printf("Syntax: %s <Query> <Database> <Results>\n", argv[0]);
    printf("Each query string will be run, one at a time, against the database.\n");
    return;
  }

  /*________________________Allocate memory on host for file name and number of sequences________________________*/

  //Allocate memory to contain the number of sequences in the file
  int* numSeq = (int*) malloc(sizeof(int));

  //Allocate memory for the database path entered by the user. Path must be less than 200 characters
  char* database_Path = argv[2];

  /*_____________________________________Compute number of sequences in file_____________________________________*/


  //Find the number of sequences contained in the database
  computeNumSeq(numSeq, database_Path);

  /*________________Allocate memory on host for protein lengths, max length, and total characters________________*/

  //Allocate Memory For Protein Lengths
  int* proteinLengths = (int*) malloc(sizeof(int) * numSeq[0] * 2);

  //Allocate Memory for and maximum protein length
  int* maxLen = (int*) malloc(sizeof(int));

  //Allocate Memory for total characters
  unsigned int* total_Characters = (unsigned int*) malloc(sizeof(int));

  /*________________Get protein and name lengths, max length, and total number of characters_____________________*/

  //Gets proteinLengths name lengths and max length from the database
  get_protein_lengths( proteinLengths, maxLen, database_Path, numSeq, total_Characters);

  /*____________________________Allocate Memory on Host for 2D Array of Sequences________________________________*/

  //Allocate Memory on Host For Array of Sequences
  char** databaseArray;

  //allocate memory for rows # of sequences                          i accesses sequences
  databaseArray = (char**)malloc(numSeq[0] * 2 * sizeof(char*));

  //for each row allocate memory for columns  Sequence Lengths       j accessings individual sequence residues
  for(int i=0; i < (numSeq[0] * 2); i++)
    *(databaseArray+i) = (char*)malloc( (proteinLengths[i]) * sizeof(char) );

  /*_________________________________________Create 2D Array on Host_____________________________________________*/

  //Create 2D Array on Host from the database file
  create_Host_Array(database_Path, databaseArray);

  /*____________________________Allocate Memory on Host for 1D Array of Sequences________________________________*/

  //Allocate Host Memory to hold 1D database array
  char* host_1D_Array = (char*) malloc(total_Characters[0] * sizeof(char));

  /*______________________________________Create 1D Array on Host________________________________________________*/

  //Create 1D Array
  Host_Array_2Dto1D(numSeq, proteinLengths, databaseArray, host_1D_Array);

  /*____________________________Allocate Memory on Device for 1D Array of Sequences______________________________*/


  //Allocate Device Memory to hold the 1D database array
  char* device_DatabaseArray;
  CUDA_SAFE_CALL(cudaMalloc((void**) &device_DatabaseArray, (total_Characters[0] * sizeof(char)) ));

  /*_________________________________Copy Host Database Array to Device Array____________________________________*/

  // copy host memory database array to allocated device array
  CUDA_SAFE_CALL(cudaMemcpy(device_DatabaseArray, host_1D_Array, (total_Characters[0] * sizeof(char)), cudaMemcpyHostToDevice));

  /*______________Allocate memory on host for database offset and protein lengths with no names__________________*/
  
  //Allocate memory for protein offset
  int* protein_Offset = (int*) malloc(sizeof(int) * numSeq[0]);
  
  //Allocate Host Memory for just protein lengths no names
  int* protein_length = (int*) malloc(sizeof(int) * numSeq[0]);
  //protein_Lengths contains lengths of protein names and corresponding sequences
  //protein_length contains lengths of only protein sequences

  /*__________________________________Find protein_Offset, and protein_length____________________________________*/

  //Create array containing base protein positions in 1D array
  //proteinLengths contains names, protein_length does not
  Offset(proteinLengths, protein_Offset, numSeq, protein_length);

  /*___________Allocate Device Memory to hold offset values, protein lengths, and results________________________*/

  //Allocate Device Memory to hold SmithWaterman Results
  int* device_SW_Results;
  CUDA_SAFE_CALL(cudaMalloc((void**) &device_SW_Results, (numSeq[0] * sizeof(int)) ));

  //Allocate Device Memory to hold Offset values
  int* device_offset;
  CUDA_SAFE_CALL(cudaMalloc((void**) &device_offset, (numSeq[0] * sizeof(int)) ));

  //Allocate Device Memory to hold protein lengths
  int* device_protein_length;
  CUDA_SAFE_CALL(cudaMalloc((void**) &device_protein_length, (numSeq[0] * sizeof(int)) ));

  /*____________________Allocate Device Memory to hold temporary calculation values______________________________*/

  //Allocate Device Memory to hold temporary calculation values **(The 64 is the current hard coded length for query sequence in the kernel)**
  short int* device_temp_1;                                              
  CUDA_SAFE_CALL(cudaMalloc((void**) &device_temp_1, (numSeq[0] * 64 * sizeof(short int)) ));

  //Create 1D Array of zero's to copy over to device_temp_1
  int* host_temp_array = (int*) malloc(numSeq[0] * 64 * sizeof(int));
  zeros_to_array(host_temp_array, numSeq);

  // copy host memory database array to allocated device array
  CUDA_SAFE_CALL(cudaMemcpy(device_temp_1, host_temp_array, (numSeq[0] * 64 * sizeof(short int)), cudaMemcpyHostToDevice));

  /*_____________________Copy offset values and protein lengths to Device Memory_________________________________*/

  // copy host memory offset to device memory
  CUDA_SAFE_CALL(cudaMemcpy(device_offset, protein_Offset, (numSeq[0] * sizeof(int)), cudaMemcpyHostToDevice));

  // copy host memory lengths to device memory
  CUDA_SAFE_CALL(cudaMemcpy(device_protein_length, protein_length, (numSeq[0] * sizeof(int)), cudaMemcpyHostToDevice));

  /*
  //print the database from 1D array using offsets
  for(int i = 0; i < numSeq[0]; i++){
    printf("\nDatabase Sequence #%d\n", i + 1);
    for(int k = 0; k < protein_length[i]; k++)
      printf("%c", host_1D_Array[ protein_Offset[i] + k]);
    printf("\n");
  }
  */



  /*_____________________________________Allocate Memory on Host for results_____________________________________*/


  //Allocate Memory on Host for Results
  int* host_scores = (int*) malloc(numSeq[0] * sizeof(int) );

  printf( "\nAllocating memory on host for results... \n");



  /*___________________________________Setup Execution Parameters on Kernel______________________________________*/


  //Number of Threads per block
  int numThreads = 64;
  int numBlocks;

  if (numSeq[0]%64 == 0)
    numBlocks = (numSeq[0]/64);
  else
    numBlocks = (numSeq[0]/64 + 1);

  // setup execution parameters
  dim3 threads(numThreads);
  dim3 grid(numBlocks);

  printf("\nnum threads is %d\n", numThreads);
  printf("\nnum blocks is %d\n", numBlocks);

  /*______________________________________________Start timer__________________________________________________*/

  //Start Timer
  unsigned int timer = 0;                                //Create a variable timer and set it to zero
  CUT_SAFE_CALL( cutCreateTimer( &timer));               //Creates a timer and sends result to variable timer
  CUT_SAFE_CALL( cutStartTimer( timer));                 //Starts the execution of the timer

  printf( "\nLaunching Kernel... \n");

  /*_____________________________________________Execute Kernel__________________________________________________*/

  // execute the kernel
  SmithWaterman_Kernel<<< grid, threads>>>(device_DatabaseArray, device_SW_Results, device_offset, device_protein_length, device_temp_1);

  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");                        //Report error if kernel did not launch

  /*_______________________________________Copy Results from GPU to Host_________________________________________*/

  printf( "\nCopying Results from GPU to host... \n");

  //Copy Results from GPU
  CUDA_SAFE_CALL(cudaMemcpy(host_scores, device_SW_Results, (numSeq[0] * sizeof(int)), cudaMemcpyDeviceToHost));

  /*_______________________________________________Check Results_________________________________________________*/


  //Stop Timer
  CUT_SAFE_CALL( cutStopTimer( timer));
  printf( "\nGPU database scan time: %f (ms)\n", cutGetTimerValue( timer));
  CUT_SAFE_CALL( cutDeleteTimer( timer));

  /*
  //Print some of the alignment scores to the screen
  for(int i = 0; i < 20; i++){
    printf("\nthe score for protein #%d is %d\n", (i+1), host_scores[i]);
    printf("It's length is %d\n", protein_length[i]);
  }
  */

  /*________________________________________Write Results to File________________________________________________*/

  //Write Results to File
  write_Results(proteinLengths, databaseArray, argv[3], numSeq, host_scores);

  /*______________________________________________Clean Up Data__________________________________________________*/

  //Free Host Memory
  free(numSeq          );
  free(maxLen          );
  free(databaseArray   );
  free(total_Characters);
  free(proteinLengths  );
  free(protein_Offset  );
  free(protein_length  );
  free(host_scores     );
  free(host_temp_array );
  free(host_1D_Array   );

  //Free Device Memory
  CUDA_SAFE_CALL(cudaFree(device_DatabaseArray ));
  CUDA_SAFE_CALL(cudaFree(device_SW_Results    ));
  CUDA_SAFE_CALL(cudaFree(device_protein_length));
  CUDA_SAFE_CALL(cudaFree(device_offset        ));
  CUDA_SAFE_CALL(cudaFree(device_temp_1        ));
}

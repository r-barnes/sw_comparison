
/*******************************************************************************/
/* Copyright 2008 by Gregory Striemer.  All rights reserved. This program      */
/* may not be sold or incorporated into a commercial product in whole or in    */
/* part, without written consent of Gregory Striemer.  For further information */
/* regarding permission for use or reproduction, please                        */
/* contact: Gregory Striemer at gmstrie@email.ece.arizona.edu                  */
/*******************************************************************************/

#ifndef _SmithWaterman_KERNEL_H_
#define _SmithWaterman_KERNEL_H_

#include <stdio.h>
#include <ctype.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#define database_character( index)  CUT_BANK_CHECKER(database_character, index)
#define temp_1( index)              CUT_BANK_CHECKER(temp_1,             index)
#define temp_2( index)              CUT_BANK_CHECKER(temp_2,             index)

//BLOSUM50 substitution matrix
__device__ __constant__  int blossum50[26][26] =
{
    // A   B   C   D   E   F   G   H   I   *   K   L   M   N   *   P   Q   R   S   T   *   V   W   X   Y   Z
/*A*/{ 5, -2, -1, -2, -1, -3,  0, -2, -1,  0, -1, -2, -1, -1,  0, -1, -1, -2,  1,  0,  0,  0, -3, -1, -2, -1},
/*B*/{-2,  5, -3,  5,  1, -4, -1,  0, -4,  0,  0, -4, -3,  4,  0, -2,  0, -1,  0,  0,  0, -4, -5, -1, -3,  2},
/*C*/{-1, -3, 13, -4, -3, -2, -3, -3, -2,  0, -3, -2, -2, -2,  0, -4, -3, -4, -1, -1,  0, -1, -5, -2, -3, -3},
/*D*/{-2,  5, -4,  8,  2, -5, -1, -1, -4,  0, -1, -4, -4,  2,  0, -1,  0, -2,  0, -1,  0, -4, -5, -1, -3,  1},
/*E*/{-1,  1, -3,  2,  6, -3, -3,  0, -4,  0,  1, -3, -2,  0,  0, -1,  2,  0, -1, -1,  0, -3, -3, -1, -2,  5},
/*F*/{-3, -4, -2, -5, -3,  8, -4, -1,  0,  0, -4,  1,  0, -4,  0, -4, -4, -3, -3, -2,  0, -1,  1, -2,  4, -4},
/*G*/{ 0, -1, -3, -1, -3, -4,  8, -2, -4,  0, -2, -4, -3,  0,  0, -2, -2, -3,  0, -2,  0, -4, -3, -2, -3, -2},
/*H*/{-2,  0, -3, -1,  0, -1, -2, 10, -4,  0,  0, -3, -1,  1,  0, -2,  1,  0, -1, -2,  0, -4, -3, -1,  2,  0},
/*I*/{-1, -4, -2, -4, -4,  0, -4, -4,  5,  0, -3,  2,  2, -3,  0, -3, -3, -4, -3, -1,  0,  4, -3, -1, -1, -3},
/***/{ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
/*K*/{-1,  0, -3, -1,  1, -4, -2,  0, -3,  0,  6, -3, -2,  0,  0, -1,  2,  3,  0, -1,  0, -3, -3, -1, -2,  1},
/*L*/{-2, -4, -2, -4, -3,  1, -4, -3,  2,  0, -3,  5,  3, -4,  0, -4, -2, -3, -3, -1,  0,  1, -2, -1, -1, -3},
/*M*/{-1, -3, -2, -4, -2,  0, -3, -1,  2,  0, -2,  3,  7, -2,  0, -3,  0, -2, -2, -1,  0,  1, -1, -1,  0, -1},
/*N*/{-1,  4, -2,  2,  0, -4,  0,  1, -3,  0,  0, -4, -2,  7,  0, -2,  0, -1,  1,  0,  0, -3, -4, -1, -2,  0},
/***/{ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
/*P*/{-1, -2, -4, -1, -1, -4, -2, -2, -3,  0, -1, -4, -3, -2,  0, 10, -1, -3, -1, -1,  0, -3, -4, -2, -3, -1},
/*Q*/{-1,  0, -3,  0,  2, -4, -2,  1, -3,  0,  2, -2,  0,  0,  0, -1,  7,  1,  0, -1,  0, -3, -1, -1, -1,  4},
/*R*/{-2, -1, -4, -2,  0, -3, -3,  0, -4,  0,  3, -3, -2, -1,  0, -3,  1,  7, -1, -1,  0, -3, -3, -1, -1,  0},
/*S*/{ 1,  0, -1,  0, -1, -3,  0, -1, -3,  0,  0, -3, -2,  1,  0, -1,  0, -1,  5,  2,  0, -2, -4, -1, -2,  0},
/*T*/{ 0,  0, -1, -1, -1, -2, -2, -2, -1,  0, -1, -1, -1,  0,  0, -1, -1, -1,  2,  5,  0,  0, -3,  0, -2, -1},
/***/{ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
/*V*/{ 0, -4, -1, -4, -3, -1, -4, -4,  4,  0, -3,  1,  1, -3,  0, -3, -1, -3, -2,  0,  0,  5, -3, -1, -1, -3},
/*W*/{-3, -5, -5, -5, -3,  1, -3, -3, -3,  0, -3, -2, -1, -4,  0, -4, -1, -3, -4, -3,  0, -3, 15, -3,  2, -2},
/*X*/{-1, -1, -2, -1, -1, -2, -2, -1, -1,  0, -1, -1, -1, -1,  0, -2, -1, -1, -1,  0,  0, -1, -3, -1, -1, -1},
/*Y*/{-2, -3, -3, -3, -2,  4, -3,  2, -1,  0, -2, -1,  0, -2,  0, -3, -1, -1, -2, -2,  0, -1,  2, -1,  8, -2},
/*Z*/{-1,  2, -3,  1,  5, -4, -2,  0, -3,  0,  1, -3, -1,  0,  0, -1,  4,  0,  0, -1,  0, -3, -2, -1, -2,  5}
};



//test query
__device__ __constant__ char query[65] =
"MPKAKTHSGASKRFRKTGTGKIVRQKANRRHLLEHKATKRTRRLDGRTVVAPNDVKRVTKMLNG";


//query length
__device__ __constant__ int query_len = 64;

//gap penalty
__device__ __constant__ int gap = 8;


/*__________________________________________SmithWaterman Kernel_______________________________________________*/

__global__ void
SmithWaterman_Kernel(
  char*      device_DatabaseArray,
  int*       device_SW_Results,
  int*       device_offset,
  int*       device_protein_length,
  short int* device_temp_1
){

  /*_________________________________________Shared Memory Allocation____________________________________________*/


  __shared__ char     database_character[64];           // Allocates memory for 1 database character per thread
  __shared__ short unsigned int  temp_1[256];           // 4 temp1 Values per thread
  __shared__ short unsigned int  temp_2[256];           // 4 temp2 values per thread




  /*_____________________________________Get access to thread ID and Block ID____________________________________*/

  // access current thread id
  const int thread_Id = threadIdx.x;

  // Result Index
  const int current_Index = (blockIdx.x * blockDim.x + thread_Id);         //represents the current protein in database





  /*__________________________Each Thread gets its corresponding protein length and index(offset)________________*/


  // Get protein lengths for each thread
  const int protein_len = device_protein_length[current_Index];

  // Get offsets for each of the proteins in the block
  const int offset = device_offset[current_Index];




  /*__________________________________________Assign Registers___________________________________________________*/


  short int s_1; //temporary value used in Smith-Waterman calculations
  short int s_2; //temporary value used in Smith-Waterman calculations

  short unsigned int k = 0;
  short unsigned int i = 0;
  short unsigned int j = 0;
  short unsigned int l = 0;


  short unsigned int  north;                  // contains north score
  short unsigned int  north_west;             // contains north_west score
  short unsigned int  score = 0;              // contains SW score




  if(query_len%4 == 0){

    /*__________________________________________Begin Outer Loop___________________________________________________*/

    //Get the characters from database 1 at a time
    for(i = 0; i < protein_len; i++){                 //Goes through length of database sequence **Outer Loop
      //Get each character from database, one at a time
      database_character[thread_Id] = device_DatabaseArray[offset + i];

      l          = 0;  // reset l to zero after getting a new database character
      k          = 0;
      north      = 0;  // For New column Reset to zero
      north_west = 0;  // For New column Reset to zero

  /*__________________________________________Begin Inner Loop___________________________________________________*/

      do {
        // Get 4 temp values from global memory
        temp_1[thread_Id * 4 + 0] = device_temp_1[current_Index * query_len + l];
        l++;
        temp_1[thread_Id * 4 + 1] = device_temp_1[current_Index * query_len + l];
        l++;
        temp_1[thread_Id * 4 + 2] = device_temp_1[current_Index * query_len + l];
        l++;
        temp_1[thread_Id * 4 + 3] = device_temp_1[current_Index * query_len + l];

        j = 0; //j accesses each of the temp score cells in shared memory




        /*__________________________________________Calculate Cell 1___________________________________________________*/

        s_1 = max(0, (north_west + blossum50[(int)database_character[thread_Id] - 65][(int)query[k] - 65]));
        s_2 = max(north - gap, s_1);
        temp_2[thread_Id * 4 + j] = max( temp_1[thread_Id * 4 + j] - gap, s_2);

        north      = temp_2[thread_Id * 4 + j];             //Set a new north
        north_west = (temp_1[thread_Id * 4 + j]);     //Set a new north_West

        if(score < north)
          score = north;

        j++;
        k++;


        /*__________________________________________Calculate Cell 2___________________________________________________*/

        s_1 = max(0, (north_west + blossum50[(int)database_character[thread_Id] - 65][(int)query[k] - 65]));
        s_2 = max(north - gap, s_1);
        temp_2[thread_Id * 4 + j] = max( temp_1[thread_Id * 4 + j] - gap, s_2);

        north      = temp_2[thread_Id * 4 + j];             //Set a new north
        north_west = (temp_1[thread_Id * 4 + j]);     //Set a new north_West

        if(score < north)
         score = north;

        j++;
        k++;


        /*__________________________________________Calculate Cell 3___________________________________________________*/

        s_1 = max(0, (north_west + blossum50[(int)database_character[thread_Id] - 65][(int)query[k] - 65]));
        s_2 = max(north - gap, s_1);
        temp_2[thread_Id * 4 + j] = max( temp_1[thread_Id * 4 + j] - gap, s_2);

        north      = temp_2[thread_Id * 4 + j];             //Set a new north
        north_west = (temp_1[thread_Id * 4 + j]);     //Set a new north_West

        if(score < north)
         score = north;

        j++;
        k++;


        /*__________________________________________Calculate Cell 4___________________________________________________*/

        s_1 = max(0, (north_west + blossum50[(int)database_character[thread_Id] - 65][(int)query[k] - 65]));
        s_2 = max(north - gap, s_1);
        temp_2[thread_Id * 4 + j] = max( temp_1[thread_Id * 4 + j] - gap, s_2);

        north      = temp_2[thread_Id * 4 + j];             //Set a new north
        north_west = (temp_1[thread_Id * 4 + j]);     //Set a new north_West

        if(score < north)
          score = north;

        k++;

        /*_________________________________Record Cell Scores To Global Memory_________________________________________*/

        //record temp2 scores to global memory
        device_temp_1[current_Index * query_len + l-3] = temp_2[thread_Id * 4 + 0];
        device_temp_1[current_Index * query_len + l-2] = temp_2[thread_Id * 4 + 1];
        device_temp_1[current_Index * query_len + l-1] = temp_2[thread_Id * 4 + 2];
        device_temp_1[current_Index * query_len + l] =   temp_2[thread_Id * 4 + 3];



        l++;
      } while(k < query_len - 1); //end of inner loop
    } //end of outer for loop


    /*________________________________________Write Scores to Global Memory________________________________________*/

    device_SW_Results[current_Index] = score;
  } else { //If query is not a multiple of 4

    /*__________________________________________Begin Outer Loop___________________________________________________*/

    //Get the characters from database 1 at a time
    for(i = 0; i < protein_len; i++){                 //Goes through length of database sequence **Outer Loop
      //Get each character from database, one at a time
      database_character[thread_Id] = device_DatabaseArray[offset + i];

      l          = 0;  // reset l to zero after getting a new database character
      k          = 0;
      north      = 0;  // For New column Reset to zero
      north_west = 0;  // For New column Reset to zero

      /*__________________________________________Begin Inner Loop___________________________________________________*/


      do {
        // Get 4 temp values from global memory
        temp_1[thread_Id * 4] = device_temp_1[current_Index * query_len + l];
        l++;

        j = 0; //j accesses each of the temp score cells in shared memory

        /*__________________________________________Calculate Cells___________________________________________________*/

        s_1 = max(0, (north_west + blossum50[(int)database_character[thread_Id] - 65][(int)query[k] - 65]));
        s_2 = max(north - gap, s_1);
        temp_2[thread_Id * 4 + j] = max( temp_1[thread_Id * 4 + j] - gap, s_2);

        north      = temp_2[thread_Id * 4 + j];             //Set a new north
        north_west = (temp_1[thread_Id * 4 + j]);     //Set a new north_West

        if(score < north)
          score = north;

        j++;
        k++;

        /*_________________________________Record Cell Scores To Global Memory_________________________________________*/

        //record temp2 scores to global memory
        device_temp_1[current_Index * query_len + l] = temp_2[thread_Id * 4];

        l++;
      } while(k < query_len - 1); //end of inner loop
    } //end of outer for loop

    /*________________________________________Write Scores to Global Memory________________________________________*/
    device_SW_Results[current_Index] = score;
  }
} //end of kernel

#endif // #ifndef _SmithWaterman_KERNEL_H_

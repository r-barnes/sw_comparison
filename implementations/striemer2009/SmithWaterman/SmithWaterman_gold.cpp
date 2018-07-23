/*******************************************************************************/
/* Copyright 2008 by Gregory Striemer.  All rights reserved. This program      */
/* may not be sold or incorporated into a commercial product in whole or in    */
/* part, without written consent of Gregory Striemer.  For further information */
/* regarding permission for use or reproduction, please                        */
/* contact: Gregory Striemer at gmstrie@email.ece.arizona.edu                  */
/*******************************************************************************/



/*_____________________________________________________________________________________________________________*/
/*                                                                                                             */
/*                                         SmithWaterman_gold.cpp                                              */
/*_____________________________________________________________________________________________________________*/




/*____________________________________________Export C Interface_______________________________________________*/

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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>




/*_______________________________________________computeNumSeq_________________________________________________*/

void computeNumSeq( int* numSeq, char *database_Path) {

/*_____________________________________Open Database File for Reading__________________________________________*/


// Open the database file for reading. Path example: H:\Users\Desktop\swissprot.txt
  printf("\nPlease enter the path to the database you wish to search:\n");
  scanf("%s",database_Path);

  FILE* dataBaseFile;                            //Create a pointer to the file
  dataBaseFile = fopen(database_Path, "r");       //Open the file for reading

  if(dataBaseFile == NULL)                       //Print error message if file cannot be opened
    do {
      printf("\nError: File Could Not be Opened. \n");
      printf("\nPlease enter the path to the database you wish to search:\n");
      scanf("%s",database_Path);
      dataBaseFile = fopen(database_Path, "r");       //Open the file for reading
  } while (dataBaseFile == NULL);

  printf("\nFile Opened Successfully.\n");

  /*_________________________________Get number of Sequences Contained in File___________________________________*/


  int fileChar = 0;   //Contains ascii value of current character value
  numSeq[0]    = 0;      //Set value located at pointer initially to zero

  fileChar = fgetc(dataBaseFile); //Get the first character from the database

  while(fileChar != EOF) {                                      
    if( (char)fileChar == '>')                                       
      numSeq[0] = numSeq[0] + 1;                                                      
    fileChar = fgetc(dataBaseFile);
  }

  printf("\n# of sequences in database: %d\n", numSeq[0]); //Prints out the number of sequences contained in the database

  rewind(dataBaseFile);
  fclose(dataBaseFile); 
}



/*__________________________________________get_protein_lengths________________________________________________*/


void get_protein_lengths(
  int*          proteinLengths,
  int*          maxLen, 
  char*         database_Path,
  int*          numSeq,
  unsigned int* total_Characters
){
  int fileChar = 0;   //Contains ascii value of current character value
  FILE* dataBaseFile;                             //Create a pointer to the file
  dataBaseFile = fopen(database_Path, "r");       //Open the file for reading

  int i = 0;
  int j;

  fileChar = fgetc(dataBaseFile);                                  //Get initial character from file

  while(1) {
    j = 0;                                                         //Reset counter to count next sequence length

    if( (char)fileChar == '>'){                                    //If character is the beginning of a protein name
      do{
        fileChar = fgetc(dataBaseFile);                            //Load next character from database file
        if((fileChar != 13) || (fileChar != 10)){                  //Only add if fileChar does not equal CR or LF
          proteinLengths[i] = j;
          j++;
        }
      } while(fileChar != 10);                                     //Stop when fileChar == 10 (LF NL linefeed, new line)
    }

    proteinLengths[i]++;                                           //Add additional digit to length
    j = 0;                                                         //Reset counter to count next sequence length
    i++;                                                           //Procede to next row

    if( (char)fileChar != '>'){
      do {
        fileChar = fgetc(dataBaseFile);                                  //Load next character from database file

        if((fileChar != 13) && (fileChar != 10) && (fileChar != EOF)){   //Only add if fileChar does not equal CR or LF
          proteinLengths[i] = j;
          j++;
        }
      } while( ( (char)fileChar != '>') && (fileChar != EOF) );
    }

    if(fileChar != EOF)
      i++;                                                             //Advance to next sequence

    if(fileChar == EOF)                                                //Add an extra 1 to the length of the final sequence
      proteinLengths[i]++;

    if(fileChar == EOF)                                                //If end of file is reached, then the loop is ended
      break;
  }

  //Gets Maximum Length of Sequences
  maxLen[0] = 0;                                       // Contains Largest Protein Length
  for(i = 0; i < (numSeq[0] * 2); i++){                // Multiply by 2 to account for protein names
    if(maxLen[0] < proteinLengths[i])                  // Return Max element of proteinLength
      maxLen[0] = proteinLengths[i];
  }

  printf("\nThe maximum protein length is %d characters \n", maxLen[0]);          //Prints the maximum protein length

  total_Characters[0] = 0;

  //Find Total number of characters in database

  for(int i=1; i < (numSeq[0] * 2); i=i+2)     //This does not include protein names
    total_Characters[0] = ( total_Characters[0] + proteinLengths[i] );

  printf("\nThe number of characters in the file is: %d\n", total_Characters[0]);

  rewind(dataBaseFile);    
  fclose(dataBaseFile);      
}




/*_____________________________________________create_Host_Array_______________________________________________*/


void create_Host_Array(char* database_Path, char** databaseArray){
  int fileChar = 0;   //Contains ascii value of current character value
  FILE* dataBaseFile = fopen(database_Path, "r");       //Open the file for reading

  int i = 0;
  int j = 0;

  fileChar = fgetc(dataBaseFile);                               //Get first character from database file

  while(1) {
    if( (char)fileChar == '>'){                                 //If character is the beginning 
      do {
        if((fileChar != 13) && (fileChar != 10)){
          databaseArray[i][j] = (char)fileChar;
          j++;                                                  //Load next character in horizontal direction into array
        }
        fileChar = fgetc(dataBaseFile);
      } while(fileChar != 10);
    }

    j = 0;
    i++;                                                          //Advance to next row

    if( (char)fileChar != '>'){
      do{
        fileChar = fgetc(dataBaseFile);                                //Load next character from database file
        if((fileChar != 13) && (fileChar != 10) && (fileChar != EOF)){ //Only add if fileChar does not equal CR or LF
          databaseArray[i][j] = (char)fileChar;
          j++;
        }
      } while( ( (char)fileChar != '>') && (fileChar != EOF) );
    }

    j = 0;

    if(fileChar != EOF)
      i++;                                                             //Advance to next row

    if(fileChar == EOF)                                                //If end of file is reached, then the loop is ended
      break;
  }     //End of While when EOF is reached

  rewind(dataBaseFile);    
  fclose(dataBaseFile);
}



/*_____________________________________________Host_Array_2Dto1D_______________________________________________*/


void Host_Array_2Dto1D(int* numSeq, int* proteinLengths, char** databaseArray, char* host_1D_Array){
  //Transfer host array to 1D  host array
  int k = 0;
  for(int i = 1; i < (numSeq[0] * 2);   i=i+2)           //database sequences are located in odd indices
  for(int j = 0; j < proteinLengths[i]; j++  ){
    host_1D_Array[k] = databaseArray[i][j];
    k++;
  }
}



/*____________________________________________________Offset___________________________________________________*/


void Offset(int* proteinLengths, int* protein_Offset, int* numSeq, int* protein_length){
  int i;
  int k=0;

  for(i=1; i < (numSeq[0] * 2); i=i+2){
    protein_length[k] = proteinLengths[i];
    k++;
  }

  protein_Offset[0] = 0;

  for(i=1; i<numSeq[0]; i++)
    protein_Offset[i] = protein_length[i-1] + protein_Offset[i-1];              
}


/*________________________________________________Copy Zeros___________________________________________________*/


void zeros_to_array(int* host_temp_array, int* numSeq){
  for (int i  = 0; i < (numSeq[0] * 64); i++)           //the 64 represents the length of the query sequence
    host_temp_array[i] = 0;
}





/*__________________________________________Get file for results_______________________________________________*/

void write_Results(int* proteinLengths, char** databaseArray, char* Results_Path, int* numSeq, int* host_scores){
  int i;
  int j;
  int k = 0;

  // Open the database file for reading
  printf("\nPlease enter the path to the Location for Results Storage:\n");
  scanf("%s",Results_Path);



  FILE* resultsFile = fopen(Results_Path, "w");         //Open the file for reading

  if(resultsFile == NULL){                        //Print error message if file cannot be opened
    do{
      printf("\nError: File Could Not be Opened. \n");
      printf("\nPlease enter the path to the Location for Results Storage:\n");
      scanf("%s",Results_Path);
      resultsFile = fopen(Results_Path, "w");       //Open the file for reading
    } while(resultsFile == NULL);
  }

  printf("\nFile Opened Successfully for writing.\n");

  for(i = 0; i < numSeq[0]; i++){
    for(j = 0; j < proteinLengths[k] - 1; j++)
      fprintf(resultsFile, "%c", databaseArray[k][j]);
    k = k + 2;
    fprintf(resultsFile, "\nThe score for sequence #%d is:%d\n", i + 1, host_scores[i]);
  }
}

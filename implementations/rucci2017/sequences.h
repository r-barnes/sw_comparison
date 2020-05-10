#ifndef DB_H_INCLUDED
#define DB_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include "arguments.h"
#include "utils.h"
#include "swimm.h"

#define BUFFER_SIZE 1000
#define ALLOCATION_CHUNK 1000

#define DUMMY_ELEMENT 'Z'+1
#define PREPROCESSED_DUMMY_ELEMENT 24

#if AVX2 || SSE41 || AVX512BW
	#define TUNING_DB_SEQ_SIZE 134217728
#else
	#define TUNING_DB_SEQ_SIZE 67108864
#endif
#define TUNING_DB_SEQ_LENGTH 335

#define TUNING_QUERY_COUNT 10

#define TUNING_MIN_BLOCK_SIZE 16
#define TUNING_MAX_BLOCK_SIZE 128
#define TUNING_BLOCK_SIZE_STEP 16

#define TUNING_REPEAT_TIMES 3


// DB preprocessing
void preprocess_db (char * input_filename, char * out_filename, int n_procs);

// DB assembly 
void assemble_single_chunk_db (char * sequences_filename, int vector_length, unsigned long int * sequences_count,
				unsigned long int * D, unsigned short int * sequences_db_max_length, int * max_title_length, unsigned long int * vect_sequences_db_count, unsigned long int * vD, char **ptr_vect_sequences_db,
				unsigned short int ** ptr_vect_sequences_db_lengths, unsigned short int ** ptr_vect_sequences_db_blocks, 
				unsigned long int ** ptr_vect_sequences_db_disp, int n_procs, int block_width);

// Load DB headers
void load_database_headers (char * sequences_filename, unsigned long int sequences_count, int max_title_length, char *** ptr_sequences_db_headers);

void load_query_sequences(char * queries_filename, char ** ptr_query_sequences, char *** ptr_query_headers, unsigned short int **ptr_query_sequences_lengths,
						unsigned short int **ptr_m, unsigned long int * query_sequences_count, unsigned long int * ptr_Q, unsigned int ** ptr_query_sequences_disp, int n_procs) ; 

// Functions for parallel sorting
void merge_sequences(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void mergesort_sequences_serial(char ** sequences, char ** titles, unsigned short int * sequences_lengths, unsigned long int size);

void sort_sequences (char ** sequences,  char ** titles, unsigned short int * sequences_lengths, unsigned long int size, int threads);


// generate synthetic query sequence
void load_tuning_query_sequence (char ** ptr_tun_query_sequence, unsigned short int ** ptr_tun_query_sequence_lengths, unsigned int ** ptr_tun_query_sequence_disps);


// genrate synthetic db sequences
void assemble_tuning_chunk_db (char ** ptr_tun_vect_db_sequences, unsigned short int ** ptr_tun_vect_db_sequences_lengths, 
								unsigned short int ** ptr_tun_vect_db_sequences_blocks, unsigned long int ** ptr_tun_vect_db_sequences_disp,
								unsigned long int * ptr_tun_vect_db_sequences_count) ;


#endif

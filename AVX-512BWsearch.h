#ifndef AVX512BWSEARCH_H_INCLUDED
#define AVX512BWSEARCH_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <immintrin.h>
#include "charSubmat.h"
#include "utils.h"
#include "sequences.h"

#define AVX512BW_INT8_VECTOR_LENGTH 64
#define AVX512BW_INT16_VECTOR_LENGTH 32
#define AVX512BW_INT32_VECTOR_LENGTH 16
#define AVX512BW_UNROLL_COUNT 10

#define SUBMAT_ROWS_x_AVX512BW_INT8_VECTOR_LENGTH 1536

// search using AVX512BW instrucions and Score Profile technique
void search_avx512bw_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime);

// search using AVX512BW instrucions and Query Profile technique
void search_avx512bw_qp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime);


#endif

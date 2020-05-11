#ifndef AVX2SEARCH_H_INCLUDED
#define AVX2EARCH_H_INCLUDED

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

#define AVX2_INT8_VECTOR_LENGTH 32
#define AVX2_INT16_VECTOR_LENGTH 16
#define AVX2_INT32_VECTOR_LENGTH 8
#define AVX2_UNROLL_COUNT 10

#define SUBMAT_ROWS_x_AVX2_INT8_VECTOR_LENGTH 768

// database search using AVX2 instrucions and Score Profile technique
void search_avx2_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime);

#endif

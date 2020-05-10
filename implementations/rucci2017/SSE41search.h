#ifndef SSE41SEARCH_H_INCLUDED
#define SSE41SEARCH_H_INCLUDED

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

#define SSE_INT8_VECTOR_LENGTH 16
#define SSE_INT16_VECTOR_LENGTH 8
#define SSE_INT32_VECTOR_LENGTH 4
#define SSE_UNROLL_COUNT 10

#define SUBMAT_ROWS_x_SSE_INT8_VECTOR_LENGTH 384

// CPU search using SSE instrucions and Score Profile technique
void search_sse41_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int cpu_block_size, int * scores, double * workTime);


#endif

#ifndef AVX512FSEARCH_H_INCLUDED
#define AVX512FSEARCH_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include <immintrin.h>
#include "intSubmat.h"
#include "utils.h"
#include "sequences.h"
#include "arguments.h"

#define AVX512F_INT32_VECTOR_LENGTH 16
#define UNROLL_COUNT 8

#define SUBMAT_ROWS_x_AVX512_INT32_VECTOR_LENGTH 384


// Host search using AVX-512 instrucions and Score Profile technique
void search_avx512f_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp, 
	char profile, unsigned short int query_length_threshold, 
	char * vect_db_sequences, unsigned short int * vect_db_sequences_lengths, unsigned short int * vect_db_nbbs, unsigned long int vect_db_sequences_count, unsigned long int * vect_db_sequences_disp,
	__m512i * submat, int open_gap, int extend_gap, int n_threads, int block_width, __m512i * scores, double * workTime);



#endif

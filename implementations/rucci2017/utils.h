#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <sys/time.h>
#include "arguments.h"
#include "charSubmat.h"
#include "intSubmat.h"
#include "sequences.h"
#include "SSE41search.h"
#include "AVX2search.h"
#include "AVX-512Fsearch.h"
#include "AVX-512BWsearch.h"


void merge_scores(int * scores, char ** titles, unsigned long int size);

void mergesort_scores_serial(int * scores, char ** titles, unsigned long int size);

void sort_scores (int * scores, char ** titles, unsigned long int size, int threads);

double dwalltime();

// tune number of threads and block size considering processor
void tuning (int * threads, int * block_size, unsigned short int * vect_db_sequences_lengths,
	unsigned short int * vect_db_sequences_blocks, unsigned long int vect_db_sequences_count);

#endif

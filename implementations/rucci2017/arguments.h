#ifndef ARGUMENTS_H_INCLUDED
#define ARGUMENTS_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "charSubmat.h"
#include "intSubmat.h"
#include "sequences.h"

#define THREADS 8
#define BLOCK_SIZE 80
#define OPEN_GAP 10
#define EXTEND_GAP 2
#define TOP 10
#define QUERY_PROFILE 'Q'
#define SCORE_PROFILE 'S'
#define QUERY_LENGTH_THRESHOLD 0


// Arguments parsing
void program_arguments_processing (int argc, char * argv[]);
static int parse_opt (int key, char *arg, struct argp_state *state);

// Global options
extern char * sequences_filename, * queries_filename, *input_filename, * output_filename, *op, submat_name[];
extern char profile;
extern int block_size, threads, open_gap, extend_gap;
extern unsigned short int query_length_threshold;
extern unsigned long int top;
#if AVX512F
extern int iBlosum45[], iBlosum50[], iBlosum62[], iBlosum80[], iBlosum90[], iPam30[], iPam70[], iPam250[], * intSubmat;
#else
extern char cBlosum45[], cBlosum50[], cBlosum62[], cBlosum80[], cBlosum90[], cPam30[], cPam70[], cPam250[], * charSubmat;
#endif

#endif
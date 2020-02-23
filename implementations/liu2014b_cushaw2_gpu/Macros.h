/*
 *  macros.h
 *
 *  Created on: Dec 23, 2011
 *      Author: yongchao
 */

#ifndef MACROS_H_
#define MACROS_H_
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdint.h>
#include <assert.h>
#include <limits.h>
#include <zlib.h>
#include <math.h>
#include <pthread.h>
#include <iterator>
#include <algorithm>
#include <bitset>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>

using namespace std;

/*the version of CUSHAW2*/
#define CUSHAW2_VERSION "2.1.8-r16"

/*maximal number of top alignments*/
#define MAX_MULTI_ALIGN			100

/*macros for the alignment path trace-back*/
#define ALIGN_DIR_STOP				0
#define	 ALIGN_DIR_DIAGONAL			1
#define ALIGN_DIR_UP				2
#define	 ALIGN_DIR_LEFT				3
#define ALIGN_OP_M	0		/*means substitution*/
#define ALIGN_OP_I	1		/*means insertion at the first sequence*/
#define ALIGN_OP_D	2		/*means deletion at the first sequence*/
#define ALIGN_OP_S	3		/*means the part out of the local alignment*/
#define ALIGN_MIN_SCORE				-9999999

//SAM format
#define SAM_FPD   1 // paired
#define SAM_FPP   2 // properly paired
#define SAM_FSU   4 // self-unmapped
#define SAM_FMU   8 // mate-unmapped
#define SAM_FSR  16 // self on the reverse strand
#define SAM_FMR  32 // mate on the reverse strand
#define SAM_FR1  64 // this is read one
#define SAM_FR2 128 // this is read two
#define SAM_FSC 256 // secondary alignment
#define	DEFAULT_MAX_MAP_QUAL	250
#define SW_MAP_QUALITY_SCORE	0.5

//single and paired end alignment
#ifndef MAX_USER_READ_LENGTH
#define MAX_SEQ_LENGTH					320
#else
#define MAX_SEQ_LENGTH		MAX_USER_READ_LENGTH
#endif

#define GLOBAL_MIN_SEED_SIZE			11
#define GLOBAL_MAX_SEED_SIZE			49
#define GLOBAL_MAX_NUM_SEED_REPEATS		1024

#define DEFAULT_SEQ_LENGTH_RATIO		0.80
#define DEFAULT_MIN_IDENTITY			0.90

#define FILE_TYPE_FASTX   0
#define FILE_TYPE_BAM     1
#define FILE_TYPE_SAM     2
#define	 FILE_TYPE_FIFO		3
#define FILE_FORMAT_FASTQ   4
#define FILE_FORMAT_FASTA   5
#define FILE_FORMAT_BSAM    6
#define FILE_FORMAT_FIFO	7

#define INS_SIZE_EST_MULTIPLE		0x10000

#include "Structs.h"
#include "BwtMacros.h"

#endif /* MACROS_H_ */

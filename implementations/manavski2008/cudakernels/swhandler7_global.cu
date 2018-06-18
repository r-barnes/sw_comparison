
// includes, project
#include <cutil.h>

#include <stdlib.h>
#include <stdio.h>

#include "sbtmatrix.h"

#include <sw_kernel7_global.h>

extern "C" void swhandler7_global( const unsigned numThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos,   unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, int* d_scores, int *d_colMemory) {

		dim3  grid( 1, 1, 1);
		dim3  threads( numThreads, 1, 1);
		sw_kernel7_global<<< grid, threads, DIMSHAREDSPACE_7 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores, d_colMemory);
		CUT_CHECK_ERROR("Kernel execution failed");
}

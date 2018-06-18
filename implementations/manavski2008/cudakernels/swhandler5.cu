
// includes, project
#include <cutil.h>

#include "sbtmatrix.h"

#include <sw_kernel5.h>

extern "C" void swhandler5( const unsigned residueThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, int* d_scores) {

		dim3  grid( 1, 1, 1);
		dim3  threads( residueThreads, 1, 1);
		sw_kernel5<<< grid, threads, DIMSHAREDSPACE_5 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);
		CUT_CHECK_ERROR("Kernel execution failed");
}



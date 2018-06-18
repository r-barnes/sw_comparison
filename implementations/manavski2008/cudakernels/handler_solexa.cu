
#include <cutil.h>
#include <cuda_runtime_api.h>

#include <string.h>

__constant__ int const_e2g_sbtmat[1024];

#include <e2g_kernel_1.h>

#include "sbtmatrix.h"

#include <stdio.h>

static bool matAssigned = false;

extern "C" void solexa_handler_1( const unsigned gridSize, const unsigned numThreads, const char* d_strToAlign, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const char *d_splice_sites, const unsigned dbsize, const short unsigned first_gap_penalty, const short unsigned next_gap_penalty, const short unsigned splice_penalty, const short unsigned intron_penalty, int* d_scores)
{
	// loading substitution matrix
	if (!matAssigned) {
		cudaMemcpyToSymbol( const_e2g_sbtmat, cpu_solexa, 1024 * sizeof(int), 0);
		matAssigned = true;
	}

	printf("num threads: %d\n", numThreads);

	cudaChannelFormatDesc channelDescQ = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	tex_queries.normalized = false;
	cudaError_t err = cudaBindTexture( 0, &tex_queries, d_strToAlign, &channelDescQ, (size_t)SOLEXA_QUERY_SIZE+1);

    cudaChannelFormatDesc channelDescSP = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	tex_splice_sites.normalized = false;
	err = cudaBindTexture( 0, &tex_splice_sites, d_splice_sites, &channelDescSP, (size_t)dbsize);

	dim3  grid( gridSize, 1, 1);
	dim3  threads( numThreads, 1, 1);
	solexa_kernel<<< grid, threads >>>( d_seqlib, d_offsets, d_sizes, first_gap_penalty, next_gap_penalty, splice_penalty, intron_penalty, d_scores);
	CUT_CHECK_ERROR("Kernel execution failed");

    cudaThreadSynchronize();

}

/*
void solexa_handler_2( const unsigned numThreads, const char* d_strToAlign, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* d_scores) {

	//caricamento matrice
	if (strcmp (lastSubMat,subMat) != 0) {
		if ( strcmp (subMat,"BL50") == 0 ) {
			//printf("loading BL50\n");
			cudaMemcpyToSymbol( blosum50_7, cpu_abl50, 1024 * sizeof(int), 0);
		} else if ( strcmp (subMat,"BL62") == 0 ) {
			//printf("loading BL62\n");
			cudaMemcpyToSymbol( blosum50_7, cpu_abl62, 1024 * sizeof(int), 0);
		} else if ( strcmp (subMat,"BL90") == 0 ) {
			//printf("loading BL90\n");
			cudaMemcpyToSymbol( blosum50_7, cpu_abl90, 1024 * sizeof(int), 0);
		} else {
			cudaMemcpyToSymbol( blosum50_7, cpu_idt, 1024 * sizeof(int), 0);
		}
	}

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	texB7.normalized = false;
	cudaError_t err = cudaBindTexture( 0, &texB7, d_strToAlign, &channelDesc, (size_t)strToAlignSize+1);

	dim3  grid( 1, 1, 1);
	dim3  threads( numThreads, 1, 1);
	sw_kernel7<<< grid, threads, DIMSHAREDSPACE_7 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);
	CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL( cudaThreadSynchronize() );

}
*/


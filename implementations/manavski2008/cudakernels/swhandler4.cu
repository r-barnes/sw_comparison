
/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>

// includes, project
#include <cutil.h>


__constant__ int blosum50_6[1024];
#include <sw_kernel6.h>

__constant__ int blosum50_7[1024];

#include <sw_kernel7.h>
#include "sbtmatrix.h"


#include <QtCore/QTime>

#define MAX_BLOCK_SIZE 32

extern "C" void swInitMem( const char *seqlib, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes,  char **d_seqlib, unsigned **d_offsets, unsigned **d_sizes, int **d_scores, unsigned **d_endpos)
{
	//printf("init device memory\n");

	CUDA_SAFE_CALL( cudaMalloc( (void**) d_seqlib, totBytesUsed));
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_offsets, numSeqs*sizeof(unsigned)) );
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_sizes, numSeqs*sizeof(unsigned)) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( *d_seqlib, seqlib, totBytesUsed, cudaMemcpyHostToDevice) );
	unsigned arrSizes = numSeqs*sizeof(unsigned);
	CUDA_SAFE_CALL( cudaMemcpy( *d_offsets, offsets, arrSizes, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( *d_sizes, sizes, arrSizes, cudaMemcpyHostToDevice) );

	// allocate device memory for result
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_scores, numSeqs*sizeof(int) ));

	//initialization of d_scores to avoid misleading results
	CUDA_SAFE_CALL( cudaMemset( *d_scores, 0, numSeqs*sizeof(int) ) );

	// allocate device memory for end positions
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_endpos, numSeqs*sizeof(int) ));

	//initialization of d_endpos to avoid misleading results
	CUDA_SAFE_CALL( cudaMemset( *d_endpos, 0, numSeqs*sizeof(int) ) );
}

extern "C" void swCleanMem( char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int *d_scores, unsigned *d_endpos )
{
	//printf("to free device memory\n");

	CUDA_SAFE_CALL(cudaFree(d_endpos));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sizes));
	CUDA_SAFE_CALL(cudaFree(d_offsets));
	CUDA_SAFE_CALL(cudaFree(d_seqlib));

}

void swhandler6( const unsigned gridSize, const unsigned numThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* d_scores,  unsigned *d_endpos, bool calc_endpos)
{
	//caricamento matrice
	if (strcmp (lastSubMat,subMat) != 0) {
		if ( strcmp (subMat,"BL50") == 0 ) {
			//printf("loading BL50\n");
			cudaMemcpyToSymbol( blosum50_6, cpu_abl50, 1024 * sizeof(int), 0);
		} else if ( strcmp (subMat,"BL62") == 0 ) {
			//printf("loading BL62\n");
			cudaMemcpyToSymbol( blosum50_6, cpu_abl62, 1024 * sizeof(int), 0);
		} else if ( strcmp (subMat,"BL90") == 0 ) {
			//printf("loading BL90\n");
			cudaMemcpyToSymbol( blosum50_6, cpu_abl90, 1024 * sizeof(int), 0);
		} else {
			//printf("loading DNA1\n");
			cudaMemcpyToSymbol( blosum50_6, cpu_dna1, 1024 * sizeof(int), 0);
		}
	}

	dim3  grid( gridSize, 1, 1);
	dim3  threads( numThreads, 1, 1);
	
	if (calc_endpos) {
		sw_kernel6_with_endpos<<< grid, threads, DIMSHAREDSPACE_6 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores, d_endpos);
	} else {
		sw_kernel6<<< grid, threads, DIMSHAREDSPACE_6 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);
	}
	CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL( cudaThreadSynchronize() );

}

void swhandler7( const unsigned numThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* d_scores,  unsigned *d_endpos, bool calc_endpos) {

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
			cudaMemcpyToSymbol( blosum50_7, cpu_dna1, 1024 * sizeof(int), 0);
		}
	}

	dim3  grid( 1, 1, 1);
	dim3  threads( numThreads, 1, 1);
	
	if (calc_endpos) {
		sw_kernel7_with_endpos<<< grid, threads, DIMSHAREDSPACE_7 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores, d_endpos);
	} else {
		sw_kernel7<<< grid, threads, DIMSHAREDSPACE_7 >>>( d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);
	}	
	
	CUT_CHECK_ERROR("Kernel execution failed");

    CUDA_SAFE_CALL( cudaThreadSynchronize() );

}

extern "C" double smithWatermanCuda2(const char* strToAlign, const unsigned sizeNotPad, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* h_scores, unsigned* h_endpos, char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int * &d_scores,  unsigned * &d_endpos, bool calc_endpos, unsigned debug) {

	// allocate device memory
	unsigned strToAlignSize = strlen(strToAlign);
	char* d_strToAlign;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_strToAlign, strToAlignSize+1 ) );

	//settaggio GRID_SIZE sulla base della sequenza di ingresso
	unsigned GRID_SIZE;
	if (strToAlignSize<50)
		GRID_SIZE = 1500;
	else if (strToAlignSize<361)
		GRID_SIZE = 500;
	else if (strToAlignSize > 360 && strToAlignSize < 600)
		GRID_SIZE = 300;
	else if (strToAlignSize > 599 && strToAlignSize < 1024)
		GRID_SIZE = 200;
	else 
		GRID_SIZE = 50;

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( d_strToAlign, strToAlign, strToAlignSize+1, cudaMemcpyHostToDevice) );

	double max_timer_value=0;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);

	texB6.normalized = false;
	texB7.normalized = false;


	CUDA_SAFE_CALL( cudaBindTexture( 0, &texB6, d_strToAlign, &channelDesc, (size_t)strToAlignSize+1) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, &texB7, d_strToAlign, &channelDesc, (size_t)strToAlignSize+1) );


	unsigned arrSizes = numSeqs*sizeof(int);

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );

	//numero sequenze effettive
	unsigned numSeqsEff = stopPos - startPos + 1;

	unsigned numTotBlocks = numSeqsEff / MAX_BLOCK_SIZE;
	unsigned residueThreads = numSeqsEff % MAX_BLOCK_SIZE;

	//chiamata per il residuo

	bool matasigned6 = false;
	bool matasigned7 = false;
	if ( residueThreads != 0 && !(MAX_BLOCK_SIZE % residueThreads) ) {
		//se il residuo è un sottomultiplo di MAX_BLOCK_SIZE

		dim3  grid( 1, 1, 1);
		dim3  threads( residueThreads, 1, 1);

		QTime timer_krl; timer_krl.start();

		if (matasigned6)
			swhandler6(1, residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, subMat, subMat, d_scores, d_endpos, calc_endpos);
		else {
			swhandler6(1, residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, subMat, lastSubMat, d_scores, d_endpos, calc_endpos);
			matasigned6 = true;
		}

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;

	} else if (residueThreads != 0 && (MAX_BLOCK_SIZE % residueThreads)) {
		//se non è sottomultiplo di MAX_BLOCK_SIZE

		QTime timer_krl; timer_krl.start();

		if (matasigned7) {
			swhandler7(residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, subMat, subMat, d_scores, d_endpos, calc_endpos);
		} else {
			swhandler7(residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, subMat, lastSubMat, d_scores, d_endpos, calc_endpos);
			matasigned7 = true;
		}

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;
	}

	//chiamata per multipli di MAX_BLOCK_SIZE

	unsigned newStartPos = startPos + residueThreads;

	unsigned cnt;
	for (cnt=0; cnt<numTotBlocks;) {

		unsigned numBlocks = (cnt + GRID_SIZE > numTotBlocks) ? (numTotBlocks - cnt) : GRID_SIZE;

		dim3  grid( numBlocks, 1, 1);
		dim3  threads( MAX_BLOCK_SIZE, 1, 1);

		QTime timer_krl; timer_krl.start();

		if (matasigned6)
			swhandler6(numBlocks, MAX_BLOCK_SIZE, d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, subMat, subMat, d_scores, d_endpos, calc_endpos);
		else {
			swhandler6(numBlocks, MAX_BLOCK_SIZE, d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, subMat, lastSubMat, d_scores, d_endpos, calc_endpos);
			matasigned6 = true;
		}

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;

		cnt += numBlocks;
		newStartPos += numBlocks * MAX_BLOCK_SIZE;
	}

	if (debug != 0)
		printf( "\nMAX TIMER VALUE: %f (ms), NUM BLOCKS: %d\n", max_timer_value, cnt);

	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy( h_endpos+startPos, d_endpos+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );

	// cleanup memory
	CUDA_SAFE_CALL(cudaFree(d_strToAlign));
	
	return 0;
}



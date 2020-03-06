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

// includes, kernels
#include "sbtmatrix.h"

texture<unsigned char, 1, cudaReadModeElementType> texB6;
texture<unsigned char, 1, cudaReadModeElementType> texB7;

extern "C" void swhandler6_global( const unsigned gridSize, const unsigned numThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos,   unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, int* d_scores, int *d_colMemory);

extern "C" void swhandler7_global( const unsigned numThreads, const char* d_strToAlign, const unsigned sizeNotPad, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const unsigned alpha, const unsigned beta, int* d_scores, int *d_colMemory);

#define MAX_BLOCK_SIZE 32
#define MAX_LENGTH_SUPPORTED 2050

extern "C" void swInitMem( const char *seqlib, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, int* h_scores, char **d_seqlib, unsigned **d_offsets, unsigned **d_sizes, int **d_scores)
{
	CUT_CHECK_DEVICE();

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

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( *d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );
}

extern "C" void swCleanMem( char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int* d_scores )
{
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sizes));
	CUDA_SAFE_CALL(cudaFree(d_offsets));
	CUDA_SAFE_CALL(cudaFree(d_seqlib));
}

extern "C" double smithWatermanCuda2( const char* strToAlign, const unsigned sizeNotPad, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, int* h_scores, char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int * & d_scores) {

	unsigned int timer = 0;
	double timeTot = 0;

	// allocate device memory
	unsigned strToAlignSize = strlen(strToAlign);
	char* d_strToAlign;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_strToAlign, strToAlignSize+1 ) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( d_strToAlign, strToAlign, strToAlignSize+1, cudaMemcpyHostToDevice) );

	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	//prova texture
	cudaChannelFormatDesc chDesc;
	chDesc.x = 8;
	chDesc.y = 0;
	chDesc.z = 0;
	chDesc.w = 0;
	chDesc.f = cudaChannelFormatKindUnsigned;
	texB6.normalized = false;
	texB7.normalized = false;

	CUDA_SAFE_CALL( cudaBindTexture( "texB6", d_strToAlign, &chDesc, (size_t)strToAlignSize+1, (size_t)0 ) );
	CUDA_SAFE_CALL( cudaBindTexture( "texB7", d_strToAlign, &chDesc, (size_t)strToAlignSize+1, (size_t)0 ) );


	unsigned arrSizes = numSeqs*sizeof(int);

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );

	const unsigned GRID_SIZE = 50;
	int *d_colMemory;
	//non va con 500 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_colMemory, (GRID_SIZE*MAX_BLOCK_SIZE)*MAX_LENGTH_SUPPORTED*3*sizeof(int) ) );

	//numero sequenze effettive
	unsigned numSeqsEff = stopPos - startPos + 1;

	unsigned numTotBlocks = numSeqsEff / MAX_BLOCK_SIZE;
	unsigned residueThreads = numSeqsEff % MAX_BLOCK_SIZE;

	//chiamata per il residuo

	if ( residueThreads != 0 && !(MAX_BLOCK_SIZE % residueThreads) ) {
		//se il residuo è un sottomultiplo di MAX_BLOCK_SIZE

		dim3  threads( residueThreads, 1, 1);

		swhandler6_global(1, residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, d_scores, d_colMemory);

	} else if (residueThreads != 0 && (MAX_BLOCK_SIZE % residueThreads)) {
		//se non è sottomultiplo di MAX_BLOCK_SIZE
		swhandler7_global(residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, d_scores, d_colMemory);
	}

	//chiamata per multipli di MAX_BLOCK_SIZE

	unsigned newStartPos = startPos + residueThreads;

	for (unsigned cnt=0; cnt<numTotBlocks;) {

		unsigned numBlocks = (cnt + GRID_SIZE > numTotBlocks) ? (numTotBlocks - cnt) : GRID_SIZE;

		dim3  grid( numBlocks, 1, 1);
		swhandler6_global(numBlocks, MAX_BLOCK_SIZE, d_strToAlign, sizeNotPad, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores, d_colMemory);

		cnt += numBlocks;
		newStartPos += numBlocks * MAX_BLOCK_SIZE;
	}

	CUT_SAFE_CALL( cutStopTimer( timer));
	//printf( "\nKernels processing time: %f (ms)\n", cutGetTimerValue( timer ));
	timeTot = cutGetTimerValue( timer );
	CUT_SAFE_CALL( cutDeleteTimer( timer));
	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );

	// cleanup memory
	CUDA_SAFE_CALL(cudaFree(d_colMemory));
	CUDA_SAFE_CALL(cudaFree(d_strToAlign));
	return timeTot;
}

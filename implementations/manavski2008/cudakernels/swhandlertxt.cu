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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include "sbtmatrix.h"

#include <sw_kernel_txt_1.h>

#define MAX_BLOCK_SIZE 32


//__________________________???????????????????????????????????
struct seqBlock {
	char blockArr[24][24];
};
//__________________________???????????????????????????????????

extern "C" double smithWatermanCudaTxt( const char* strToAlign, const unsigned sizeNotPad, seqBlock *seqlib, const unsigned linLibSize, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, int* h_scores ) {
/*
	CUT_CHECK_DEVICE();

	// allocate device memory
	char* d_strToAlign;
	unsigned strToAlignSize = strlen(strToAlign);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_strToAlign, strToAlignSize+1 ) );
	unsigned *d_offsets;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_offsets, numSeqs*sizeof(unsigned)) );
	unsigned *d_sizes;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_sizes, numSeqs*sizeof(unsigned)) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( d_strToAlign, strToAlign, strToAlignSize+1, cudaMemcpyHostToDevice) );

	unsigned arrSizes = numSeqs*sizeof(unsigned);
	CUDA_SAFE_CALL( cudaMemcpy( d_offsets, offsets, arrSizes, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( d_sizes, sizes, arrSizes, cudaMemcpyHostToDevice) );

	// allocate device memory for result
	int* d_scores;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_scores, numSeqs*sizeof(int) ));

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );

	//allocate array memory on the device and copy from the host
	cudaArray* d_libArr;
	CUDA_SAFE_CALL( cudaMallocArray( (void**) &d_libArr, &texLib.channelDesc, linLibSize*24, 24) );
	CUDA_SAFE_CALL( cudaMemcpyToArray( d_libArr, 0, 0, seqlib, linLibSize*24*24,cudaMemcpyHostToDevice) );

	//create and bind the texture
	texLib.normalized = false;
	CUDA_SAFE_CALL( cudaBindTexture( texLib, d_libArr); );
	
	unsigned int timer = 0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	//numero sequenze effettive
	unsigned numSeqsEff = stopPos - startPos + 1;

	unsigned numTotBlocks = numSeqsEff / MAX_BLOCK_SIZE;
	unsigned residueThreads = numSeqsEff % MAX_BLOCK_SIZE;

	nel risultato sbaglierà le prime 24/25 sequenze
	//chiamata per il residuo

	if ( residueThreads != 0 && !(MAX_BLOCK_SIZE % residueThreads) ) {
		//se il residuo è un sottomultiplo di MAX_BLOCK_SIZE

		dim3  grid( 1, 1, 1);
		dim3  threads( residueThreads, 1, 1);
	
		sw_kernel4<<< grid, threads, DIMSHAREDSPACE_4 >>>( d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, d_scores);
		CUT_CHECK_ERROR("Kernel execution failed");

	} else if (residueThreads != 0 && (MAX_BLOCK_SIZE % residueThreads)) {
		//se non è sottomultiplo di MAX_BLOCK_SIZE

		swhandler5(residueThreads, d_strToAlign, sizeNotPad, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, d_scores);
	}

	//chiamata per multipli di MAX_BLOCK_SIZE

	const unsigned GRID_SIZE = 500;
	unsigned newStartPos = startPos + residueThreads;

	for (unsigned cnt=0; cnt<numTotBlocks;) {

		unsigned numBlocks = (cnt + GRID_SIZE > numTotBlocks) ? (numTotBlocks - cnt) : GRID_SIZE;
		
		dim3  grid( numBlocks, 1, 1);
		dim3  threads( MAX_BLOCK_SIZE, 1, 1);

		sw_kernel_txt_1<<< grid, threads, DIMSHAREDSPACE_TXT_1 >>>( d_strToAlign, sizeNotPad, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);
		CUT_CHECK_ERROR("Kernel execution failed");

		cnt += numBlocks;
		newStartPos += numBlocks * MAX_BLOCK_SIZE;
	}

	CUT_SAFE_CALL( cutStopTimer( timer));
	//printf( "\nKernels processing time: %f (ms)\n", cutGetTimerValue( timer ));
	double timerTot = cutGetTimerValue( timer );
	CUT_SAFE_CALL( cutDeleteTimer( timer));
	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );

	// cleanup memory
	CUDA_SAFE_CALL(cudaFreeArray(d_libArr));
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sizes));
	CUDA_SAFE_CALL(cudaFree(d_offsets));
	CUDA_SAFE_CALL(cudaFree(d_strToAlign));

	return timerTot;
*/
	return 0;
}



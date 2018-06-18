
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

#include <e2g_krlscan.h>
#include <e2g_krlscanhashed.h>

extern "C" int searchMatches(unsigned word_len, unsigned numThreadBlocksX, WORD_CODE_TYPE *d_queries, WORD_CODE_TYPE *d_seqlib_composed, unsigned dbsize_blobks, unsigned rumorThreshold, unsigned *d_res_position, unsigned *d_res_numMatches, unsigned *d_res_position_rev, unsigned *d_res_numMatches_rev) {

    //cudaChannelFormatDesc channelDescMono = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaChannelFormatDesc channelDescBi = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned);

	texSUBJECT.normalized = false;
	texSBJBOTH.normalized = false;

	// for the composed texture
	unsigned texture_width = dbsize_blobks*NUM_THREADS;
	if (texture_width > MAX_TEXTURE_WIDTH) {
		dbsize_blobks = MAX_TEXTURE_WIDTH / NUM_THREADS;
		texture_width = dbsize_blobks*NUM_THREADS;
	}
	
	CUDA_SAFE_CALL( cudaBindTexture( 0, &texSBJBOTH, d_seqlib_composed, &channelDescBi, (size_t)texture_width*sizeof(WORD_CODE_TYPE)*2 ) );
		
	//CUDA_SAFE_CALL( cudaBindTexture( 0, &texSUBJECT, d_seqlib, &channelDescMono, (size_t)dbsize_blobks*NUM_THREADS*sizeof(WORD_CODE_TYPE)) );
	//CUDA_SAFE_CALL( cudaBindTexture( 0, &texRevSUBJECT, d_seqlib_rev, &channelDescMono, (size_t)dbsize_blobks*NUM_THREADS*sizeof(WORD_CODE_TYPE)) );

	//QTime timer_krl; timer_krl.start();

	//printf("num blocks: %u\n", numThreadBlocksX);
	
	//dim3  grid( 1, 1, 1);
	dim3  grid( numThreadBlocksX, 1, 1);
	dim3  threads( NUM_THREADS, 1, 1);
	//search_kernel_4<<< grid, threads >>>( d_queries, d_seqlib, dbsize_blobks, rumorThreshold, d_res_position, d_res_numMatches);
	
	if (word_len == 4)
		search_kernel_w4_v6<<< grid, threads >>>( d_queries, dbsize_blobks, rumorThreshold, d_res_position, d_res_numMatches, d_res_position_rev, d_res_numMatches_rev);
	else if (word_len == 6)
		search_kernel_w6_v2<<< grid, threads >>>( d_queries, dbsize_blobks, rumorThreshold, d_res_position, d_res_numMatches, d_res_position_rev, d_res_numMatches_rev);
	
	CUT_CHECK_ERROR("Kernel execution failed");

	//int last_time = timer_krl.elapsed();

	return 0;
}


extern "C" int searchMatchesHashed(unsigned word_len, unsigned numThreadBlocksX, WORD_CODE_TYPE *d_queries, unsigned *d_dbfwhashed, unsigned dbFwSize, unsigned *d_dbrevhashed, unsigned dbRevSize, unsigned *d_dboffsets, unsigned *d_dbsizes, unsigned maxWords, unsigned *d_res_position, unsigned *d_res_numMatches)
{

    cudaChannelFormatDesc channelDescSbjHashed = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaChannelFormatDesc channelDescRevSbjHashed = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaChannelFormatDesc channelDescOffsets = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);
    cudaChannelFormatDesc channelDescSizes = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);

	texSubjectHashed.normalized = false;
	texRevSubjectHashed.normalized = false;
	texOffsetsBoth.normalized = false;
	texSizesBoth.normalized = false;

	CUDA_SAFE_CALL( cudaBindTexture( 0, &texSubjectHashed, d_dbfwhashed, &channelDescSbjHashed, (size_t)dbFwSize*sizeof(unsigned) ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, &texRevSubjectHashed, d_dbrevhashed, &channelDescRevSbjHashed, (size_t)dbRevSize*sizeof(unsigned) ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, &texOffsetsBoth, d_dboffsets, &channelDescOffsets, (size_t)maxWords*sizeof(unsigned)*2 ) );
	CUDA_SAFE_CALL( cudaBindTexture( 0, &texSizesBoth, d_dbsizes, &channelDescSizes, (size_t)maxWords*sizeof(unsigned)*2 ) );
	
	//dim3  grid( 1, 1, 1);
	dim3  grid( numThreadBlocksX, 1, 1);
	dim3  threads( NUM_THREADS_HASHED, 1, 1);
	
	search_kernel_hashed_w6_v3<<< grid, threads >>>( d_queries, d_res_position, d_res_numMatches );
	
	CUT_CHECK_ERROR("Kernel execution failed");

	return 0;
}


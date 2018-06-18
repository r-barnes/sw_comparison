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

/* 
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL3_H_
#define _TEMPLATE_KERNEL3_H_


#define HSH( index )      CUT_BANK_CHECKER(Hsh, (index))
#define ESH( index )      CUT_BANK_CHECKER(Esh, (index))
#define FSH( index )      CUT_BANK_CHECKER(Fsh, (index))


#include "sbtmatrix.h"
#include "swutils.h"


__global__ void smithwatermanKernel_midantidiag_64threads( const char* g_strToAlign, const unsigned size_strToAlign, const char* g_seqlib, unsigned totBytesUsed, unsigned numSeqs, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_Hdata, int *g_Edata, int *g_Fdata, int *g_scores, unsigned Aoffset, unsigned forwardRunSteps) 
{
	const unsigned int NUM_THREADS = 64;

  
	// shared memory

	// 2 previous of H diagonals need to be recorded for the calculation
	// one initial 0 is always needed for border conditions
	// the last position of H is needed for max current scores
	__shared__  int Hsh[NUM_THREADS*2+3];
	__shared__  int Esh[NUM_THREADS+1];
	__shared__  int Fsh[NUM_THREADS+1];


	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// read the substitution matrix
	loadPAM_64threads(tid);

	unsigned sizeA = g_sizes[blid+seqOffset];
	unsigned sizeB = size_strToAlign;

	// read in input data from global memory
	// use the bank checker macro to check for bank conflicts during host
	// emulation
	unsigned libOffset = g_offsets[blid+seqOffset];
	
	//ogni thread carica 2 elementi di A che è costituita da 128 elementi
	ASEQ128(tid) = g_seqlib[ libOffset + (2+NUM_THREADS*(Aoffset-1)) + tid ];
	//l'ultimo thread carica roba inutile
	ASEQ128(tid+64) = g_seqlib[ libOffset + (66+NUM_THREADS*(Aoffset-1)) + tid ];


	//printf("A[%u] = %c, A[%u] = %c \n", tid, ASEQ128(tid), tid+64, ASEQ128(tid+64));

	// the one we search for (only for 64 for now)
	BSEQ128(tid) = g_strToAlign[sizeB - NUM_THREADS + tid ];
	
	HSH(tid+1) = g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + tid]; 
	HSH(NUM_THREADS+1 + tid + 1) = g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + NUM_THREADS + tid];

	ESH(tid+1) = g_Edata[(blid+seqOffset)*NUM_THREADS + tid];
	FSH(tid+1) = g_Fdata[(blid+seqOffset)*NUM_THREADS + tid]; 
	
	HSH(0) = 0; 
	HSH(NUM_THREADS+1) = 0;
	ESH(0) = 0; 
	FSH(0) = 0; 
	HSH(NUM_THREADS*2+2) = g_scores[blid+seqOffset];

	unsigned lastDiag;

	for (unsigned cnt=1; cnt<forwardRunSteps+1; cnt++) {

		lastDiag = 0; // switches between 0 and 1
		unsigned curDiagIdx = 64; 
	
		// la prima antidiagonale e' quella che per prima esclude un carattere delle due sequenze
		// to be curDiagIdx < 128....da generalizzare
		for (; curDiagIdx < 128; ++curDiagIdx) {
			__syncthreads();
	
			int h=0,e=0,f=0;
	
			swcalc_complete_64(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);
	
			lastDiag = 1 - lastDiag;
		
			__syncthreads();
			HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
			ESH(tid + 1) = e;
			FSH(tid + 1) = f;
	
			++curDiagIdx;
			__syncthreads();
	
			h=0,e=0,f=0;
			swcalc_complete_64(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);
	
			lastDiag = 1 - lastDiag;
		
			__syncthreads();
			HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
			ESH(tid + 1) = e;
			FSH(tid + 1) = f;
	
			SCORES(tid) = h;
			SCORES(NUM_THREADS + tid) = HSH( (1-lastDiag)*(NUM_THREADS+1) + tid + 1);
			__syncthreads();
	
			max128( tid, HSH(NUM_THREADS*2+2) );
		}

		ASEQ128(tid) = ASEQ128(tid+64);
		ASEQ128(tid+64) = g_seqlib[ libOffset + (66+NUM_THREADS*(Aoffset-1)) + (NUM_THREADS*cnt) + tid ];
	}


	// write data to global memory
	__syncthreads();
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + tid] = HSH(lastDiag*(NUM_THREADS+1) + tid + 1);
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + NUM_THREADS + tid] = HSH( (1-lastDiag)*(NUM_THREADS+1) + tid + 1);
	g_Edata[(blid+seqOffset)*NUM_THREADS + tid] = ESH(tid + 1);
	g_Fdata[(blid+seqOffset)*NUM_THREADS + tid] = FSH(tid + 1);
	g_scores[blid+seqOffset] = HSH(NUM_THREADS*2+2);

}

//if (ASEQ(idxSeqA) != '[' && BSEQ(idxSeqB) != '[') printf("tid=%u diag=%u %c-%c res=%d\t\t", tid, curDiagIdx, ASEQ(idxSeqA), BSEQ(idxSeqB), pam[ASEQ(idxSeqA)-60][BSEQ(idxSeqB)-60]);


#endif // #ifndef _TEMPLATE_KERNEL3_H_

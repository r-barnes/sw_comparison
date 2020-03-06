


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


/* 
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_



#define HSH( index )      CUT_BANK_CHECKER(Hsh, (index))
#define ESH( index )      CUT_BANK_CHECKER(Esh, (index))
#define FSH( index )      CUT_BANK_CHECKER(Fsh, (index))


#include "sbtmatrix.h"
#include "swutils.h"
#include "cutil.h"


//per emulazione: -Xcuda "-deviceemu" -cuda -D_DEVICEEMU


/*
////////////////////////////////////////////////////////////////////////////////
// Simple test kernel for device functionality
////////////////////////////////////////////////////////////////////////////////
__global__ void smithwatermanKernel_first65antidiag( const char* g_seqlib, unsigned totBytesUsed, unsigned numSeqs, 
unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta,
int *g_Hdata, int *g_Edata, int *g_Fdata, int *g_SCORES) 
{
	const unsigned int NUM_THREADS = 64;
  
	// shared memory

	// 2 previous of H diagonals need to be recorded for the calculation
	// one initial 0 is always needed for border conditions
	// the last position of H is needed for max current SCORES
	__shared__  int Hsh[(NUM_THREADS+1)*2+1];
	__shared__  int Esh[NUM_THREADS+1];
	__shared__  int Fsh[NUM_THREADS+1];

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	loadPAM_64threads(tid);

	// read in input data from global memory
	// use the bank checker macro to check for bank conflicts during host
	// emulation
	unsigned libOffset = g_offsets[blid+seqOffset];
	ASEQ64(tid) = g_seqlib[ libOffset + tid ];
	ASEQ64(NUM_THREADS) = g_seqlib[ libOffset + NUM_THREADS ];
	
	BSEQ64(tid) = g_seqlib[tid]; // the one we search for
	BSEQ64(NUM_THREADS) = g_seqlib[NUM_THREADS]; // the one we search for
	
	HSH(0) = 0; 
	HSH(NUM_THREADS+1) = 0;
	ESH(0) = 0; 
	FSH(0) = 0; 
	HSH(tid+1) = 0; 
	HSH(NUM_THREADS+1 + tid + 1) = 0;
	ESH(tid+1) = 0; 
	FSH(tid+1) = 0; 
	// per salvare il max di H
	HSH(NUM_THREADS*2+2) = 0;
	
	unsigned lastDiag = 1; // switches between 0 and 1
	unsigned curDiagIdx = 2; 

	// must be < 66 for the last diag
	for (; curDiagIdx < 66; ) {
		__syncthreads();

		int h=0,e=0,f=0;
		swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

		lastDiag = 1 - lastDiag;
	
		__syncthreads();
		HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
		ESH(tid + 1) = e;
		FSH(tid + 1) = f;

		// here we start max64 procedure
		SCORES(tid) = h;
		//SCORES(NUM_THREADS + tid) = 0;
		__syncthreads();

		max64( tid, HSH(NUM_THREADS*2+2) );

		++curDiagIdx;
	}	

	// write data to global memory
	__syncthreads();
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + tid] = HSH(lastDiag*(NUM_THREADS+1) + tid + 1);
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + NUM_THREADS + tid] = HSH( (1-lastDiag)*(NUM_THREADS+1) + tid + 1);
	g_Edata[(blid+seqOffset)*NUM_THREADS + tid] = ESH(tid + 1);
	g_Fdata[(blid+seqOffset)*NUM_THREADS + tid] = FSH(tid + 1);
	g_SCORES[blid+seqOffset] = HSH(NUM_THREADS*2+2);
}
*/



__global__ void smithwatermanKernel_first65antidiag( const char* g_strToAlign, const char* g_seqlib, unsigned totBytesUsed,  const unsigned numSeqs, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_Hdata, int *g_Edata, int *g_Fdata, int *g_SCORES)
{
	const unsigned int NUM_THREADS = 64;
  
	// shared memory

	// 2 previous of H diagonals need to be recorded for the calculation
	// one initial 0 is always needed for border conditions
	// the last position of H is needed for max current SCORES
	__shared__  int Hsh[(NUM_THREADS+1)*2+1];
	__shared__  int Esh[NUM_THREADS+1];
	__shared__  int Fsh[NUM_THREADS+1];

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	loadPAM_64threads(tid);
	// read in input data from global memory
	// use the bank checker macro to check for bank conflicts during host
	// emulation
	unsigned libOffset = g_offsets[blid+seqOffset];
	ASEQ64(tid) = g_seqlib[ libOffset + tid ];
	ASEQ64(NUM_THREADS) = g_seqlib[ libOffset + NUM_THREADS ];
	
	BSEQ64(tid) = g_strToAlign[tid]; // the one we search for
	BSEQ64(NUM_THREADS) = g_strToAlign[NUM_THREADS]; // the one we search for

	HSH(0) = 0; 
	HSH(NUM_THREADS+1) = 0;
	ESH(0) = 0; 
	FSH(0) = 0; 
	HSH(tid+1) = 0; 
	HSH(NUM_THREADS+1 + tid + 1) = 0;
	ESH(tid+1) = 0; 
	FSH(tid+1) = 0; 
	// per salvare il max di H
	HSH(NUM_THREADS*2+2) = 0;
	
	unsigned lastDiag = 1; // switches between 0 and 1
	unsigned curDiagIdx = 2;

	// loop unrolling curDiagIdx must be < 66 for the last diag
	
	//____________________________________________RUN_1____________________________________________
	// curDiagIdx = 2
	__syncthreads();

	int h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max1 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	HSH(NUM_THREADS*2+2) = SCORES(0);

	//________________________________________END_OF_RUN_1________________________________________


	//____________________________________________RUN_2____________________________________________
	curDiagIdx = 3;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max2 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	HSH(NUM_THREADS*2+2) = max(SCORES(0), SCORES(1));

	//________________________________________END_OF_RUN_2________________________________________

	//____________________________________________RUN_3____________________________________________
	curDiagIdx = 4;
	
	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max4 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max4( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_3________________________________________


	//____________________________________________RUN_4____________________________________________
	curDiagIdx = 5;
	
	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max4 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max4( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_4________________________________________


	//____________________________________________RUN_5____________________________________________
	curDiagIdx = 6;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max8 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max8( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_5________________________________________

	//____________________________________________RUN_6____________________________________________
	curDiagIdx = 7;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max8 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max8( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_6________________________________________


	//____________________________________________RUN_7____________________________________________
	curDiagIdx = 8;
	
	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max8 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max8( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_7________________________________________

	//____________________________________________RUN_8____________________________________________
	curDiagIdx = 9;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max8 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max8( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_8________________________________________


	//____________________________________________RUN_9____________________________________________
	curDiagIdx = 10;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_9________________________________________


	//____________________________________________RUN_10____________________________________________
	curDiagIdx = 11;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_10________________________________________

	//____________________________________________RUN_11____________________________________________
	curDiagIdx = 12;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_11________________________________________

	//____________________________________________RUN_12____________________________________________
	curDiagIdx = 13;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_12________________________________________

	//____________________________________________RUN_13____________________________________________
	curDiagIdx = 14;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_13________________________________________

	//____________________________________________RUN_14____________________________________________
	curDiagIdx = 15;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_14________________________________________

	//____________________________________________RUN_15____________________________________________
	curDiagIdx = 16;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_15________________________________________

	//____________________________________________RUN_16____________________________________________
	curDiagIdx = 17;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max16 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max16( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_16________________________________________


	//____________________________________________RUN_17____________________________________________
	curDiagIdx = 18;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_17________________________________________


	//____________________________________________RUN_18____________________________________________
	curDiagIdx = 19;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_18________________________________________


	//____________________________________________RUN_19____________________________________________
	curDiagIdx = 20;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_19________________________________________


	//____________________________________________RUN_20____________________________________________
	curDiagIdx = 21;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_20________________________________________


	//____________________________________________RUN_21____________________________________________
	curDiagIdx = 22;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_21________________________________________


	//____________________________________________RUN_22____________________________________________
	curDiagIdx = 23;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_22________________________________________


	//____________________________________________RUN_23____________________________________________
	curDiagIdx = 24;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_23________________________________________


	//____________________________________________RUN_24____________________________________________
	curDiagIdx = 25;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_24________________________________________


	//____________________________________________RUN_25____________________________________________
	curDiagIdx = 26;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_25________________________________________


	//____________________________________________RUN_26____________________________________________
	curDiagIdx = 27;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_26________________________________________


	//____________________________________________RUN_27____________________________________________
	curDiagIdx = 28;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_27________________________________________


	//____________________________________________RUN_28____________________________________________
	curDiagIdx = 29;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_28________________________________________


	//____________________________________________RUN_29____________________________________________
	curDiagIdx = 30;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_29________________________________________


	//____________________________________________RUN_30____________________________________________
	curDiagIdx = 31;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_30________________________________________


	//____________________________________________RUN_31____________________________________________
	curDiagIdx = 32;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_31________________________________________


	//____________________________________________RUN_32____________________________________________
	curDiagIdx = 33;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max32( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_32________________________________________


	//____________________________________________RUN_33____________________________________________
	curDiagIdx = 34;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_33________________________________________


	//____________________________________________RUN_34____________________________________________
	curDiagIdx = 35;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_34________________________________________

	//____________________________________________RUN_35____________________________________________
	curDiagIdx = 36;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_35________________________________________

	//____________________________________________RUN_36____________________________________________
	curDiagIdx = 37;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_36________________________________________

	//____________________________________________RUN_37____________________________________________
	curDiagIdx = 38;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_37________________________________________


	//____________________________________________RUN_38____________________________________________
	curDiagIdx = 39;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_38________________________________________


	//____________________________________________RUN_39____________________________________________
	curDiagIdx = 40;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_39________________________________________


	//____________________________________________RUN_40____________________________________________
	curDiagIdx = 41;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_40________________________________________


	//____________________________________________RUN_41____________________________________________
	curDiagIdx = 42;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_41________________________________________


	//____________________________________________RUN_42____________________________________________
	curDiagIdx = 43;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_42________________________________________


	//____________________________________________RUN_43____________________________________________
	curDiagIdx = 44;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_43________________________________________

	
	//____________________________________________RUN_44____________________________________________
	curDiagIdx = 45;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_44________________________________________

	
	//____________________________________________RUN_45____________________________________________
	curDiagIdx = 46;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_45________________________________________


	//____________________________________________RUN_46____________________________________________
	curDiagIdx = 47;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_46________________________________________


	//____________________________________________RUN_47____________________________________________
	curDiagIdx = 48;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_47________________________________________


	//____________________________________________RUN_48____________________________________________
	curDiagIdx = 49;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_48________________________________________


	//____________________________________________RUN_49____________________________________________
	curDiagIdx = 50;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_49________________________________________


	//____________________________________________RUN_50____________________________________________
	curDiagIdx = 51;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_50________________________________________


	//____________________________________________RUN_51____________________________________________
	curDiagIdx = 52;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_51________________________________________


	//____________________________________________RUN_52____________________________________________
	curDiagIdx = 53;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_52________________________________________


	//____________________________________________RUN_53____________________________________________
	curDiagIdx = 54;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_53________________________________________


	//____________________________________________RUN_54____________________________________________
	curDiagIdx = 55;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_54________________________________________


	//____________________________________________RUN_55____________________________________________
	curDiagIdx = 56;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_55________________________________________


	//____________________________________________RUN_56____________________________________________
	curDiagIdx = 57;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_56________________________________________


	//____________________________________________RUN_57____________________________________________
	curDiagIdx = 58;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_57________________________________________


	//____________________________________________RUN_58____________________________________________
	curDiagIdx = 59;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_58________________________________________


	//____________________________________________RUN_59____________________________________________
	curDiagIdx = 60;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_59________________________________________


	//____________________________________________RUN_60____________________________________________
	curDiagIdx = 61;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_60________________________________________


	//____________________________________________RUN_61____________________________________________
	curDiagIdx = 62;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_61________________________________________


	//____________________________________________RUN_62____________________________________________
	curDiagIdx = 63;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_62________________________________________


	//____________________________________________RUN_63____________________________________________
	curDiagIdx = 64;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	//________________________________________END_OF_RUN_63________________________________________


	//____________________________________________RUN_64____________________________________________
	curDiagIdx = 65;

	h=0,e=0,f=0;
	swcalc_upleft(curDiagIdx, HSH( lastDiag*(NUM_THREADS+1) + tid ), FSH( tid ), HSH( lastDiag*(NUM_THREADS+1)+ tid+ 1 ), ESH( tid + 1 ), HSH( (1 - lastDiag)*(NUM_THREADS+1) + tid  ), alpha, beta, h, e, f);

	lastDiag = 1 - lastDiag;
	
	__syncthreads();
	HSH(lastDiag*(NUM_THREADS+1) + tid + 1) = h;
	ESH(tid + 1) = e;
	FSH(tid + 1) = f;

	// here we start max32 procedure
	SCORES(tid) = h;
	//SCORES(NUM_THREADS + tid) = 0;
	__syncthreads();

	max64( tid, HSH(NUM_THREADS*2+2) );

	++curDiagIdx;
	//________________________________________END_OF_RUN_64________________________________________
	
	// write data to global memory
	__syncthreads();
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + tid] = HSH(lastDiag*(NUM_THREADS+1) + tid + 1);
	g_Hdata[(2*(blid+seqOffset))*NUM_THREADS + NUM_THREADS + tid] = HSH( (1-lastDiag)*(NUM_THREADS+1) + tid + 1);
	g_Edata[(blid+seqOffset)*NUM_THREADS + tid] = ESH(tid + 1);
	g_Fdata[(blid+seqOffset)*NUM_THREADS + tid] = FSH(tid + 1);
	g_SCORES[blid+seqOffset] = HSH(NUM_THREADS*2+2);
}


#endif // #ifndef _TEMPLATE_KERNEL_H_

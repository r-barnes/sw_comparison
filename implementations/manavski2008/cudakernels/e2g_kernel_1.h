
#ifndef _TEMPLATE_E2GKERNEL1_H_
#define _TEMPLATE_E2GKERNEL1_H_

#define HSH( index )		CUT_BANK_CHECKER(Hsh, (index))
#define ESH( index )		CUT_BANK_CHECKER(Esh, (index))

//extern  __shared__  short int int_sharedspace_sbt_solexa[];

// contiene: int pam[32][32]
//#define DIMSHAREDSPACE_6		((1024)*sizeof(short int))

#define PAM_6( mac_i, mac_j )	CUT_BANK_CHECKER(int_sharedspace_sbt_solexa, ((mac_i)*32 + (mac_j)))

#include "e2g_consts.h"


texture<unsigned char, 1, cudaReadModeElementType> tex_queries;
texture<unsigned char, 1, cudaReadModeElementType> tex_splice_sites;

//#include <stdio.h>
/*
__device__ void loadMAT_6_64threads(const int &tid)
{
	// read the substitution matrix
	unsigned idxp = tid/32, restp = tid%32;

	PAM_6(idxp, restp)				= const_e2g_sbtmat[(idxp*32+restp)];	//riga 0,1
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 2,3
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 4,5
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 6,7
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 8,9
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 10,11
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 12,13
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 14,15
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 16,17
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 18,19
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 20,21
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 22,23
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 24,25
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 26,27
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 28,29
	idxp += 2; PAM_6(idxp, restp)	= const_e2g_sbtmat[(idxp*32+restp)];	//riga 30,31
}
*/
__device__ void swcalc_kernel_solexa(const char &aElem, const char &bElem, const short int &h_n, const short int &f_n, const short int &h_w, const short int &e_w, const short int &h_nw, const short unsigned int &alpha, const short unsigned int &beta, short int &h, short int &e, short int &f)
{
	short int tmp1 = h_n - alpha;
	short int tmp2 = f_n - beta;
	f = max(tmp1, tmp2);
	
	tmp1 = h_w - alpha;
	tmp2 = e_w - beta;
	e = max(tmp1, tmp2);
	
	tmp1 = aElem - 60;
	tmp2 = bElem - 60;

	h = h_nw + const_e2g_sbtmat[(tmp1*32+tmp2)]; //PAM_6(tmp1, tmp2); ( ( tmp1 != tmp2) ? -4 : 13 )
	h = max(0, h);
	h = max(e, h);
	h = max(f, h);
}


/// MUST FIX IT TO WORK WITH SEQUENCES WITHOUT STARTING '@' !!!!!
__global__ void solexa_kernel( const char* g_seqlib, unsigned *g_offsets, unsigned *g_sizes, short unsigned first_gap_penalty, short unsigned next_gap_penalty, short unsigned splice_penalty, short unsigned intron_penalty, int *g_scores) 
{
	// shared memory
	//needed 1 column of the dyn programming matrix for H and one for F
	__shared__  short int	Hsh[MAX_NUM_THREADS*(SOLEXA_QUERY_SIZE+1)];
	__shared__  short int	Esh[MAX_NUM_THREADS*(SOLEXA_QUERY_SIZE+1)];

	__shared__  short int best_intron_coord[MAX_NUM_THREADS*(SOLEXA_QUERY_SIZE+1)];
	__shared__  short int best_intron_score[MAX_NUM_THREADS*(SOLEXA_QUERY_SIZE+1)];
	
	// set thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;
	unsigned columnOffset = tid * (SOLEXA_QUERY_SIZE+1);

	for (unsigned j=0; j<=SOLEXA_QUERY_SIZE; ++j) best_intron_coord[(columnOffset+j)] = 0;
	for (unsigned j=0; j<=SOLEXA_QUERY_SIZE; ++j) best_intron_score[(columnOffset+j)] = 0;
	

	// read the substitution matrix
	//loadMAT_6_64threads(tid);

	//index of the of pair to compute
	unsigned seqNum = (blid*MAX_NUM_THREADS) + tid;

	//offset of the DB seq of that pair
	unsigned libOffset = g_offsets[seqNum];

	unsigned sizeA = g_sizes[seqNum];

	__syncthreads();

	//variabili locali
	int score = 0;

	short int f=0, e=0, h=0;
	short int h_nord=0;
	short int e_nord=0;
	short int f_nord=0;
	short int intron = 0;
	char bElem, aElem, spliceElem;
	unsigned tempVar=0;
	bool is_acceptor;

	HSH(columnOffset) = 0; HSH(columnOffset+1) = 0; HSH(columnOffset+2) = 0; HSH(columnOffset+3) = 0;
	HSH(columnOffset+4) = 0; HSH(columnOffset+5) = 0; HSH(columnOffset+6) = 0; HSH(columnOffset+7) = 0;
	HSH(columnOffset+8) = 0; HSH(columnOffset+9) = 0; HSH(columnOffset+10) = 0; HSH(columnOffset+11) = 0;
	HSH(columnOffset+12) = 0; HSH(columnOffset+13) = 0; HSH(columnOffset+14) = 0; HSH(columnOffset+15) = 0;
	HSH(columnOffset+16) = 0; HSH(columnOffset+17) = 0; HSH(columnOffset+18) = 0; HSH(columnOffset+19) = 0;
	HSH(columnOffset+20) = 0; HSH(columnOffset+21) = 0; HSH(columnOffset+22) = 0; HSH(columnOffset+23) = 0;
	HSH(columnOffset+24) = 0; HSH(columnOffset+25) = 0; HSH(columnOffset+26) = 0; HSH(columnOffset+27) = 0;
	HSH(columnOffset+28) = 0; HSH(columnOffset+29) = 0; HSH(columnOffset+30) = 0; HSH(columnOffset+31) = 0;
	HSH(columnOffset+32) = 0; HSH(columnOffset+33) = 0; HSH(columnOffset+33) = 0; HSH(columnOffset+34) = 0;
	HSH(columnOffset+35) = 0; HSH(columnOffset+36) = 0;

	ESH(columnOffset) = 0; ESH(columnOffset+1) = 0; ESH(columnOffset+2) = 0; ESH(columnOffset+3) = 0;
	ESH(columnOffset+4) = 0; ESH(columnOffset+5) = 0; ESH(columnOffset+6) = 0; ESH(columnOffset+7) = 0;
	ESH(columnOffset+8) = 0; ESH(columnOffset+9) = 0; ESH(columnOffset+10) = 0; ESH(columnOffset+11) = 0;
	ESH(columnOffset+12) = 0; ESH(columnOffset+13) = 0; ESH(columnOffset+14) = 0; ESH(columnOffset+15) = 0;
	ESH(columnOffset+16) = 0; ESH(columnOffset+17) = 0; ESH(columnOffset+18) = 0; ESH(columnOffset+19) = 0;
	ESH(columnOffset+20) = 0; ESH(columnOffset+21) = 0; ESH(columnOffset+22) = 0; ESH(columnOffset+23) = 0;
	ESH(columnOffset+24) = 0; ESH(columnOffset+25) = 0; ESH(columnOffset+26) = 0; ESH(columnOffset+27) = 0;
	ESH(columnOffset+28) = 0; ESH(columnOffset+29) = 0; ESH(columnOffset+30) = 0; ESH(columnOffset+31) = 0;
	ESH(columnOffset+32) = 0; ESH(columnOffset+33) = 0; ESH(columnOffset+33) = 0; ESH(columnOffset+34) = 0;
	ESH(columnOffset+35) = 0; ESH(columnOffset+36) = 0;
	

	// external cycle goes through the subject query
	for (unsigned j=1; j<sizeA; j++) {
		tempVar = columnOffset;
		HSH(tempVar) 			= 0;
		ESH(tempVar) 			= 0;
		aElem		= g_seqlib[ (libOffset + j) ];

		// j is last base of putative intron 
		spliceElem = tex1Dfetch(tex_splice_sites, j);
		is_acceptor = ( spliceElem & ACCEPTOR ) ? true : false;

		h_nord = 0;
		e_nord = 0;
		f_nord = 0;
		
		for (unsigned i=1; i<SOLEXA_QUERY_SIZE; i++) { 
			
			//taking the single element of the query sequence
			bElem = tex1Dfetch(tex_queries, i);
				
			++tempVar;
				
			swcalc_kernel_solexa(aElem, bElem, h_nord, f_nord, HSH(tempVar), ESH(tempVar), HSH(tempVar-1), first_gap_penalty, next_gap_penalty, h, e, f);


			int bic_epos = best_intron_coord[(columnOffset+i)];
			spliceElem = tex1Dfetch(tex_splice_sites, bic_epos);
			
			
			if (is_acceptor && (spliceElem & DONOR))
				intron = best_intron_score[(columnOffset+i)] - splice_penalty;
			else
				intron = best_intron_score[(columnOffset+i)] - intron_penalty;

			h = max(h, intron);
			
			if ( best_intron_score[(columnOffset+i)] < h ) {
				best_intron_score[(columnOffset+i)] = h;
				best_intron_coord[(columnOffset+i)] = j;
			}

			score = max(score, h);

			HSH(tempVar-1) = h_nord;
			ESH(tempVar-1) = e_nord;
			h_nord = h;
			e_nord = e;
			f_nord = f;

		}
	
		HSH(tempVar) = h_nord;
		ESH(tempVar) = e_nord;
	}

	// write data to global memory
	g_scores[ blid*MAX_NUM_THREADS + tid ] = score;
}

#endif

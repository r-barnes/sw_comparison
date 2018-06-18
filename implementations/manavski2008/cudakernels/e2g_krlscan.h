
#ifndef _TEMPLATE_KERNEL6_H_
#define _TEMPLATE_KERNEL6_H_

#include "e2g_krlconsts.h"

#define SBJ( index )		CUT_BANK_CHECKER(SBJ_sh, (index))
#define SBJREV( index )		CUT_BANK_CHECKER(SBJREV_sh, (index))
#define QRY( index )		CUT_BANK_CHECKER(QRY_sh, (index))
#define RES( index )		CUT_BANK_CHECKER(RES_sh, (index))

texture<unsigned short, 1, cudaReadModeElementType> texSUBJECT;
texture<unsigned short, 1, cudaReadModeElementType> texRevSUBJECT;

texture<ushort2, 1, cudaReadModeElementType> texSBJBOTH;


/**
	CORRECT: ?? (need to check cases when queries contain repeated AAAAA...) 
	DOES RECORD WHICH PART OF THE QUERY MATCHES IN THE POSITIONS OF THE SUBJECT
	TIMES: 64 threads x 400 blocks OR 128 threads x 200 blocks gives 5ms per query on 8800GT (DB size is 10MB)
	THIS KERNEL REQUIRES CUDA DEVICE 1.1!!!!!!!!!!!!!!!!
	expects the queries to have 5 words of 6 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_w6_v2( const WORD_CODE_TYPE* g_queries, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned *g_res_position, unsigned *g_res_numMatches, unsigned *g_res_position_rev, unsigned *g_res_numMatches_rev) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
	__shared__  unsigned	SBJREV_sh[NUM_THREADS*2];
	
	unsigned reg_result = 0, reg_result_rev = 0;
	unsigned maxRealResults = MAX_RESULTS_PER_QUERY-2;
	unsigned qres_offset = blid*MAX_RESULTS_PER_QUERY;
	
	ushort2 txtval = tex1Dfetch(texSBJBOTH, tid);
	SBJ(NUM_THREADS + tid) = txtval.x;
	SBJREV(NUM_THREADS + tid) = txtval.y;
	
	// every block works on one query
	unsigned qry_0 = g_queries[ (blid*QUERY_LENGTH6 ) ];
	unsigned qry_1 = g_queries[ (blid*QUERY_LENGTH6 +1) ];
	unsigned qry_2 = g_queries[ (blid*QUERY_LENGTH6 +2) ];
	unsigned qry_3 = g_queries[ (blid*QUERY_LENGTH6 +3) ];
	unsigned qry_4 = g_queries[ (blid*QUERY_LENGTH6 +4) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		txtval = tex1Dfetch(texSBJBOTH, i*NUM_THREADS + NUM_THREADS + tid); 
		
		__syncthreads();
		SBJ(tid)						= SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid)			= txtval.x;

		SBJREV(tid)						= SBJREV(NUM_THREADS + tid);
		SBJREV(NUM_THREADS + tid)		= txtval.y;
		__syncthreads();

		// every thread compares the query to its offset 
		reg_result =  ( SBJ(tid)    != qry_0 ) ? 0 : 0x1001;
		reg_result += ( SBJ(tid+6)  != qry_1 ) ? 0 : 0x1101;
		reg_result += ( SBJ(tid+12) != qry_2 ) ? 0 :  0x101;
		reg_result += ( SBJ(tid+18) != qry_3 ) ? 0 :  0x111;
		reg_result += ( SBJ(tid+24) != qry_4 ) ? 0 :   0x11;
		
		// REVERSE, every thread compares the query to its offset 
		reg_result_rev =  ( SBJREV(tid)    != qry_0 ) ? 0 : 0x1001;
		reg_result_rev += ( SBJREV(tid+6)  != qry_1 ) ? 0 : 0x1101;
		reg_result_rev += ( SBJREV(tid+12) != qry_2 ) ? 0 :  0x101;
		reg_result_rev += ( SBJREV(tid+18) != qry_3 ) ? 0 :  0x111;
		reg_result_rev += ( SBJREV(tid+24) != qry_4 ) ? 0 :   0x11;

		if ( (reg_result & 0xF) > rumorThreshold ) {
			unsigned count_matches = atomicInc( &(g_res_numMatches[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position[ (qres_offset + count_matches+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches[ (qres_offset + count_matches+1) ] = reg_result;
		}
		if ( (reg_result_rev & 0xF) > rumorThreshold ) {
			unsigned count_matches_rev = atomicInc( &(g_res_numMatches_rev[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position_rev[ (qres_offset + count_matches_rev+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches_rev[ (qres_offset + count_matches_rev+1) ] = reg_result_rev;
		}
		
	}

}


/**
	CORRECT: OK (need to check cases when queries contain repeated AAAAA...) 
	DOES FORWARD AND REVERSE IN a SINGLE TEXTURE READ
	TIMES: 64 threads x 400 blocks OR 128 threads x 200 blocks gives 5ms per query on 8800GT (DB size is 10MB)
	THIS KERNEL REQUIRES CUDA DEVICE 1.1!!!!!!!!!!!!!!!!
	expects the queries to have 8 words of 4 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_w6_v1( const WORD_CODE_TYPE* g_queries, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned *g_res_position, unsigned *g_res_numMatches, unsigned *g_res_position_rev, unsigned *g_res_numMatches_rev) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
	__shared__  unsigned	SBJREV_sh[NUM_THREADS*2];
	
	unsigned reg_result = 0, reg_result_rev = 0;
	unsigned maxRealResults = MAX_RESULTS_PER_QUERY-2;
	unsigned qres_offset = blid*MAX_RESULTS_PER_QUERY;
	
	ushort2 txtval = tex1Dfetch(texSBJBOTH, tid);
	SBJ(NUM_THREADS + tid) = txtval.x;
	SBJREV(NUM_THREADS + tid) = txtval.y;
	
	// every block works on one query
	unsigned qry_0 = g_queries[ (blid*QUERY_LENGTH6 ) ];
	unsigned qry_1 = g_queries[ (blid*QUERY_LENGTH6 +1) ];
	unsigned qry_2 = g_queries[ (blid*QUERY_LENGTH6 +2) ];
	unsigned qry_3 = g_queries[ (blid*QUERY_LENGTH6 +3) ];
	unsigned qry_4 = g_queries[ (blid*QUERY_LENGTH6 +4) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		txtval = tex1Dfetch(texSBJBOTH, i*NUM_THREADS + NUM_THREADS + tid); 
		
		__syncthreads();
		SBJ(tid)						= SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid)			= txtval.x;

		SBJREV(tid)						= SBJREV(NUM_THREADS + tid);
		SBJREV(NUM_THREADS + tid)		= txtval.y;
		__syncthreads();

		// every thread compares the query to its offset 
		reg_result =  ( SBJ(tid)    != qry_0 ) ? 0 : 1;
		reg_result += ( SBJ(tid+6)  != qry_1 ) ? 0 : 1;
		reg_result += ( SBJ(tid+12) != qry_2 ) ? 0 : 1;
		reg_result += ( SBJ(tid+18) != qry_3 ) ? 0 : 1;
		reg_result += ( SBJ(tid+24) != qry_4 ) ? 0 : 1;
		
		// REVERSE, every thread compares the query to its offset 
		reg_result_rev =  ( SBJREV(tid)    != qry_0 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+6)  != qry_1 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+12) != qry_2 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+18) != qry_3 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+24) != qry_4 ) ? 0 : 1;

		if ( reg_result > rumorThreshold ) {
			unsigned count_matches = atomicInc( &(g_res_numMatches[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position[ (qres_offset + count_matches+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches[ (qres_offset + count_matches+1) ] = reg_result;
		}
		if ( reg_result_rev > rumorThreshold ) {
			unsigned count_matches_rev = atomicInc( &(g_res_numMatches_rev[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position_rev[ (qres_offset + count_matches_rev+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches_rev[ (qres_offset + count_matches_rev+1) ] = reg_result_rev;
		}
		
	}

}


/**
	CORRECT: OK (need to check cases when queries contain repeated AAAAA...) 
	DOES FORWARD AND REVERSE IN a SINGLE TEXTURE READ
	TIMES: 64 threads x 400 blocks OR 128 threads x 200 blocks gives 5ms per query on 8800GT (DB size is 10MB)
	THIS KERNEL REQUIRES CUDA DEVICE 1.1!!!!!!!!!!!!!!!!
	expects the queries to have 8 words of 4 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_w4_v6( const WORD_CODE_TYPE* g_queries, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned *g_res_position, unsigned *g_res_numMatches, unsigned *g_res_position_rev, unsigned *g_res_numMatches_rev) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
	__shared__  unsigned	SBJREV_sh[NUM_THREADS*2];
	
	unsigned reg_result = 0, reg_result_rev = 0;
	unsigned maxRealResults = MAX_RESULTS_PER_QUERY-2;
	unsigned qres_offset = blid*MAX_RESULTS_PER_QUERY;
	
	ushort2 txtval = tex1Dfetch(texSBJBOTH, tid);
	SBJ(NUM_THREADS + tid) = txtval.x;
	SBJREV(NUM_THREADS + tid) = txtval.y;
	
	// every block works on one query
	unsigned qry_0 = g_queries[ (blid*QUERY_LENGTH4 ) ];
	unsigned qry_1 = g_queries[ (blid*QUERY_LENGTH4 +1) ];
	unsigned qry_2 = g_queries[ (blid*QUERY_LENGTH4 +2) ];
	unsigned qry_3 = g_queries[ (blid*QUERY_LENGTH4 +3) ];
	unsigned qry_4 = g_queries[ (blid*QUERY_LENGTH4 +4) ];
	unsigned qry_5 = g_queries[ (blid*QUERY_LENGTH4 +5) ];
	unsigned qry_6 = g_queries[ (blid*QUERY_LENGTH4 +6) ];
	unsigned qry_7 = g_queries[ (blid*QUERY_LENGTH4 +7) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		txtval = tex1Dfetch(texSBJBOTH, i*NUM_THREADS + NUM_THREADS + tid); 
		
		__syncthreads();
		SBJ(tid)						= SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid)			= txtval.x;

		SBJREV(tid)						= SBJREV(NUM_THREADS + tid);
		SBJREV(NUM_THREADS + tid)		= txtval.y;
		__syncthreads();

		// every thread compares the query to its offset 
		reg_result =  ( SBJ(tid)    != qry_0 ) ? 0 : 1;
		reg_result += ( SBJ(tid+4)  != qry_1 ) ? 0 : 1;
		reg_result += ( SBJ(tid+8)  != qry_2 ) ? 0 : 1;
		reg_result += ( SBJ(tid+12) != qry_3 ) ? 0 : 1;
		reg_result += ( SBJ(tid+16) != qry_4 ) ? 0 : 1;
		reg_result += ( SBJ(tid+20) != qry_5 ) ? 0 : 1;
		reg_result += ( SBJ(tid+24) != qry_6 ) ? 0 : 1;
		reg_result += ( SBJ(tid+28) != qry_7 ) ? 0 : 1;
		
		// REVERSE, every thread compares the query to its offset 
		reg_result_rev =  ( SBJREV(tid)    != qry_0 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+4)  != qry_1 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+8)  != qry_2 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+12) != qry_3 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+16) != qry_4 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+20) != qry_5 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+24) != qry_6 ) ? 0 : 1;
		reg_result_rev += ( SBJREV(tid+28) != qry_7 ) ? 0 : 1;

		if ( reg_result > rumorThreshold ) {
			unsigned count_matches = atomicInc( &(g_res_numMatches[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position[ (qres_offset + count_matches+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches[ (qres_offset + count_matches+1) ] = reg_result;
		}
		if ( reg_result_rev > rumorThreshold ) {
			unsigned count_matches_rev = atomicInc( &(g_res_numMatches_rev[(qres_offset)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position_rev[ (qres_offset + count_matches_rev+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches_rev[ (qres_offset + count_matches_rev+1) ] = reg_result_rev;
		}
		
	}

}






/**
	CORRECT: OK ONLY FORWARD DIRECTION (need to check cases when queries contain repeated AAAAA...)
	TIMES: 
	THIS KERNEL REQUIRES CUDA DEVICE 1.1!!!!!!!!!!!!!!!!
	expects the queries to have 8 words of 4 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_w4_v4( const WORD_CODE_TYPE* g_queries, const WORD_CODE_TYPE* g_seqlib, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned *g_res_position, unsigned *g_res_numMatches) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
		
	unsigned reg_result = 0;
	unsigned maxRealResults = MAX_RESULTS_PER_QUERY-2;
	
	SBJ(NUM_THREADS + tid) = tex1Dfetch(texSUBJECT, tid); //g_seqlib[tid];
	
	__syncthreads();

	// every block works on one query
	unsigned qry_0 = g_queries[ (blid*QUERY_LENGTH4 ) ];
	unsigned qry_1 = g_queries[ (blid*QUERY_LENGTH4 +1) ];
	unsigned qry_2 = g_queries[ (blid*QUERY_LENGTH4 +2) ];
	unsigned qry_3 = g_queries[ (blid*QUERY_LENGTH4 +3) ];
	unsigned qry_4 = g_queries[ (blid*QUERY_LENGTH4 +4) ];
	unsigned qry_5 = g_queries[ (blid*QUERY_LENGTH4 +5) ];
	unsigned qry_6 = g_queries[ (blid*QUERY_LENGTH4 +6) ];
	unsigned qry_7 = g_queries[ (blid*QUERY_LENGTH4 +7) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		SBJ(tid) = SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid) = tex1Dfetch(texSUBJECT, i*NUM_THREADS + NUM_THREADS + tid); // g_seqlib[i*NUM_THREADS + NUM_THREADS + tid];
		__syncthreads();

		// every thread compares the query to its offset 

		reg_result =  ( SBJ(tid)    != qry_0 ) ? 0 : 1;
		reg_result += ( SBJ(tid+4)  != qry_1 ) ? 0 : 1;
		reg_result += ( SBJ(tid+8)  != qry_2 ) ? 0 : 1;
		reg_result += ( SBJ(tid+12) != qry_3 ) ? 0 : 1;
		reg_result += ( SBJ(tid+16) != qry_4 ) ? 0 : 1;
		reg_result += ( SBJ(tid+20) != qry_5 ) ? 0 : 1;
		reg_result += ( SBJ(tid+24) != qry_6 ) ? 0 : 1;
		reg_result += ( SBJ(tid+28) != qry_7 ) ? 0 : 1;

		if ( reg_result > rumorThreshold ) {
			unsigned count_matches = atomicInc( &(g_res_numMatches[(blid*MAX_RESULTS_PER_QUERY)]), maxRealResults );
			// add counter of my query if thr counter is < maxResultsPerQuery
			// save results
			g_res_position[ (blid*MAX_RESULTS_PER_QUERY + count_matches+1) ] = i*NUM_THREADS + tid;
			g_res_numMatches[ (blid*MAX_RESULTS_PER_QUERY + count_matches+1) ] = reg_result;
		}
		
	}

}




/**
	CORRECT: OK
	TIMES: 8800GTX -> 27 ms / query, 8600M -> 500ms / query
	expects the queries to have 8 words of 4 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_w4_v3( const WORD_CODE_TYPE* g_queries, const WORD_CODE_TYPE* g_seqlib, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned maxResultsPerQuery, unsigned *g_res_position, unsigned *g_res_numMatches) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
	__shared__  unsigned	QRY_sh[QUERY_LENGTH4];
	__shared__  unsigned	RES_sh[NUM_THREADS];
	
	unsigned short count_matches = 0;
	
	SBJ(NUM_THREADS + tid) = tex1Dfetch(texSUBJECT, tid); //g_seqlib[tid];
	
	__syncthreads();
		
	// every block works on one query
	QRY(0) = g_queries[ (blid*QUERY_LENGTH4 ) ];
	QRY(1) = g_queries[ (blid*QUERY_LENGTH4 +1) ];
	QRY(2) = g_queries[ (blid*QUERY_LENGTH4 +2) ];
	QRY(3) = g_queries[ (blid*QUERY_LENGTH4 +3) ];
	QRY(4) = g_queries[ (blid*QUERY_LENGTH4 +4) ];
	QRY(5) = g_queries[ (blid*QUERY_LENGTH4 +5) ];
	QRY(6) = g_queries[ (blid*QUERY_LENGTH4 +6) ];
	QRY(7) = g_queries[ (blid*QUERY_LENGTH4 +7) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		SBJ(tid) = SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid) = tex1Dfetch(texSUBJECT, i*NUM_THREADS + NUM_THREADS + tid); // g_seqlib[i*NUM_THREADS + NUM_THREADS + tid];

		// every thread compares the query to its offset 
			
		RES(tid) =  ( SBJ(tid)    != QRY(0) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+4)  != QRY(1) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+8)  != QRY(2) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+12) != QRY(3) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+16) != QRY(4) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+20) != QRY(5) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+24) != QRY(6) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+28) != QRY(7) ) ? 0 : 1;
		
		if (!tid) {
			for (unsigned k=0; k<NUM_THREADS; ++k) {

				if ( RES(k) > rumorThreshold && count_matches < maxResultsPerQuery) {
					// add counter of my query if thr counter is < maxResultsPerQuery
					// save results
					g_res_position[ (blid*maxResultsPerQuery + count_matches) ] = i*NUM_THREADS + k;
					g_res_numMatches[ (blid*maxResultsPerQuery + count_matches) ] = RES(k);
					++count_matches;
				}
			}
		} 
		
		__syncthreads();
	}

}


/**
	CORRECT: OK
	TIMES: 8800GTX -> 27 ms / query, 8600M -> 500ms / query

	expects the queries to have 8 words of 4 bytes each
	the number of queries must be == the number of thread blocks ( each block does one query)
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
	maxResultsPerQuery - maximum number of result structures to be saved per each query
 */
__global__ void search_kernel_w4_v2( const WORD_CODE_TYPE* g_queries, const WORD_CODE_TYPE* g_seqlib, const unsigned dbsize_blobks, unsigned rumorThreshold, unsigned maxResultsPerQuery, unsigned *g_res_position, unsigned *g_res_numMatches) {

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// shared memory
	__shared__  unsigned	SBJ_sh[NUM_THREADS*2];
	__shared__  unsigned	QRY_sh[QUERY_LENGTH4];
	__shared__  unsigned	RES_sh[NUM_THREADS];
	
	unsigned short count_matches = 0;
	
	SBJ(NUM_THREADS + tid) = g_seqlib[tid];
	
	__syncthreads();
		
	// every block works on one query
	QRY(0) = g_queries[ (blid*QUERY_LENGTH4 ) ];
	QRY(1) = g_queries[ (blid*QUERY_LENGTH4 +1) ];
	QRY(2) = g_queries[ (blid*QUERY_LENGTH4 +2) ];
	QRY(3) = g_queries[ (blid*QUERY_LENGTH4 +3) ];
	QRY(4) = g_queries[ (blid*QUERY_LENGTH4 +4) ];
	QRY(5) = g_queries[ (blid*QUERY_LENGTH4 +5) ];
	QRY(6) = g_queries[ (blid*QUERY_LENGTH4 +6) ];
	QRY(7) = g_queries[ (blid*QUERY_LENGTH4 +7) ];
	
	for(unsigned i=0; i<dbsize_blobks; ++i ) { 
		SBJ(tid) = SBJ(NUM_THREADS + tid);
		SBJ(NUM_THREADS + tid) = g_seqlib[i*NUM_THREADS + NUM_THREADS + tid];

		// every thread compares the query to its offset 
			
		RES(tid) =  ( SBJ(tid)    != QRY(0) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+4)  != QRY(1) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+8)  != QRY(2) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+12) != QRY(3) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+16) != QRY(4) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+20) != QRY(5) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+24) != QRY(6) ) ? 0 : 1;
		RES(tid) += ( SBJ(tid+28) != QRY(7) ) ? 0 : 1;
		
		if (!tid) {
			for (unsigned k=0; k<NUM_THREADS; ++k) {

				if ( RES(k) > rumorThreshold && count_matches < maxResultsPerQuery) {
					// add counter of my query if thr counter is < maxResultsPerQuery
					// save results
					g_res_position[ (blid*maxResultsPerQuery + count_matches) ] = i*NUM_THREADS + k;
					g_res_numMatches[ (blid*maxResultsPerQuery + count_matches) ] = RES(k);
					++count_matches;
				}
			}
		} 
		
		__syncthreads();
	}

}




#endif


#ifndef _TEMPLATE_SERACHKRLHASHED_H_
#define _TEMPLATE_SERACHKRLHASHED_H_

#include "e2g_krlconsts.h"

texture<unsigned, 1, cudaReadModeElementType> texSubjectHashed;
texture<unsigned, 1, cudaReadModeElementType> texRevSubjectHashed;
texture<uint2, 1, cudaReadModeElementType> texOffsetsBoth;
texture<uint2, 1, cudaReadModeElementType> texSizesBoth;



/**
	CORRECT: ??
	
	g_queries - list of the queries each 8 words long, total number of NUM_BLOCKS queries
	g_seqlib - the database in encoded words
	dbsize_blocks - number of the words in the database / NUM_THREADS (adjusted to keep the whole db)
	rumorThreshold - rumor threshold
 */
__global__ void search_kernel_hashed_w6_v3( const WORD_CODE_TYPE* g_queries, unsigned *g_res_position, unsigned *g_res_numMatches)
{
	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;
	const unsigned int totThreads = blockDim.x;

	unsigned query_idx = blid * totThreads + tid;
	unsigned qres_offset = query_idx*MAX_RESULTS_PER_QUERY_HASHED*2;
	
	unsigned totResFw = 0, totResRev = 0;
	const unsigned limitRes = MAX_RESULTS_PER_QUERY_HASHED - 2;
	
	// every block works on one query
	unsigned qry_0 = g_queries[ (query_idx*QUERY_LENGTH6 ) ];
	unsigned qry_1 = g_queries[ (query_idx*QUERY_LENGTH6 +1) ];
	unsigned qry_2 = g_queries[ (query_idx*QUERY_LENGTH6 +2) ];
	unsigned qry_3 = g_queries[ (query_idx*QUERY_LENGTH6 +3) ];
	unsigned qry_4 = g_queries[ (query_idx*QUERY_LENGTH6 +4) ];

	__syncthreads();

	uint2 offsetsVals = tex1Dfetch(texOffsetsBoth, qry_0);
	uint2 sizesVals = tex1Dfetch(texSizesBoth, qry_0);
	unsigned offsetFwQ0 = offsetsVals.x;
	unsigned offsetRvQ0 = offsetsVals.y;
	unsigned endFwQ0 = offsetsVals.x + sizesVals.x;
	unsigned endRvQ0 = offsetsVals.y + sizesVals.y;

	offsetsVals = tex1Dfetch(texOffsetsBoth, qry_1);
	sizesVals = tex1Dfetch(texSizesBoth, qry_1);
	unsigned offsetFwQ1 = offsetsVals.x;
	unsigned offsetRvQ1 = offsetsVals.y;
	unsigned endFwQ1 = offsetsVals.x + sizesVals.x;
	unsigned endRvQ1 = offsetsVals.y + sizesVals.y;

	offsetsVals = tex1Dfetch(texOffsetsBoth, qry_2);
	sizesVals = tex1Dfetch(texSizesBoth, qry_2);
	unsigned offsetFwQ2 = offsetsVals.x;
	unsigned offsetRvQ2 = offsetsVals.y;
	unsigned endFwQ2 = offsetsVals.x + sizesVals.x;
	unsigned endRvQ2 = offsetsVals.y + sizesVals.y;

	offsetsVals = tex1Dfetch(texOffsetsBoth, qry_3);
	sizesVals = tex1Dfetch(texSizesBoth, qry_3);
	unsigned offsetFwQ3 = offsetsVals.x;
	unsigned offsetRvQ3 = offsetsVals.y;
	unsigned endFwQ3 = offsetsVals.x + sizesVals.x;
	unsigned endRvQ3 = offsetsVals.y + sizesVals.y;

	offsetsVals = tex1Dfetch(texOffsetsBoth, qry_4);
	sizesVals = tex1Dfetch(texSizesBoth, qry_4);
	unsigned offsetFwQ4 = offsetsVals.x;
	unsigned offsetRvQ4 = offsetsVals.y;
	unsigned endFwQ4 = offsetsVals.x + sizesVals.x;
	unsigned endRvQ4 = offsetsVals.y + sizesVals.y;

	int val0=0, val1=0, val2=0, val3=0, val4=0;

	int minval = 0;
	while (minval < 0x7FFFFFFF) {
		minval = 0x7FFFFFFF;
		unsigned curNumMatches = 0;

		if (offsetFwQ0 < endFwQ0) {
			val0 = (val0) ? val0 : tex1Dfetch(texSubjectHashed, offsetFwQ0);
			minval = min(val0, minval);
		}

		if (offsetFwQ1 < endFwQ1) {
			val1 = (val1) ? val1 : tex1Dfetch(texSubjectHashed, offsetFwQ1) -6;
			minval = min(val1, minval);
		}

		if (offsetFwQ2 < endFwQ2) {
			val2 = (val2) ? val2 : tex1Dfetch(texSubjectHashed, offsetFwQ2) -12;
			minval = min(val2, minval);
		}

		if (offsetFwQ3 < endFwQ3) {
			val3 = (val3) ? val3 : tex1Dfetch(texSubjectHashed, offsetFwQ3) -18;
			minval = min(val3, minval);
		}
	
		if (offsetFwQ4 < endFwQ4) {
			val4 = (val4) ? val4 : tex1Dfetch(texSubjectHashed, offsetFwQ4) -24;
			minval = min(val4, minval);
		}

		if ( (offsetFwQ0 < endFwQ0) && !(val0>minval) ) {
			++curNumMatches;
			++offsetFwQ0;
			val0 = 0;
		}
		if ( (offsetFwQ1 < endFwQ1) && !(val1>minval) ) {
			++curNumMatches;
			++offsetFwQ1;
			val1 = 0;
		}
		if ( (offsetFwQ2 < endFwQ2) && !(val2>minval) ) {
			++curNumMatches;
			++offsetFwQ2;
			val2 = 0;
		}
		if ( (offsetFwQ3 < endFwQ3) && !(val3>minval) ) {
			++curNumMatches;
			++offsetFwQ3;
			val3 = 0;
		}
		if ( (offsetFwQ4 < endFwQ4) && !(val4>minval) ) {
			++curNumMatches;
			++offsetFwQ4;
			val4 = 0;
		}

		if (curNumMatches > GUARANTEED_RUMOR_LEVEL_HASHED) {
			g_res_position[(qres_offset + totResFw)] = minval; g_res_position[(qres_offset + totResFw + 1)] = curNumMatches;
			totResFw = (totResFw < limitRes) ? (totResFw + 2) : 0;
		}
	}

	minval = 0;
	while (minval < 0x7FFFFFFF) {
		minval = 0x7FFFFFFF;
		unsigned curNumMatches = 0;

		if (offsetRvQ0 < endRvQ0) {
			val0 = (val0) ? val0 : tex1Dfetch(texRevSubjectHashed, offsetRvQ0);
			minval = min(val0, minval);
		}

		if (offsetRvQ1 < endRvQ1) {
			val1 = (val1) ? val1 : tex1Dfetch(texRevSubjectHashed, offsetRvQ1) -6;
			minval = min(val1, minval);
		}

		if (offsetRvQ2 < endRvQ2) {
			val2 = (val2) ? val2 : tex1Dfetch(texRevSubjectHashed, offsetRvQ2) -12;
			minval = min(val2, minval);
		}

		if (offsetRvQ3 < endRvQ3) {
			val3 = (val3) ? val3 : tex1Dfetch(texRevSubjectHashed, offsetRvQ3) -18;
			minval = min(val3, minval);
		}
	
		if (offsetRvQ4 < endRvQ4) {
			val4 = (val4) ? val4 : tex1Dfetch(texRevSubjectHashed, offsetRvQ4) -24;
			minval = min(val4, minval);
		}

		if ( (offsetRvQ0 < endRvQ0) && !(val0>minval) ) {
			++curNumMatches;
			++offsetRvQ0;
			val0 = 0;
		}
		if ( (offsetRvQ1 < endRvQ1) && !(val1>minval) ) {
			++curNumMatches;
			++offsetRvQ1;
			val1 = 0;
		}
		if ( (offsetRvQ2 < endRvQ2) && !(val2>minval) ) {
			++curNumMatches;
			++offsetRvQ2;
			val2 = 0;
		}
		if ( (offsetRvQ3 < endRvQ3) && !(val3>minval) ) {
			++curNumMatches;
			++offsetRvQ3;
			val3 = 0;
		}
		if ( (offsetRvQ4 < endRvQ4) && !(val4>minval) ) {
			++curNumMatches;
			++offsetRvQ4;
			val4 = 0;
		}

		if (curNumMatches > GUARANTEED_RUMOR_LEVEL_HASHED) {
			g_res_position[(qres_offset + MAX_RESULTS_PER_QUERY_HASHED + totResRev )] = minval; 
			g_res_position[(qres_offset + MAX_RESULTS_PER_QUERY_HASHED + totResRev + 1)] = curNumMatches;
			totResRev = (totResRev < limitRes) ? (totResRev + 2) : 0;
		}
	}


	g_res_numMatches[(query_idx*2)] = totResFw;
	g_res_numMatches[(query_idx*2+1)] = totResRev;
}


#endif

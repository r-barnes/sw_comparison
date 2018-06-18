/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CFASTASW_H
#define _CFASTASW_H

#include "GenericFunction_cu.h"

typedef struct tagDatabaseHash
{
	int cx;
	int cy;
	int length;
	int alignedLen;
}DatabaseHash;

typedef struct tagSeqEntry
{
	int idx;
	int value;
}SeqEntry;
//the maximum query sequence length
#define MAX_QUERY_LEN		60416			//59KB by default

#define VECTOR_LANE		16
//the number of vector lanes is 32
#if	VECTOR_LANE == 32

#define	THREADS_PER_BLOCK				192
#define THREADS_PER_WARP				32
#define THREADS_PER_WARP_MASK			31
#define THREADS_PER_WARP_SHIFT			5
#define THREADS_PER_HALF_WARP			16
#define WARPS_PER_BLOCK					(THREADS_PER_BLOCK / THREADS_PER_WARP)
#define QUERY_SEGMENT_LENGTH			512/*(THREADS_PER_WARP * 16)*/
#define QUERY_SEGMENT_LENGTH_QUARTER	(QUERY_SEGMENT_LENGTH >> 2)
#define SUBJECT_SEQUENCE_LENGTH			648

#else	//the number of vector lanes is 16

#define	THREADS_PER_BLOCK				192
#define THREADS_PER_WARP				16
#define THREADS_PER_WARP_MASK			15
#define THREADS_PER_WARP_SHIFT			4
#define THREADS_PER_HALF_WARP			8
#define WARPS_PER_BLOCK					(THREADS_PER_BLOCK / THREADS_PER_WARP)
#define QUERY_SEGMENT_LENGTH			256/*(THREADS_PER_WARP * 16)*/
#define QUERY_SEGMENT_LENGTH_QUARTER	(QUERY_SEGMENT_LENGTH >> 2)
#define SUBJECT_SEQUENCE_LENGTH			324

#endif	//#if VECTOR_LANE == 32

//this threshold is important
#define SEQ_LENGTH_THRESHOLD        	3072		//3KB

#define SEQ_LENGTH_ALIGNED				8

#define pChannelFormatKindSignedInt       	 	0	
#define pChannelFormatKindUnsignedChar      	1
#define pChannelFormatKindUnsignedChar4      	2
#define pChannelFormatKindChar4      					3
#define pChannelFormatKindSignedInt4      		4
#define pChannelFormatKindUnsigned						5

class  CFastaSW
{
public:
	CFastaSW();
	virtual ~CFastaSW();

	virtual void 	swMemcpyParameters(int matrix[32][32], int gapOpen, int gapExtend);
 	virtual void 	swMemcpyQuery(unsigned char* query, int qlen, int qAlignedLen, int offset, int matrix[32][32]);
 	virtual void	swMemcpyQueryLength(int qlen, int qAlignedLen);
 	virtual void	swCreateChannelFormatDesc();
 	virtual	void*	swMallocArray(int width, int height, int type);
 	virtual void	swBindTextureToArray();
 	virtual void	swBindQueryProfile();
 	virtual void	swUnbindTexture();
 	virtual void	swUnbindQueryProfile();

	//inter-task parallelization
 	virtual void 	swInterMallocThreadSlots(int threads, int multiProcessors, int slotSize);
 	virtual void 	swInterFreeThreadSlots();
 	virtual void 	InterRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstBlk);

	//intra-task parallelization
 	virtual void 	swIntraMallocThreadSlots(int multiProcessors, int slotSize);
 	virtual void 	swIntraFreeThreadSlots();
 	virtual void 	IntraRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstSeq);

	//transfer back results
 	virtual void 	transferResult(int numSeqs );

public:
	//intra-task subject sequences
	void* cudaInterSeqs;
	//inter-task subject sequences
	void * cudaIntraSeqs;
	//sequence hash table
	DatabaseHash* cudaSeqHash;
	//result buffers
	SeqEntry* cudaResult;
	SeqEntry* hostResult;
protected:
	struct cudaChannelFormatDesc uchar_channelDesc;
	struct cudaChannelFormatDesc uchar4_channelDesc;
	struct cudaChannelFormatDesc uint_channelDesc;
	struct cudaChannelFormatDesc char4_channelDesc;
	struct cudaChannelFormatDesc sint_channelDesc;
	struct cudaChannelFormatDesc sint4_channelDesc;

	//global memory variables for intra-task parallelization
	void* cudaDA, *cudaDB, *cudaDC;
	void* cudaHH, *cudaVV_O, *cudaVV_C;
	
	size_t cudaDAPitch, cudaDBPitch, cudaDCPitch;
	size_t cudaHHPitch, cudaVV_OPitch, cudaVV_CPitch;

};
#endif

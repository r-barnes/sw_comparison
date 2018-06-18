/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CFASTASW_VEC_H
#define _CFASTASW_VEC_H

#include "CFastaSW.h"

class  CFastaSWVec : public CFastaSW
{
public:
	CFastaSWVec();
	~CFastaSWVec();

	void 	swMemcpyParameters(int matrix[32][32], int gapOpen, int gapExtend);
 	void 	swMemcpyQuery(unsigned char* query, int qlen, int qAlignedLen, int offset, int matrix[32][32]);
 	void	swMemcpyQueryLength(int qlen, int qAlignedLen);
 	void	swBindTextureToArray();
 	void	swBindQueryProfile();
 	void	swUnbindTexture();
 	void	swUnbindQueryProfile();

	//inter-task parallelization
 	void 	swInterMallocThreadSlots(int threads, int multiProcessors, int slotSize);
 	void 	swInterFreeThreadSlots();
 	void 	InterRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstBlk);

	//intra-task parallelization
 	void 	IntraRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstSeq);

private:
	//for query profile construction
	void* cudaPackedQueryPrf;
	int	queryPrfLength;
	
	//for the H and F values of the previous partition
	ushort2* cudaHF;
	size_t cudaHFPitch;

};
#endif

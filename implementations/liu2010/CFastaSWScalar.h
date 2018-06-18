/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CFASTASW_SCALAR_H
#define _CFASTASW_SCALAR_H

#include "CFastaSW.h"
#include "GenericFunction_cu.h"

class  CFastaSWScalar : public CFastaSW
{
public:
	CFastaSWScalar();
	~CFastaSWScalar();

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
	//query profile
	void* cudaInterQueryPrf;

	//for inter-task parallelization
	void* cudaGlobal;
	size_t cudaGlobalPitch;
};
#endif

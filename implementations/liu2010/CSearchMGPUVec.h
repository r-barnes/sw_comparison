/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CSEARCH_MGPU_VEC_H_
#define _CSEARCH_MGPU_VEC_H_
#include "CSearchVec.h"

class CSearchMGPUVec: public CSearchVec
{
public:
	CSearchMGPUVec(CParams* params);
	~CSearchMGPUVec();

	int dbsearch(char* query);

private:
	static void* swthreads_func(void *plan);
	SeqEntry* globalHostResult;
};
#endif



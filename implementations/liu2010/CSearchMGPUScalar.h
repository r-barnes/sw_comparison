/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CSEARCH_MGPU_SCALAR_H_
#define _CSEARCH_MGPU_SCALAR_H_
#include "CSearchScalar.h"

class CSearchMGPUScalar : public CSearchScalar
{
public:
	CSearchMGPUScalar(CParams* params);
	~CSearchMGPUScalar();

	int dbsearch(char* query);
private:
	static void* swthreads_func(void* arg);
	SeqEntry* globalHostResult;
};
#endif



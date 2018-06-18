/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _C_SEARCH_VEC_H
#define _C_SEARCH_VEC_H
#include "CFastaSWVec.h"
#include "CParams.h"
#include "CSearch.h"

class CSearchVec : public CSearch 
{
public:
	CSearchVec(CParams* params);
	virtual ~CSearchVec();

protected:
	virtual int loaddb (char* dbFile);
	virtual int dbsearch(char* query);
};
#endif


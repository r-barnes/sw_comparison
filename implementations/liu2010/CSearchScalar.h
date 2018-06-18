/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _C_SEARCH_SCALAR_H
#define _C_SEARCH_SCALAR_H

#include "CFastaSWScalar.h"
#include "CParams.h"
#include "CSearch.h"

class CSearchScalar : public CSearch
{
public:
	CSearchScalar(CParams* params);
	virtual ~CSearchScalar();

protected:
	virtual int loaddb (char* dbFile);
	virtual int dbsearch(char* query);
};
#endif


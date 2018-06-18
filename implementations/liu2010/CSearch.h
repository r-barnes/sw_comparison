/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _C_SEARCH_H
#define _C_SEARCH_H
#include "CFastaSW.h"
#include "CParams.h"

class CSearch
{
public:
	CSearch(CParams* params);
	virtual ~CSearch();

	void	run ();
protected:
	static int compar_ascent(const void * va, const void * vb);
	static int compar_descent(const void* va, const void* vb);
	virtual void printResults(SeqEntry* hostResult, char** dbSeqsName, int numSeqs, int top, int scoreThreshold);
	
	virtual int loaddb (char* dbFile);
	virtual int dbsearch(char* query);

	//memory variables
	int dbSeqsSize;
	unsigned char** dbSeqs;
	char** dbSeqsName;
	int* dbSeqsLen;
	int* dbSeqsAlignedLen; 
	int numSeqs;
	int numThreshold;
	int maxSeqLength;
	unsigned int totalAminoAcids;
	SeqEntry* sortedSeqs;
	SeqEntry* hostResult;

	static int matrix[32][32];
	//gap opening penalty
	static int gapOpen;
	//gap extension penalty
	static int gapExtend;

	//parameters
	CParams* params;
};
//for multi-GPU support
typedef struct tagTaskPlan
{
	CFastaSW* cudasw;
	//device
	int device;
	int threads;
	GPUInfo* info;
	//query sequence
	int qLen;
	int qAlignedLen;
	unsigned char* query;
	//on the host memory
	int cx, cy;
	int maxSeqLength;
	int index;
	int	cudaInterTexWidth;
	int cudaInterTexHeight;
	int cudaIntraTexWidth;
	int cudaIntraTexHeight;
	int hostResultPos;
	SeqEntry* hostResult;
	SeqEntry* cudaHostResult;
	SeqEntry* globalHostResult;
	void* interHostSeqArray;
	unsigned char* intraHostSeqArray;
	DatabaseHash * hostSeqHash;

	//on the device memory
	int 	numSeqs;
	int 	interSeqNo;
	void* 	cudaInterSeqs;
	int		intraSeqNo;
	void* 	cudaIntraSeqs;

}TaskPlan;

#endif


/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "CFastaFile.h"
#include "CFastaSW.h"
#include "CSearch.h"
#include <math.h>

int CSearch::gapOpen = DEFAULT_GAPO;
int CSearch::gapExtend = DEFAULT_GAPE;
int CSearch::matrix[32][32] = {{0, 0}};
CSearch::CSearch(CParams* params)
{
    numSeqs = 0;
    numThreshold = 0;
    maxSeqLength = 0;
    totalAminoAcids = 0;
    dbSeqsSize = 0;
    dbSeqs = 0;
    dbSeqsLen = 0;
    dbSeqsAlignedLen = 0;
    dbSeqsName = 0;
	sortedSeqs = 0;

	this->params = params;
}

CSearch::~CSearch()
{
	int i;
	if(sortedSeqs){
		pFreeHost(sortedSeqs);
	}
	//free database
	if(dbSeqs){
		for(i = 0; i < numSeqs; i++){
			free(dbSeqs[i]);
			free(dbSeqsName[i]);
		}
		free(dbSeqs);
		free(dbSeqsName);
		free(dbSeqsLen);
		free(dbSeqsAlignedLen);
	}
}
void CSearch::run ()
{
    //read in the input sequence
    char* query = params->getQueryFile();
	char* mat = params->getSubMatrixName();
    char* db = params->getDbFile();

	//get the gap penalties
	gapOpen = params->getGapOpen();
	gapExtend = params->getGapExtend();

	printf("/**********************************/\n");
	printf("\tModel:\t\t\t%s\n", params->isUseSIMTModel() == true ? "SIMT scalar" : "SIMD vectorized");
	printf("\tScoring matrix:\t\t\t%s\n", mat);
	printf("\tGap Open penalty:\t\t%d\n", gapOpen);
	printf("\tGap Extension penalty:\t\t%d\n", gapExtend);
	printf("/**********************************/\n");

	//loading substitution matrix
	params->getMatrix (matrix);

	//load database
	loaddb(db);

	//generate the object
    dbsearch(query);    

	printf("Finished!\n");
}

//for quick sort
int	CSearch::compar_ascent(const void * va, const void * vb)
{
	const SeqEntry* a = (const SeqEntry*)va;
	const SeqEntry* b = (const SeqEntry*)vb;

	if(a->value > b->value)	return 1;
	if(a->value < b->value) return -1;

	return 0;
}
int CSearch::compar_descent(const void* va, const void* vb)
{
    const SeqEntry* a = (const SeqEntry*)va;
    const SeqEntry* b = (const SeqEntry*)vb;

    if(a->value > b->value) return -1;
    if(a->value < b->value) return 1;

    return 0;
}
void CSearch::printResults(SeqEntry* hostResult, char** dbSeqsName, int numSeqs, int top, int scoreThreshold)
{
	int i;

 	//sorting the scores
  	qsort(hostResult, numSeqs, sizeof(SeqEntry), compar_descent);

	//display the results. Here, nothing to do!
	for(i = 0; i < top ; i++){
		if(hostResult[i].value < scoreThreshold){
			printf("the score reaches the threshold (%d)\n", scoreThreshold);
			break;
		}
		if( i && i% 128 == 0){
			printf("press 'y' to quit and another key to continue\n");
			int c = getchar();
			if(c == 'y' || c == 'Y') break;
				
		}
		printf("score: %d -- %s\n",hostResult[i].value, dbSeqsName[hostResult[i].idx]);
	}
}
int CSearch::loaddb (char* dbFile)
{
	return 1;
}

int CSearch::dbsearch (char*queryFile)
{
	//do nothing
	return 0;
}

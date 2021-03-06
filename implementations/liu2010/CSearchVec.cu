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
#include "CSearchVec.h"
#include <math.h>

CSearchVec::CSearchVec(CParams* params) : CSearch(params)
{
}
CSearchVec::~CSearchVec()
{
}
int CSearchVec::loaddb (char* dbFile)
{
	int i;
	CFastaFile *dbLib;
	int seqLen;
	int seqAlignedLen;
	unsigned char* seq;

	printf("Loading database sequences from file into host memory...\n");

#define INIT_SIZE 		819200
	numSeqs = 0;
	numThreshold = 0;
	maxSeqLength = 0;
	totalAminoAcids = 0;
	dbSeqsSize = INIT_SIZE;
	dbSeqs = (unsigned char**) malloc (sizeof(unsigned char*) * dbSeqsSize);
	dbSeqsLen = (int*) malloc( sizeof(int) * dbSeqsSize);
	dbSeqsAlignedLen = (int*)malloc(sizeof(int) * dbSeqsSize);
	dbSeqsName = (char**) malloc( sizeof(char*) * dbSeqsSize);

	//open the database
	dbLib = new CFastaFile;
    dbLib->open(dbFile);
	//save all the sequences
	seq = dbLib->nextSeq (&seqLen, &seqAlignedLen, SEQ_LENGTH_ALIGNED);
    while (seqLen > 0) {

		if(numSeqs >= dbSeqsSize){
			dbSeqsSize *= 2;
			dbSeqs = (unsigned char**)realloc(dbSeqs, sizeof(unsigned char*) * dbSeqsSize);
			dbSeqsName = (char**)realloc(dbSeqsName, sizeof(char*) * dbSeqsSize);
			dbSeqsLen = (int*)realloc(dbSeqsLen, sizeof(int) * dbSeqsSize);
			dbSeqsAlignedLen = (int*)realloc(dbSeqsAlignedLen, sizeof(int) * dbSeqsSize);
			if(dbSeqs == NULL ||dbSeqsLen == NULL || dbSeqsName == NULL || dbSeqsAlignedLen == NULL){
				printf("no memory space for database sequences\n");
				return 0;
			}
		}

		dbSeqs[numSeqs] = (unsigned char*) malloc (sizeof(unsigned char) * (seqAlignedLen + 1));
		if(dbSeqs[numSeqs] == NULL){
			printf("no memory space available for the database\n");
			return 1;
		}
		//save sequence name;
		dbSeqsName[numSeqs] = strdup((char*)dbLib->getSeqName());
		//save sequence length
		dbSeqsLen[numSeqs] = seqLen;
		dbSeqsAlignedLen[numSeqs] = seqAlignedLen;
		//save sequence symbols
		memcpy(dbSeqs[numSeqs], seq, sizeof(unsigned char) * seqAlignedLen);

		//printf("No.:%d  %d %d\n", numSeqs, seqLen, seqAlignedLen);
		if(seqLen <= SEQ_LENGTH_THRESHOLD){
			numThreshold ++;
		}

		numSeqs ++;
		totalAminoAcids += seqLen;

		if(maxSeqLength < seqLen){
			maxSeqLength = seqLen;
		}

		seq = dbLib->nextSeq(&seqLen, &seqAlignedLen, SEQ_LENGTH_ALIGNED);
	}
	//start out to sort the database
	sortedSeqs = (SeqEntry*)pMallocHost(sizeof(SeqEntry) * numSeqs);
	//initialize the sorted sequence vector
	for(i = 0; i< numSeqs; i++){
		sortedSeqs[i].idx = i;
		sortedSeqs[i].value = dbSeqsLen[i];
	}
	//using quick sort to sort the vector
	qsort(sortedSeqs, numSeqs, sizeof(SeqEntry), compar_ascent);

	//releaset database structure
	dbLib->close();
	delete dbLib;

	return 1;
}

int CSearchVec::dbsearch (char*queryFile)
{
    int i,j;
    unsigned char* seq;
    CFastaFile* queryLib = new CFastaFile;

	CFastaSW* cudasw = new CFastaSWVec;

	//build coalesced sequence array for the sorted database sequence indexed less than numThreshold
	int n;
	int threads;
	int cx, cy;
	int width, height;
	int maxProcessors;
	DatabaseHash * hash;

	//copy the subsitution matrix, gap opening penalty and gap extending penalty from host to GPU
	cudasw->swMemcpyParameters(matrix, gapOpen,gapExtend);
	//create the channel format descriptor
	cudasw->swCreateChannelFormatDesc();

	//compute the width of the array
	threads = THREADS_PER_BLOCK;
	maxProcessors = pGetMultiProcessorCount();
	hash = (DatabaseHash*)pMallocHost(numSeqs * sizeof(DatabaseHash));

#define MINIMAL_WIDTH			(THREADS_PER_BLOCK * 30)
	/*****************************************************************
	********************************Stage 1***************************
 	*****************************************************************/
	//compute the width and height
	width = max(threads * maxProcessors, MINIMAL_WIDTH);
	bool done;
	do{
		done = true;
		cx = cy = 0;
		height = 0;
		for(i = 0; i < numThreshold; i++){
			int idx = sortedSeqs[i].idx;
			n = dbSeqsLen[idx];
			if(n + cx > width){
				cx = 0;
				cy ++;
			}
			cx += n;
		}
		height = cy + 1;
		if(height > 32768){
            if(width == 65536){
                fprintf(stderr, "No availabe device memory space for the database (width %d height: %d)\n", width, height);
                exit(0);
            }
			width = 65536;
			done = false;
		}
	}while(!done);

	printf("Loading Stage 1 ---- width:%d height:%d size:%d (MB)\n",width, height, width * height/1024/1024);

	//allocate space for inter-task sequence
	unsigned char* interArray = (unsigned char*)pMallocHost(width * height * sizeof(unsigned char));
	cx = cy = 0;
	for(i = 0; i < numThreshold; i++){
		//get the sequence and its length
		int idx = sortedSeqs[i].idx;
		seq = dbSeqs[idx];
		n = dbSeqsLen[idx];

		if(n + cx > width){
			//adjust the coordinates
			cx = 0;
			cy ++;
		}
		//copy the sequence
		unsigned char* ptr = interArray + cy * width + cx;
		memcpy(ptr, seq, n * sizeof(unsigned char));
		//record the position informantion of the sequence
		hash[i].cx = cx;
		hash[i].cy = cy;
		hash[i].length = n;
		hash[i].alignedLen = n;

		//adjust the coordinates
		cx += n;
	}
	//copy the sequences into cudaArray
	cudasw->cudaInterSeqs = cudasw->swMallocArray(width, height, pChannelFormatKindUnsignedChar);
	pMemcpyToArray(cudasw->cudaInterSeqs, 0, 0, interArray, width * height * sizeof(unsigned char), pMemcpyHostToDevice);
	pFreeHost(interArray);

    /*****************************************************************
 	*********************************Stage 2***************************
 	******************************************************************/

	unsigned char* intraArray;

	if(numThreshold == numSeqs){
		/*if there is not sequence of lengths > SEQ_LENGTH_THRESHOLD, goto stage3;
 		to avoid warning, allocate pseudo space for cudaIntraSeqs;
		*/
		cudasw->cudaIntraSeqs = cudasw->swMallocArray(1, 1, pChannelFormatKindUnsignedChar);
		//jump to stage3
		goto stage3;
	}

	cx = cy = 0;
	width = maxSeqLength + 1;
	height = numSeqs - numThreshold;	//set maximum height

	//for the intra-task parallelization, all the sequences start from 1 instead of 0
	intraArray = (unsigned char*) pMallocHost( width * height * sizeof(unsigned char));
	for(i = numThreshold; i < numSeqs; i++){
		//get the sequence and its length
		int idx = sortedSeqs[i].idx;
		seq = dbSeqs[idx];
		n = dbSeqsLen[idx];

		if(n + 1 + cx > width){
			//adjust the coordinates
			cx = 0;
			cy ++;
		}
		//copy the sequence
		unsigned char* ptr = intraArray + cy * width + cx;
		*ptr = DUMMY_AMINO_ACID;
		for(j = 0; j < n; j++){
			ptr[j + 1] = seq[j];
		}
		//record the position informantion of the sequence
		hash[i].cx = cx;
		hash[i].cy = cy;
		hash[i].length = n;

		//adjust the coordinates
		cx += n + 1;
	}

	//set the real height
	height = cy + 1;
	printf("Loading Stage 2 ---- width:%d height:%d size:%d (MB)\n",width, height, width * height/1024/1024);
	//copy the sequences into cudaArray
	cudasw->cudaIntraSeqs = cudasw->swMallocArray(width, height, pChannelFormatKindUnsignedChar);
	pMemcpyToArray(cudasw->cudaIntraSeqs, 0, 0, intraArray, width * height * sizeof(unsigned char), pMemcpyHostToDevice);
	pFreeHost(intraArray);
   	/*****************************************************************
 	*********************************Stage 3***************************
 	******************************************************************/
stage3:
	//bind the CUDA Array to texture
	cudasw->swBindTextureToArray();
	//copy the hash table from host to GPU
	cudasw->cudaSeqHash = (DatabaseHash*) pMallocPitch(sizeof(hash[0]), numSeqs, 1, 0);
	pMemcpy(cudasw->cudaSeqHash, hash, numSeqs * sizeof(hash[0]), pMemcpyHostToDevice);
	pFreeHost(hash);

	//malloc result for host and device
	cudasw->hostResult = (SeqEntry*)pMallocHost(sizeof(SeqEntry) * numSeqs);
	for(i = 0; i < numSeqs; i++){
		cudasw->hostResult[i].idx = sortedSeqs[i].idx;
		cudasw->hostResult[i].value = 65536;
	}
	cudasw->cudaResult = (SeqEntry*)pMallocPitch(sizeof(SeqEntry), numSeqs, 1, 0);
	//initialize the result vector on GPU
	pMemcpy(cudasw->cudaResult, cudasw->hostResult, numSeqs * sizeof(SeqEntry), pMemcpyHostToDevice);

	printf("Loading database successfully\n");

	printf("numSeqs: %d numThreshold: %d\n", numSeqs, numThreshold);
	printf("maxSeqLength: %d totalAminoAcids: %d\n", maxSeqLength, totalAminoAcids);

	printf("******************************\n");
	printf("******************************\n");
	//load queries
	int qlen, qAlignedLen;
	unsigned char* query;

	//printf("Loading the query sequences...\n");
	//open the query file
    queryLib->open(queryFile);
	//only load the first query sequence
	query = queryLib->nextSeq(&qlen, &qAlignedLen, SEQ_LENGTH_ALIGNED);
	if(qlen == 0){
		printf("query file is empty!");

		goto out;
	}

	while(qlen > 0){
		double start, end;
		//get the system time
		CParams::getSysTime(&start);

		//start computing the scores
		int blocks;
		int blk;
		blk = 0;

		//copy the query sequence from host to GPU, indexing from 1
		cudasw->swMemcpyQuery(query, qlen, qAlignedLen, sizeof(unsigned char), matrix);

		//compute the total number of thread blocks
		threads = THREADS_PER_BLOCK;
		int warpNum = threads / THREADS_PER_WARP;
		blocks = (numThreshold + warpNum - 1) / warpNum;

		//allocate memory slots for intermediate results
		int memSlotSize;
		memSlotSize = (SEQ_LENGTH_THRESHOLD + SEQ_LENGTH_ALIGNED - 1) / SEQ_LENGTH_ALIGNED;
		memSlotSize = memSlotSize * SEQ_LENGTH_ALIGNED + 1;
		//allocate memory slot
		int procsPerPass = maxProcessors * 128;
		cudasw->swInterMallocThreadSlots(warpNum, procsPerPass, memSlotSize);
		//binding the query profile
		cudasw->swBindQueryProfile();
		while(blocks > 0){
			if(blocks > procsPerPass){
				n = procsPerPass;
			}else{
				n = blocks;
			}
			cudasw->InterRunGlobalDatabaseScanning (n, threads, numThreshold, blk);
			blk += n;
			blocks -=n;
		}
		//releaset the memory slots for intermediate results
		cudasw->swInterFreeThreadSlots();
		//unbind the query profile
		cudasw->swUnbindQueryProfile();

		//decide whether to perform the intra-task parallelization
		if(numThreshold < numSeqs){
			threads = 256;
			blk = numThreshold;	//the index of the first sequence in the remaining sequences
			blocks = numSeqs - numThreshold;	//the total number of the remaining sequences
			//please decrease the maxSeqsOnePass value when there is no enough global memory on the device
			int maxSeqsOnePass = 256;
			cudasw->swIntraMallocThreadSlots(maxSeqsOnePass, maxSeqLength + 2);
    		while(blocks>0){

				if(blocks > maxSeqsOnePass){
            		n =	maxSeqsOnePass;
        		}else{
            		n = blocks;
       	 		}
        		cudasw->IntraRunGlobalDatabaseScanning (n, threads, numSeqs, blk);

        		blk += n;
        		blocks -=n;
    		}
			cudasw->swIntraFreeThreadSlots();
		}
		//transfer result from GPU to host
		cudasw->transferResult(numSeqs);
		//get the system time
		CParams::getSysTime(&end);
		double dif = end - start;
		double gcups = ((double)totalAminoAcids)/1000000.0;
		gcups /= 1000.0;
		gcups *= qlen;
		gcups /= dif;

		#ifndef BENCHMARKING
			printf("query:%s \n", queryLib->getSeqName());
			printf("Length: %d --- time: %g (s) GCUPS: %g\n", qlen, dif, gcups);
		#endif

		//display results
		int top = numSeqs > params->getTopScoresNum() ? params->getTopScoresNum(): numSeqs;
		int scoreThreshold = params->getScoreThreshold();
		printResults(cudasw->hostResult, dbSeqsName, numSeqs, top, scoreThreshold);
		//load the next query sequence
		query = queryLib->nextSeq(&qlen, &qAlignedLen, SEQ_LENGTH_ALIGNED);
    	if(qlen == 0){
        	printf("Reaching the end of the query file!\n");
    	}
	}
out:
	//free array
	pFree(cudasw->cudaSeqHash);
	cudasw->swUnbindTexture();
	pFreeArray(cudasw->cudaInterSeqs);
	pFreeArray(cudasw->cudaIntraSeqs);
	pFree(cudasw->cudaResult);
	pFreeHost(cudasw->hostResult);

	delete queryLib;
    delete cudasw;

    return 0;
}

/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#include "CSearchMGPUScalar.h"
#include "CFastaFile.h"
#include "CFastaSWScalar.h"
#include <pthread.h>
 
CSearchMGPUScalar::CSearchMGPUScalar(CParams* params) : CSearchScalar(params)
{
	globalHostResult = 0;
}
CSearchMGPUScalar::~CSearchMGPUScalar()
{
	if(globalHostResult){
		pFreeHost(globalHostResult);
	}
}
int CSearchMGPUScalar::dbsearch(char* queryFile)
{
	int i,j;
    CFastaFile* queryLib = new CFastaFile;
	
	//build coalesced sequence array for the sorted database sequence indexed less than numThreshold
	int n;
	int threads;
	int slen;
	unsigned char* seq;
	int width, height;

	//compute the width of the array
	//must make sure threads = 256; if 256 threads in a thread block is not allowed due to
	//the number of available registers, please modify the loading substitution matrix CUDA code 
	threads = 256;
	//malloc global host result buffer
	globalHostResult = (SeqEntry*)pMallocHost(sizeof(SeqEntry) * numSeqs);
#define MINIMAL_WIDTH	7680
	//initialize the TaskPlan objects
	GPUInfo* info = pGetGPUInfo();
	TaskPlan * plans = (TaskPlan*) malloc (info->n_device * sizeof(TaskPlan));
	for(i = 0; i < info->n_device; i++){
		plans[i].threads = threads;
		plans[i].device = i;
		plans[i].info = info;
		plans[i].cudaInterTexWidth =  max(pGetMultiProcessorCount(info, i) * threads, MINIMAL_WIDTH);
		plans[i].cudaInterTexHeight = 0;
		plans[i].maxSeqLength = maxSeqLength;
		//
		plans[i].hostSeqHash = (DatabaseHash*)pMallocHost(numSeqs * sizeof(DatabaseHash));
		plans[i].numSeqs = 0;
		plans[i].interSeqNo = 0;
		plans[i].intraSeqNo = 0;
		plans[i].cudasw = new CFastaSWScalar;
	}

	/*****************************************************************
	***************(Inter-Task Parallelization) Stage 1***************
 	*****************************************************************/
	
	bool done;
	do{
		n = 0;
		done = true;
		//allocate sequences for each GPU
		while(n < numThreshold){
			for(i = 0; i < info->n_device; i++){
				if(n >= numThreshold){
					break;
				}
				width = plans[i].cudaInterTexWidth;
				if(n + width < numThreshold){
					n += width;
					plans[i].interSeqNo += width;
					plans[i].cudaInterTexHeight += dbSeqsAlignedLen[sortedSeqs[n - 1].idx] / 4 + 1;
				}else{
					plans[i].interSeqNo += numThreshold - n;
					n = numThreshold;
					plans[i].cudaInterTexHeight += dbSeqsAlignedLen[sortedSeqs[n - 1].idx] / 4 + 1;
					break;
				}
			}
		}
		for(i = 0; i < info->n_device; i++){
			plans[i].cudaInterTexHeight ++;
			if(plans[i].cudaInterTexHeight > 32768){
				done = false;
				for(int j = 0; j < info->n_device; j++){
					plans[j].cudaInterTexWidth = 65536;
					plans[j].cudaInterTexHeight = 0;
					plans[j].interSeqNo = 0;
				}
				break;
			}
		}
	}while(!done);
	
	    
	//fill the array with the sorted sequences
	for(i = 0; i < info->n_device; i++){
		plans[i].cx = 0;
		plans[i].cy = 0;
		if(plans[i].cudaInterTexHeight == 0){
			plans[i].cudaInterTexHeight = 1;
		}
		plans[i].interHostSeqArray = pMallocHost(
				plans[i].cudaInterTexWidth * plans[i].cudaInterTexHeight * sizeof(unsigned int));

		//allocate result slot for host
		plans[i].hostResult = (SeqEntry*)pMallocHost(sizeof(SeqEntry) * numSeqs);
		plans[i].hostResultPos = 0;
		plans[i].globalHostResult = globalHostResult;
	}
	
	n = 0;
	while( n < numThreshold){

		for( i = 0; i < info->n_device; i++){
			width =  plans[i].cudaInterTexWidth;
			height = plans[i].cudaInterTexHeight;
			unsigned int* array = (unsigned int*)plans[i].interHostSeqArray;
			
			if( n >= numThreshold){
				break;
			}
			//get the sequence and its length after sorting
			int realWidth = min(width, numThreshold - n);
			for(int k = 0; k < realWidth; k++){
				int idx = sortedSeqs[n + k].idx;
				seq = dbSeqs[idx];
				slen = dbSeqsAlignedLen[idx];
		
				//copy the sequence into the array,starting from index 1 instead of 0
 				unsigned int * ptr = array + plans[i].cy * width + plans[i].cx;
     		for(j = 0; j < slen; j += 4){
					/*ptr->x = seq[j];
					ptr->y = seq[j + 1];
					ptr->w = seq[j + 2];
					ptr->z = seq[j + 3];*/
      		*ptr = ((unsigned int)seq[j]) | (((unsigned int )seq[j + 1]) << 8) | (((unsigned int)seq[j + 2]) << 16) | (((unsigned int)seq[j + 3]) << 24);
					ptr += width;
				}
				//build the corresponding hash item
				int index = plans[i].numSeqs;
				plans[i].hostSeqHash[index].cx = plans[i].cx;
				plans[i].hostSeqHash[index].cy = plans[i].cy;
				plans[i].hostSeqHash[index].alignedLen = slen;
				plans[i].hostSeqHash[index].length = dbSeqsLen[idx];
				
				//save the corresponding sequence index and initialize the value
				plans[i].hostResult[index].idx = idx;
				plans[i].hostResult[index].value = 65536;
				plans[i].numSeqs ++;

				//adjust the coordinates
				plans[i].cx ++;
				if(plans[i].cx >= width){
					plans[i].cx = 0;
					plans[i].cy += slen / 4;
				}
		
				if(plans[i].cy >= height){
					printf("the array overflowed at the bottom (cy:%d heigth:%d)! press any key to continue\n",
								plans[i].cy, height);
					getchar();
					break;
				}
			}
			//increase the n
			n += realWidth;
		}
	}

	for(i = 0; i < info->n_device; i++){
		width = plans[i].cudaInterTexWidth;
		height = plans[i].cudaInterTexHeight;
		if(width > 65536 || height > 32768){
			fprintf(stderr, "width(%d) or height(%d) out of texture reference range\n",
					width, height);
		}
	}

    /*****************************************************************
 	***************Intra-Task Parallelization Stage 2*****************
 	******************************************************************/
	
	if(numThreshold == numSeqs){
		//pseudo number to avoid binding errors
		for(i = 0; i < info->n_device; i++){
		 	plans[i].cx = 0;
        	plans[i].cy = 0;
       	 	plans[i].cudaIntraTexWidth = 1;  //not out of the texture reference range
        	//set maximum height
        	plans[i].cudaIntraTexHeight = 1;

        	//allocate host memory
        	plans[i].intraHostSeqArray = (unsigned char*) pMallocHost(plans[i].cudaIntraTexWidth *
                	plans[i].cudaIntraTexHeight * sizeof(unsigned char));
		}
		goto stage3;
	}
	for(i = 0; i < info->n_device; i++){
		plans[i].cx = 0;
		plans[i].cy = 0;
		plans[i].cudaIntraTexWidth = maxSeqLength + 1;	//not out of the texture reference range
		//set maximum height
		plans[i].cudaIntraTexHeight = (numSeqs - numThreshold + info->n_device - 1)/info->n_device;

		//allocate host memory
		if(plans[i].cudaIntraTexHeight <= 0){
			plans[i].cudaIntraTexHeight = 0;
		}
		plans[i].intraHostSeqArray = (unsigned char*) pMallocHost(plans[i].cudaIntraTexWidth * 
				plans[i].cudaIntraTexHeight * sizeof(unsigned char));
	}

	n = numThreshold;
	while ( n < numSeqs){
		for(i = 0; i < info->n_device; i++){
			
			width = plans[i].cudaIntraTexWidth;
			if(n >= numSeqs){
				break;
			}
			//get the sequence and its length
			int idx = sortedSeqs[n].idx;
			seq = dbSeqs[idx];
			slen = dbSeqsLen[idx];
		
			if(slen + 1 + plans[i].cx > width){
				//adjust the coordinates
				plans[i].cx = 0;
				plans[i].cy ++;
			}
			//copy the sequence
			unsigned char* ptr = plans[i].intraHostSeqArray + plans[i].cy * width + plans[i].cx;
			for(j = 0; j < slen; j++){
				ptr[j + 1] = seq[j];
			}
 			//build the corresponding hash item
         	int index = plans[i].numSeqs;
         	plans[i].hostSeqHash[index].cx = plans[i].cx;
          	plans[i].hostSeqHash[index].cy = plans[i].cy;
       		plans[i].hostSeqHash[index].length = slen;

       		//save the corresponding sequence index and initialize the value
         	plans[i].hostResult[index].idx = idx;
         	plans[i].hostResult[index].value = 65536;
           	plans[i].numSeqs ++;
			plans[i].intraSeqNo++;

			//adjust the coordinates
			plans[i].cx += slen + 1;

			//increase n
			n++;
		}
	}
	
	for(i = 0; i < info->n_device; i++){
		plans[i].cudaIntraTexHeight = plans[i].cy + 1;
	}
   	/*****************************************************************
 	*********************************Stage 3***************************
 	******************************************************************/
stage3:

	int pos = 0;
	for(i = 0; i < info->n_device; i++){
		if(plans[i].interSeqNo + plans[i].intraSeqNo != plans[i].numSeqs){
			printf("seq number error-----i:%d\n",i);
			getchar();
			return 0;
		}
		//initialize the results buffer on the GPU
		plans[i].hostResultPos = pos;
		pos += plans[i].numSeqs;
	}
	printf("Loading database successfully\n");

	printf("numSeqs: %d numThreshold: %d\n", numSeqs, numThreshold);
	printf("maxSeqLength: %d totalAminoAcids: %d\n", maxSeqLength, totalAminoAcids);

	printf("******************************\n");
	printf("******************************\n");
	//load queries
	int qlen;
	int qAlignedLen;
	unsigned char* query;
	//OS thread ID
 	pthread_t * threadID = (pthread_t*)malloc(sizeof(pthread_t) * info->n_device);

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
		//get the systeme time
		CParams::getSysTime(&start);
		
		for(i = 0; i < info->n_device; i++){
			plans[i].qLen = qlen;
			plans[i].qAlignedLen = qAlignedLen;
			plans[i].query = query;
		}
		//create threads
		for(i = 0; i < info->n_device;i++){
			pthread_create(&threadID[i], NULL, swthreads_func, (void*)&plans[i]);
		}
		/*wait for the completion of threads*/
		for(i = 0; i < info->n_device; ++i){
			pthread_join(threadID[i], NULL);
		}
		
		//get the system time
		CParams::getSysTime(&end);

		double dif = end - start;
		double gcups = ((float)totalAminoAcids)/1000000.0;
		gcups /= 1000.0;
		gcups *= qlen;
		gcups /= dif;
		
		printf("query:%s \n", queryLib->getSeqName());
		printf("Length: %d --- time: %g (s) GCUPS: %g\n", qlen, dif, gcups);
		
		//display results
		int top = numSeqs > params->getTopScoresNum() ? params->getTopScoresNum(): numSeqs;
        int scoreThreshold = params->getScoreThreshold();
		printf("----------Display the top %d ----------\n", top);
		printResults(globalHostResult, dbSeqsName, numSeqs, top, scoreThreshold);
		//load the next query sequence
		query = queryLib->nextSeq(&qlen, &qAlignedLen, SEQ_LENGTH_ALIGNED);
    	if(qlen == 0){
        	printf("Reaching the end of the query file!\n");
    	}
	}
	
out:
	for(i = 0; i < info->n_device; i++){
		TaskPlan * plan = &plans[i];
		pFreeHost(plan->interHostSeqArray);
		pFreeHost(plan->intraHostSeqArray);
		pFreeHost(plan->hostSeqHash);
		pFreeHost(plan->hostResult);
		delete plan->cudasw;
	}
	free(plans);
	free(info);
	
	if(globalHostResult){
		pFreeHost(globalHostResult);
		globalHostResult = 0;
	}

	//close the database files
	delete queryLib;

    return 0;
}
void* CSearchMGPUScalar::swthreads_func(void* arg)
{
	int n;
	int threads;
	int blocks;
	int blk;
	int width, height;
	TaskPlan* plan = (TaskPlan*)arg;
	CFastaSW* cudasw = plan->cudasw;

	//select device
	pSetDevice(plan->info, plan->info->devices[plan->device]);

 	//copy the subsitution matrix, gap penalties from host to GPU
   	cudasw->swMemcpyParameters(CSearch::matrix, CSearch::gapOpen, CSearch::gapExtend);

	//copy the inter-task database sequences from host to GPU
	width = plan->cudaInterTexWidth;
	height = plan->cudaInterTexHeight;
	
	cudasw->cudaInterSeqs = cudasw->swMallocArray(width, height, pChannelFormatKindUnsignedChar4);
	pMemcpyToArray(cudasw->cudaInterSeqs, 0, 0, plan->interHostSeqArray,
                width * height * sizeof(unsigned int), pMemcpyHostToDevice);
	
	//copy the intra-task sequences into cudaArray	
    width = plan->cudaIntraTexWidth;
 	height = plan->cudaIntraTexHeight;
	cudasw->cudaIntraSeqs = cudasw->swMallocArray(width, height, pChannelFormatKindUnsignedChar);
	pMemcpyToArray(cudasw->cudaIntraSeqs, 0, 0, plan->intraHostSeqArray, 
					width * height * sizeof(unsigned char), pMemcpyHostToDevice);

  	//bind the CUDA Array to texture
	cudasw->swBindTextureToArray();

 	//copy the hash table from host to GPU
	cudasw->cudaSeqHash = (DatabaseHash*)pMallocPitch(sizeof(DatabaseHash),
                plan->numSeqs, 1, 0);	
	pMemcpy(cudasw->cudaSeqHash, plan->hostSeqHash,
                plan->numSeqs * sizeof(DatabaseHash), pMemcpyHostToDevice);

	//initialize the results buffer on the GPU
	memcpy(plan->globalHostResult +  plan->hostResultPos, plan->hostResult, plan->numSeqs * sizeof(SeqEntry));
	cudasw->hostResult = &plan->globalHostResult[plan->hostResultPos];

	cudasw->cudaResult = (SeqEntry*)pMallocPitch(sizeof(SeqEntry), plan->numSeqs, 1, 0);
	pMemcpy(cudasw->cudaResult, cudasw->hostResult, plan->numSeqs * sizeof(SeqEntry), pMemcpyHostToDevice);

	//calcuate the total number of blocks for inter-task parallelization
	blk = 0;
	threads = plan->threads;
	blocks = (plan->interSeqNo + threads - 1) / threads;
	
	//copy the query sequence
	cudasw->swMemcpyQuery(plan->query, plan->qLen, plan->qAlignedLen, sizeof(unsigned char), CSearch::matrix);

	//allocate memory slots for intermediate results
	int memSlotSize = (SEQ_LENGTH_THRESHOLD + SEQ_LENGTH_ALIGNED - 1) / SEQ_LENGTH_ALIGNED;
	memSlotSize = memSlotSize * SEQ_LENGTH_ALIGNED + 1;

	int maxProcessors = pGetMultiProcessorCount(plan->info, plan->device);
	int procsPerPass = maxProcessors * 4;
	cudasw->swInterMallocThreadSlots(threads, procsPerPass, memSlotSize);
	cudasw->swBindQueryProfile();

	while(blocks > 0){
		if(blocks > procsPerPass){
			n = procsPerPass;
		}else{
			n = blocks;
		}
		cudasw->InterRunGlobalDatabaseScanning (n, threads, plan->interSeqNo, blk);
		blk += n;
		blocks -=n;
	}
	//releaset the memory slots for intermediate results
	cudasw->swInterFreeThreadSlots();
	cudasw->swUnbindQueryProfile();
	
	if(plan->intraSeqNo > 0){
		//the index of the first sequence in the result buffer
		blk = plan->interSeqNo;
		blocks = plan->intraSeqNo;
	
		//set query sequence length
		cudasw->swMemcpyQueryLength(plan->qLen, plan->qAlignedLen);
		//please decrease the maxSeqsOnePass value 
		//when there is no enough global memory on the device
		int maxSeqsOnePass = 256;
		cudasw->swIntraMallocThreadSlots(maxSeqsOnePass, plan->maxSeqLength + 2);
    	while(blocks>0){
        
			if(blocks > maxSeqsOnePass){
           		n =	maxSeqsOnePass;
        	}else{
           		n = blocks;
       		}	
        	cudasw->IntraRunGlobalDatabaseScanning(n, threads, plan->numSeqs, blk);

        	blk += n;
        	blocks -=n;
    	}
		cudasw->swIntraFreeThreadSlots();
	}

	//transfer result from GPU to host
	cudasw->transferResult(plan->numSeqs);
	
	//free device resources
	cudasw->swUnbindTexture();
	pFree(cudasw->cudaSeqHash);
	pFreeArray(cudasw->cudaInterSeqs);
	pFreeArray(cudasw->cudaIntraSeqs);
	pFree(cudasw->cudaResult);

	return NULL;
}


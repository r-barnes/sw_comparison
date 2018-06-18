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

#include "CFastaSWVec.h"
#define CUERR { cudaError_t err;			\
	if ((err = cudaGetLastError()) != cudaSuccess) {		\
  		printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }}

__device__ __constant__ int cudaSubMatrix[32][32];
__device__ __constant__ int cudaGapOpen;
__device__ __constant__ int cudaGapExtend;
__device__ __constant__ int cudaGapOE;		//cudaGapOpen + cudaGapExtend;
__device__ __constant__ unsigned char cudaQuery[MAX_QUERY_LEN];
__device__ __constant__ int cudaQueryLen;
__device__ __constant__ int cudaQueryAlignedLen;
__device__ __constant__ int cudaPrfLength;
__device__ __constant__ int cudaPrfLengthLimit;
__device__ __constant__ int cudaPackedPrfLengthLimit;
__device__ __constant__ int cudaNumSeqs;


texture<int, 2, cudaReadModeElementType> InterSeqs;
texture<int, 2, cudaReadModeElementType> IntraSeqs;
texture<int4, 2, cudaReadModeElementType> InterPackedQueryPrf;

CFastaSWVec::CFastaSWVec() : CFastaSW()
{
	//zero the query profile length
	this->queryPrfLength = 0;  
}
CFastaSWVec::~CFastaSWVec()
{
	//do nothing
}
//global functions
void CFastaSWVec::swMemcpyParameters(int matrix[32][32], int gapOpen, int gapExtend)
{
	int gapOE = gapOpen + gapExtend;

	cudaMemcpyToSymbol(cudaSubMatrix,matrix, 32 * 32 *sizeof(int));
	CUERR
	
	cudaMemcpyToSymbol(cudaGapOpen, &gapOpen, sizeof(int));
	CUERR

	cudaMemcpyToSymbol(cudaGapExtend, &gapExtend,sizeof(int));
	CUERR

	cudaMemcpyToSymbol(cudaGapOE, &gapOE, sizeof(int));
	CUERR
}
void CFastaSWVec::swMemcpyQuery(unsigned char* query, int qlen, int qAlignedLen, int offset, int matrix[32][32])
{
	int i, j;
	int aligned;
	int* hostQueryPrf;
	int packedPrfLength;
	char4* packedQueryPrf;
	
	//build the profile for inter-task parallelization
	aligned = THREADS_PER_WARP;
	int segLength = (qlen + aligned - 1) / aligned;
	int prfLength = segLength * aligned;	//aligned to THREADS_PER_WARP boundary

	//copy the query sequence length
    cudaMemcpyToSymbol(cudaQuery, query, qlen * sizeof(unsigned char), offset, cudaMemcpyHostToDevice);
    CUERR

    cudaMemcpyToSymbol(cudaQueryLen, &qlen, sizeof(int));
    CUERR
	
	this->queryPrfLength = prfLength;
	cudaMemcpyToSymbol(cudaPrfLength, &prfLength, sizeof(int));
	CUERR

	int prfLengthLimit = max(prfLength - QUERY_SEGMENT_LENGTH, 0);
	cudaMemcpyToSymbol(cudaPrfLengthLimit, &prfLengthLimit, sizeof(int));
	CUERR

    cudaMemcpyToSymbol(cudaQueryAlignedLen, &qAlignedLen, sizeof(int));
    CUERR

	int packedPrfLengthLimit = max(prfLength / 4  - QUERY_SEGMENT_LENGTH_QUARTER, 0);
	cudaMemcpyToSymbol(cudaPackedPrfLengthLimit, &packedPrfLengthLimit, sizeof(int));
	CUERR

	//compute the query profile
	aligned = THREADS_PER_WARP * 4;
	packedPrfLength = (qlen + aligned - 1) / aligned;
	packedPrfLength = packedPrfLength * aligned;
	packedPrfLength /= 4;
	hostQueryPrf = (int*)malloc(sizeof(int) * prfLength * 32);
	packedQueryPrf = (char4*)pMallocHost(sizeof(char4) * packedPrfLength * 32);

	if(prfLength <= QUERY_SEGMENT_LENGTH){
		//compute the normal striped query profile
		for(i = 0; i < 32; i++){
			int* p = hostQueryPrf + i * prfLength;
			for(int k = 0; k < segLength; k++){
				for(j = 0; j < THREADS_PER_WARP; j++){
					if( j * segLength + k < qlen){
						*p++ = matrix[i][query[j * segLength + k]];
					}else{
						*p++ = 0;
					}
				}
			}
		}
		//transfrom and pack the normal striped query profile
		for(i = 0; i < 32; i++){
			int k, s;
			int* prf1;
			char4* prf2;
			int segments = segLength & (~3);
			for(k = 0, s = 0; k < segments; k += 4, s++){
				prf1= hostQueryPrf + i * prfLength + k * THREADS_PER_WARP;
				prf2 = packedQueryPrf + i * packedPrfLength + s * THREADS_PER_WARP;
				for(j = 0; j < THREADS_PER_WARP; j++){
					int* p = prf1 + j;
					char4* q = prf2 + j;
				
					q->x = *p; p += THREADS_PER_WARP;	//k-th
					q->y = *p; p += THREADS_PER_WARP;	//(k+1)-th
					q->z = *p; p += THREADS_PER_WARP;	//(k+2)-th
					q->w = *p;							//(k+3)-th
				}
			}
			if((segLength & 3)> 0){
				prf1= hostQueryPrf + i * prfLength + k * THREADS_PER_WARP;
				prf2 = packedQueryPrf + i * packedPrfLength + s * THREADS_PER_WARP;
				switch(segLength & 3){
				case 1:
					for(j = 0; j < THREADS_PER_WARP; j++){
						int* p = prf1 + j;
						char4* q = prf2 + j;
						
						q->x = *p;
						q->y = 0;
						q->z = 0;
						q->w = 0;
					}
					break;	
				case 2:
					for(j = 0; j < THREADS_PER_WARP; j++){
						int* p = prf1 + j;
						char4* q = prf2 + j;
						
						q->x = *p; p += THREADS_PER_WARP;
						q->y = *p;
						q->z = 0;
						q->w = 0;
					}
					break;	
				case 3:
					for(j = 0; j < THREADS_PER_WARP; j++){
						int* p = prf1 + j;
						char4* q = prf2 + j;
						
						q->x = *p; p += THREADS_PER_WARP;
						q->y = *p; p += THREADS_PER_WARP;
						q->z = *p;
						q->w = 0;
					}
					break;	
				};
			}
		}
	}else{
		int baseOff = 0;
		int* base = hostQueryPrf;
		char4* packedBase = packedQueryPrf;
		int numIters = prfLength / QUERY_SEGMENT_LENGTH;
		//compute the normal query profile
		for(int iter = 0; iter < numIters; iter++){
			for(i = 0; i < 32; i++){
				int* p = base + i * prfLength;
				for(int k = 0; k < 16; k++){
					for(j = 0; j < THREADS_PER_WARP; j++){
						int index = j * 16 + k + baseOff;
						if(index < qlen){
							*p++ = matrix[i][query[index]];
						}else{
							*p++ = 0;
						}
					}
				}
			}
			//transform and pack the normal striped query profile
			for(i = 0; i < 32; i++){
				for(int k = 0, s = 0; k < 16; k += 4, s++){
					int* prf1= base + i * prfLength + k * THREADS_PER_WARP;
					char4* prf2 = packedBase + i * packedPrfLength + s * THREADS_PER_WARP;
					for(j = 0; j < THREADS_PER_WARP; j++){
						int* p = prf1 + j;
						char4* q = prf2 + j;
					
						q->x = *p; p += THREADS_PER_WARP;	//k-th
						q->y = *p; p += THREADS_PER_WARP;	//(k+1)-th
						q->z = *p; p += THREADS_PER_WARP;	//(k+2)-th
						q->w = *p; p += THREADS_PER_WARP;	//(k+3)-th
					}
				}
			}

			baseOff += QUERY_SEGMENT_LENGTH;
			base += QUERY_SEGMENT_LENGTH;
			packedBase += QUERY_SEGMENT_LENGTH_QUARTER;
		}
		if(baseOff < prfLength){
			//compute the normal query profile
			segLength = (prfLength - baseOff) / THREADS_PER_WARP;
			for(i = 0; i < 32; i++){
				int* p = base + i * prfLength;
				for(int k = 0; k < segLength; k++){
					for(j = 0; j < THREADS_PER_WARP; j++){
						int index = j * segLength + k + baseOff;
						if(index < qlen){
							*p++ = matrix[i][query[index]];
						}else{
							*p++ = 0;
						}
					}
				}
			}
			//transfrom and pack the normal striped query profile
			for(i = 0; i < 32; i++){
				int k, s;
				int* prf1;
				char4* prf2;
				int segments = segLength & (~3);
				for(k = 0, s = 0; k < segments; k += 4, s++){
					prf1= base + i * prfLength + k * THREADS_PER_WARP;
					prf2 = packedBase + i * packedPrfLength + s * THREADS_PER_WARP;
					for(j = 0; j < THREADS_PER_WARP; j++){
						int* p = prf1 + j;
						char4* q = prf2 + j;
					
						q->x = *p; p += THREADS_PER_WARP;	//k-th
						q->y = *p; p += THREADS_PER_WARP;	//(k+1)-th
						q->z = *p; p += THREADS_PER_WARP;	//(k+2)-th
						q->w = *p;							//(k+3)-th
					}
				}
				if((segLength & 3)> 0){
					prf1= base + i * prfLength + k * THREADS_PER_WARP;
					prf2 = packedBase + i * packedPrfLength + s * THREADS_PER_WARP;
					switch(segLength & 3){
					case 1:
						for(j = 0; j < THREADS_PER_WARP; j++){
							int* p = prf1 + j;
							char4* q = prf2 + j;
							
							q->x = *p;
							q->y = 0;
							q->z = 0;
							q->w = 0;
						}
						break;	
					case 2:
						for(j = 0; j < THREADS_PER_WARP; j++){
							int* p = prf1 + j;
							char4* q = prf2 + j;
							
							q->x = *p; p += THREADS_PER_WARP;
							q->y = *p;
							q->z = 0;
							q->w = 0;
						}
						break;	
					case 3:
						for(j = 0; j < THREADS_PER_WARP; j++){
							int* p = prf1 + j;
							char4* q = prf2 + j;
						
							q->x = *p; p += THREADS_PER_WARP;
							q->y = *p; p += THREADS_PER_WARP;
							q->z = *p;
							q->w = 0;
						}
						break;	
					};
				}
			}

		}
	}
	free(hostQueryPrf);

 	cudaPackedQueryPrf = swMallocArray(packedPrfLength, 32, pChannelFormatKindChar4);
    pMemcpy2DToArray(cudaPackedQueryPrf, 0, 0, packedQueryPrf, packedPrfLength * sizeof(char4), 
						packedPrfLength * sizeof(char4), 32, pMemcpyHostToDevice);
	pFreeHost(packedQueryPrf);
}
void CFastaSWVec::swMemcpyQueryLength( int qlen, int qAlignedLen)
{
    cudaMemcpyToSymbol(cudaQueryLen, &qlen, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUERR

    cudaMemcpyToSymbol(cudaQueryAlignedLen, &qAlignedLen, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUERR
}
void CFastaSWVec::swBindTextureToArray()
{
    cudaBindTextureToArray(InterSeqs,(cudaArray*)cudaInterSeqs, uchar_channelDesc);
    CUERR

    InterSeqs.addressMode[0] = cudaAddressModeClamp;
    InterSeqs.addressMode[1] = cudaAddressModeClamp;
    InterSeqs.filterMode = cudaFilterModePoint;
    InterSeqs.normalized = false;

    cudaBindTextureToArray(IntraSeqs,(cudaArray*)cudaIntraSeqs, uchar_channelDesc);
    CUERR

    IntraSeqs.addressMode[0] = cudaAddressModeClamp;
    IntraSeqs.addressMode[1] = cudaAddressModeClamp;
    IntraSeqs.filterMode = cudaFilterModePoint;
    IntraSeqs.normalized = false;

}
void CFastaSWVec::swBindQueryProfile()
{
	cudaBindTextureToArray(InterPackedQueryPrf,(cudaArray*)cudaPackedQueryPrf, char4_channelDesc);
	CUERR

	InterPackedQueryPrf.addressMode[0] = cudaAddressModeClamp;
    InterPackedQueryPrf.addressMode[1] = cudaAddressModeClamp;
    InterPackedQueryPrf.filterMode = cudaFilterModePoint;
    InterPackedQueryPrf.normalized = false;  

}
void CFastaSWVec::swUnbindTexture()
{
	cudaUnbindTexture(InterSeqs);
	CUERR

	cudaUnbindTexture(IntraSeqs);
	CUERR
}
void CFastaSWVec::swUnbindQueryProfile()
{
	cudaUnbindTexture(InterPackedQueryPrf);
	CUERR
	

	pFreeArray(cudaPackedQueryPrf);
	cudaPackedQueryPrf = 0;
}
void CFastaSWVec::swInterMallocThreadSlots(int threads, int multiProcessors, int slotSize)
{
	int slots, aligned;
	//calculate the number of slots to be allocated, each warp, each slot
	slots = threads * multiProcessors;

	aligned = THREADS_PER_WARP;
	slotSize = (slotSize + aligned - 1) / aligned;
	slotSize *= aligned;

	cudaHF = (ushort2*)pMallocPitch(sizeof(ushort2), slotSize, slots, &cudaHFPitch);
}
void CFastaSWVec::swInterFreeThreadSlots()
{
	pFree(cudaHF);
}

#define add_sat(a, b)  	max((a) + (b), 0)
#define sub_sat(a, b)	max((a) - (b), 0)
/*************************************************************
		Smith-Waterman for inter-task parallelization
**************************************************************/
__device__ int interSWCorePart1(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int regMaxH;
	int4 regSubScore;
	int regH, regE;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	regH = 0;
	regE = 0;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;

      	//initialize vecP
     	vecShift[tid] = regH;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH;					//save the old H value
		regH = max(regT, regE);  	//adjust vecH value with vecE and vecShift
		regH = max(regH, regF);
				
		regT = sub_sat(regH, cudaGapOE);	//calculate the new vecE
		regE = max(regE - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
			//shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH = max(regH, regF);
			//update vecE values from the new vecH
			regE = max(regE, regH - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart2(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int2 regH, regE;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int2 initZero = {0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;

      	//initialize vecP
     	vecShift[tid] = regH.y;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
			//shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}

__device__ int interSWCorePart3(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int3 regH, regE;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int3 initZero = {0, 0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;

      	//initialize vecP
     	vecShift[tid] = regH.z;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}

__device__ int interSWCorePart4(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH.w;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/\
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart5(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int	regH0, regE0;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = 0;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0;					//save the old H value
		regH0 = max(regT, regE0);  	//adjust vecH value with vecE and vecShift
		regH0 = max(regH0, regF);
				
		regT = sub_sat(regH0, cudaGapOE);	//calculate the new vecE
		regE0 = max(regE0 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0 = max(regH0, regF);
			//update vecE values from the new vecH
			regE0 = max(regE0, regH0 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart6(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int2 regH0, regE0;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int2 int2Zero = {0, 0};
	regH = regE = initZero;
	regH0 = regE0 = int2Zero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH0.y;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart7(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int3 regH0, regE0;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int3 int3Zero = {0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = int3Zero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH0.z;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart8(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH0.w;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart9(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int	regH1, regE1;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = 0;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH1;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1;					//save the old H value
		regH1 = max(regT, regE1);  	//adjust vecH value with vecE and vecShift
		regH1 = max(regH1, regF);
				
		regT = sub_sat(regH1, cudaGapOE);	//calculate the new vecE
		regE1 = max(regE1 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1 = max(regH1, regF);
			//update vecE values from the new vecH
			regE1 = max(regE1, regH1 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}

	return regMaxH;
}
__device__ int interSWCorePart10(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int2 regH1, regE1;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int2 int2Zero = {0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = int2Zero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH1.y;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart11(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int3 regH1, regE1;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int3 int3Zero = {0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = int3Zero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH1.z;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart12(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH1.w;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart13(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int	regH2, regE2;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = 0;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH2;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2;					//save the old H value
		regH2 = max(regT, regE2);  	//adjust vecH value with vecE and vecShift
		regH2 = max(regH2, regF);
				
		regT = sub_sat(regH2, cudaGapOE);	//calculate the new vecE
		regE2 = max(regE2 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2 = max(regH2, regF);
			//update vecE values from the new vecH
			regE2 = max(regE2, regH2 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart14(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int2	regH2, regE2;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = (int2){0, 0};

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH2.y;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}

__device__ int interSWCorePart15(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int3 regH2, regE2;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = (int3){0, 0, 0};

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH2.z;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 14*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.z;					//save the old H value
		regH2.z = max(regT, regE2.z);  	//adjust vecH value with vecE and vecShift
		regH2.z = max(regH2.z, regF);
				
		regT = sub_sat(regH2.z, cudaGapOE);	//calculate the new vecE
		regE2.z = max(regE2.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 14*/
			cmpRes = regF <= sub_sat(regH2.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.z = max(regH2.z, regF);
			//update vecE values from the new vecH
			regE2.z = max(regE2.z, regH2.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCorePart16(int tid, ushort2* globalHF, int queryPrfOff, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int4 regH2, regE2;
	int	regF, regP, regT, regM;
	int cmpRes;
	ushort2 regS;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = initZero;

	//starting the main loop
	regMaxH = 0;
	regP = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
    	regS = globalHF[i];
      	if(tid == 0){
       		regF = regS.y;
     	}
		regM = regS.x;
      	//initialize vecP
     	vecShift[tid] = regH2.w;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = queryPrfOff + tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 14*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.z;					//save the old H value
		regH2.z = max(regT, regE2.z);  	//adjust vecH value with vecE and vecShift
		regH2.z = max(regH2.z, regF);
				
		regT = sub_sat(regH2.z, cudaGapOE);	//calculate the new vecE
		regE2.z = max(regE2.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 15*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.w;					//save the old H value
		regH2.w = max(regT, regE2.w);  	//adjust vecH value with vecE and vecShift
		regH2.w = max(regH2.w, regF);
				
		regT = sub_sat(regH2.w, cudaGapOE);	//calculate the new vecE
		regE2.w = max(regE2.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 14*/
			cmpRes = regF <= sub_sat(regH2.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.z = max(regH2.z, regF);
			//update vecE values from the new vecH
			regE2.z = max(regE2.z, regH2.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 15*/
			cmpRes = regF <= sub_sat(regH2.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.w = max(regH2.w, regF);
			//update vecE values from the new vecH
			regE2.w = max(regE2.w, regH2.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
     	//save the H value of the previous segment
       	regP = regM;
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart1(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int regH, regE;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	regH = 0;
	regE = 0;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	
      	//initialize vecP
     	vecShift[tid] = regH;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH;					//save the old H value
		regH = max(regT, regE);  	//adjust vecH value with vecE and vecShift
		regH = max(regH, regF);
				
		regT = sub_sat(regH, cudaGapOE);	//calculate the new vecE
		regE = max(regE - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH = max(regH, regF);
			//update vecE values from the new vecH
			regE = max(regE, regH - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart2(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int2 regH, regE;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int2 initZero = {0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH.y;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}

__device__ int interSWCoreInitPart3(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int3 regH, regE;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int3 initZero = {0, 0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH.z;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		//the lazy-F loop
        for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}

__device__ int interSWCoreInitPart4(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = initZero;
	regE = initZero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH.w;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/\
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart5(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int	regH0, regE0;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = 0;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH0;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0;					//save the old H value
		regH0 = max(regT, regE0);  	//adjust vecH value with vecE and vecShift
		regH0 = max(regH0, regF);
				
		regT = sub_sat(regH0, cudaGapOE);	//calculate the new vecE
		regE0 = max(regE0 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0 = max(regH0, regF);
			//update vecE values from the new vecH
			regE0 = max(regE0, regH0 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart6(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int2 regH0, regE0;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int2 int2Zero = {0, 0};
	regH = regE = initZero;
	regH0 = regE0 = int2Zero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH0.y;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart7(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int3 regH0, regE0;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int3 int3Zero = {0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = int3Zero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH0.z;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart8(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH0.w;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
        j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart9(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int	regH1, regE1;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = 0;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH1;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1;					//save the old H value
		regH1 = max(regT, regE1);  	//adjust vecH value with vecE and vecShift
		regH1 = max(regH1, regF);
				
		regT = sub_sat(regH1, cudaGapOE);	//calculate the new vecE
		regE1 = max(regE1 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1 = max(regH1, regF);
			//update vecE values from the new vecH
			regE1 = max(regE1, regH1 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}

	return regMaxH;
}
__device__ int interSWCoreInitPart10(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int2 regH1, regE1;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int2 int2Zero = {0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = int2Zero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH1.y;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart11(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int3 regH1, regE1;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	int3 int3Zero = {0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = int3Zero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH1.z;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart12(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH1.w;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart13(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int	regH2, regE2;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = 0;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH2;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2;					//save the old H value
		regH2 = max(regT, regE2);  	//adjust vecH value with vecE and vecShift
		regH2 = max(regH2, regF);
				
		regT = sub_sat(regH2, cudaGapOE);	//calculate the new vecE
		regE2 = max(regE2 - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2 = max(regH2, regF);
			//update vecE values from the new vecH
			regE2 = max(regE2, regH2 - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart14(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int2	regH2, regE2;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = (int2){0, 0};

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH2.y;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}	
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}

__device__ int interSWCoreInitPart15(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int3 regH2, regE2;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = (int3){0, 0, 0};

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH2.z;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 14*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.z;					//save the old H value
		regH2.z = max(regT, regE2.z);  	//adjust vecH value with vecE and vecShift
		regH2.z = max(regH2.z, regF);
				
		regT = sub_sat(regH2.z, cudaGapOE);	//calculate the new vecE
		regE2.z = max(regE2.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;


		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 14*/
			cmpRes = regF <= sub_sat(regH2.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.z = max(regH2.z, regF);
			//update vecE values from the new vecH
			regE2.z = max(regE2.z, regH2.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWCoreInitPart16(int tid, int db_cx, int db_cy, int dblen,
				volatile int* vecShift)
{
	int i, j;
	int4 regSubScore;
	int regMaxH;
	int4 regH, regE;
	int4 regH0, regE0;
	int4 regH1, regE1;
	int4 regH2, regE2;
	int	regF, regP, regT;
	int cmpRes;

	//initialize all vectors
	int4 initZero = {0, 0, 0, 0};
	regH = regE = initZero;
	regH0 = regE0 = initZero;
	regH1 = regE1 = initZero;
	regH2 = regE2 = initZero;

	//starting the main loop
	regMaxH = 0;
	for(i = 0; i < dblen; i++){
    	//initialize vecShift
      	regF = 0;
      	//initialize vecP
     	vecShift[tid] = regH2.w;
     	regP = 0;
      	if(tid > 0){
        	regP = vecShift[tid - 1];
     	}
		//loading database residue
	    int res = tex2D(InterSeqs, db_cx + i, db_cy);
		
		j = tid;
		/*compute the vector segment starting from 0*/
		regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
	
		regP = regH.x;					//save the old H value
		regH.x = max(regT, regE.x);  	//adjust vecH value with vecE and vecShift
		regH.x = max(regH.x, regF);
				
		regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
		regE.x = max(regE.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 1*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.y;					//save the old H value
		regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
		regH.y = max(regH.y, regF);
				
		regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
		regE.y = max(regE.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 2*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.z;					//save the old H value
		regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
		regH.z = max(regH.z, regF);
				
		regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
		regE.z = max(regE.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 3*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH.w;					//save the old H value
		regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
		regH.w = max(regH.w, regF);
				
		regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
		regE.w = max(regE.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 4*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.x;					//save the old H value
		regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
		regH0.x = max(regH0.x, regF);
				
		regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
		regE0.x = max(regE0.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
		/*compute the vector segment starting from 5*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.y;					//save the old H value
		regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
		regH0.y = max(regH0.y, regF);
				
		regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
		regE0.y = max(regE0.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 6*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.z;					//save the old H value
		regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
		regH0.z = max(regH0.z, regF);
				
		regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
		regE0.z = max(regE0.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 7*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH0.w;					//save the old H value
		regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
		regH0.w = max(regH0.w, regF);
				
		regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
		regE0.w = max(regE0.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 8*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.x;					//save the old H value
		regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
		regH1.x = max(regH1.x, regF);
				
		regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
		regE1.x = max(regE1.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 9*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.y;					//save the old H value
		regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
		regH1.y = max(regH1.y, regF);
				
		regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
		regE1.y = max(regE1.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 10*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.z;					//save the old H value
		regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
		regH1.z = max(regH1.z, regF);
				
		regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
		regE1.z = max(regE1.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 11*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH1.w;					//save the old H value
		regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
		regH1.w = max(regH1.w, regF);
				
		regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
		regE1.w = max(regE1.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
		/*compute the vector segment starting from 12*/
		j += THREADS_PER_WARP;
		regSubScore = tex2D(InterPackedQueryPrf, j, res);  //to the upper-left direction
		regT = regP + regSubScore.x;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.x;					//save the old H value
		regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
		regH2.x = max(regH2.x, regF);
				
		regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
		regE2.x = max(regE2.x - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 13*/
		regT = regP + regSubScore.y;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.y;					//save the old H value
		regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
		regH2.y = max(regH2.y, regF);
				
		regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
		regE2.y = max(regE2.y - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
		/*compute the vector segment starting from 14*/
		regT = regP + regSubScore.z;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.z;					//save the old H value
		regH2.z = max(regT, regE2.z);  	//adjust vecH value with vecE and vecShift
		regH2.z = max(regH2.z, regF);
				
		regT = sub_sat(regH2.z, cudaGapOE);	//calculate the new vecE
		regE2.z = max(regE2.z - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		/*compute the vector segment starting from 15*/
		regT = regP + regSubScore.w;
		regMaxH = max(regMaxH, regT);
		
		regP = regH2.w;					//save the old H value
		regH2.w = max(regT, regE2.w);  	//adjust vecH value with vecE and vecShift
		regH2.w = max(regH2.w, regF);
				
		regT = sub_sat(regH2.w, cudaGapOE);	//calculate the new vecE
		regE2.w = max(regE2.w - cudaGapExtend, regT);
		regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

		//the lazy-F loop
		for(j = 1; j < THREADS_PER_WARP; j++){
            //shift left by one element
			vecShift[tid] = regF;			//shift left
			regF = 0;
			if(tid > 0){
				regF = vecShift[tid - 1];
			}
			/*check the vector segment starting from 0*/
			cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.x = max(regH.x, regF);
			//update vecE values from the new vecH
			regE.x = max(regE.x, regH.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 1*/
			cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.y = max(regH.y, regF);
			//update vecE values from the new vecH
			regE.y = max(regE.y, regH.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 2*/
			cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.z = max(regH.z, regF);
			//update vecE values from the new vecH
			regE.z = max(regE.z, regH.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 3*/
			cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH.w = max(regH.w, regF);
			//update vecE values from the new vecH
			regE.w = max(regE.w, regH.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 4*/
			cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.x = max(regH0.x, regF);
			//update vecE values from the new vecH
			regE0.x = max(regE0.x, regH0.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 5*/
			cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.y = max(regH0.y, regF);
			//update vecE values from the new vecH
			regE0.y = max(regE0.y, regH0.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 6*/
			cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.z = max(regH0.z, regF);
			//update vecE values from the new vecH
			regE0.z = max(regE0.z, regH0.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 7*/
			cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH0.w = max(regH0.w, regF);
			//update vecE values from the new vecH
			regE0.w = max(regE0.w, regH0.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;

			/*check the vector segment starting from 8*/
			cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.x = max(regH1.x, regF);
			//update vecE values from the new vecH
			regE1.x = max(regE1.x, regH1.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 9*/
			cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.y = max(regH1.y, regF);
			//update vecE values from the new vecH
			regE1.y = max(regE1.y, regH1.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 10*/
			cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.z = max(regH1.z, regF);
			//update vecE values from the new vecH
			regE1.z = max(regE1.z, regH1.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		
			/*check the vector segment starting from 11*/
			cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH1.w = max(regH1.w, regF);
			//update vecE values from the new vecH
			regE1.w = max(regE1.w, regH1.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 12*/
			cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.x = max(regH2.x, regF);
			//update vecE values from the new vecH
			regE2.x = max(regE2.x, regH2.x - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 13*/
			cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.y = max(regH2.y, regF);
			//update vecE values from the new vecH
			regE2.y = max(regE2.y, regH2.y - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
			
			/*check the vector segment starting from 14*/
			cmpRes = regF <= sub_sat(regH2.z, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.z = max(regH2.z, regF);
			//update vecE values from the new vecH
			regE2.z = max(regE2.z, regH2.z - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
	
			/*check the vector segment starting from 15*/
			cmpRes = regF <= sub_sat(regH2.w, cudaGapOE);
			if(__all(cmpRes)){
				break;
			}
			//re-compute the vecH value from vecShift
			regH2.w = max(regH2.w, regF);
			//update vecE values from the new vecH
			regE2.w = max(regE2.w, regH2.w - cudaGapOE);
			//update the vecShift value
			regF -= cudaGapExtend;
		}
	}
	return regMaxH;
}
__device__ int interSWPartitionStep2(int tid, ushort2* globalHF, int db_cx, int db_cy, int dblen, volatile int* vecShift, int queryBaseOff)
{
	int regMaxH;

	regMaxH = 0;
	//recompute segment length
	switch((cudaPrfLength - (queryBaseOff << 2)) >> THREADS_PER_WARP_SHIFT){
	case 1: regMaxH = interSWCorePart1(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;	
	case 2: regMaxH = interSWCorePart2(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 3: regMaxH = interSWCorePart3(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 4: regMaxH = interSWCorePart4(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 5: regMaxH = interSWCorePart5(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 6: regMaxH = interSWCorePart6(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 7: regMaxH = interSWCorePart7(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 8: regMaxH = interSWCorePart8(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 9: regMaxH = interSWCorePart9(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 10: regMaxH = interSWCorePart10(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 11: regMaxH = interSWCorePart11(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 12: regMaxH = interSWCorePart12(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 13: regMaxH = interSWCorePart13(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 14: regMaxH = interSWCorePart14(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 15: regMaxH = interSWCorePart15(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	case 16: regMaxH = interSWCorePart16(tid, globalHF, queryBaseOff, db_cx, db_cy, dblen, vecShift); break;
	};
	return regMaxH;

}
__device__ int interSWPartition(int tid, ushort2* globalHF, int db_cx, int db_cy, int dblen, volatile int* vecShift, int* queryPrfOff)
{
	int i, j;
	int queryBaseOff;
	int regMaxH;
	int4 regSubScore;
    int4 regH, regE, regH0, regE0;
	int4 regH1, regE1, regH2, regE2;
    int regF, regP, regT, regM;
	ushort2 regS;
    int cmpRes;

	
	//initialize all vectors
	ushort2 initZero = {0, 0};
	for(i = tid; i < dblen; i += THREADS_PER_WARP){
		globalHF[i] = initZero;
	}
	regMaxH = 0;
	//compute one column of the alignment matrix
	for(queryBaseOff = 0; queryBaseOff < cudaPackedPrfLengthLimit; queryBaseOff += QUERY_SEGMENT_LENGTH_QUARTER){
		//initialize all vectors
    	regH = regE = (int4){0, 0, 0, 0};
    	regH0 = regE0 = regH;
		regH1 = regE1 = regH0;
		regH2 = regE2 = regH1;

		regP = 0;
		for(i = 0; i < dblen; i++){
			//load the F value of the cell
			regF = 0;
			regS = globalHF[i];
			if(tid == 0){
				regF = regS.y;
			}
			regM = regS.x;

			//initialize vecP
			vecShift[tid] = regH2.w;
			if(tid > 0){
				regP = vecShift[tid - 1];
			}
			//loading database residue
			int res = tex2D(InterSeqs, db_cx + i, db_cy);
			
			/*compute the vector segment starting from 0*/
			j = queryBaseOff + tid;
			regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
			regT = regP + regSubScore.x;
			regMaxH = max(regMaxH, regT);
	
			regP = regH.x;					//save the old H value
			regH.x = max(regT, regE.x);  	//adjust H value with E and F values
			regH.x = max(regH.x, regF);
				
			regT = sub_sat(regH.x, cudaGapOE);	//calculate the new vecE
			regE.x = max(regE.x - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 1*/
			regT = regP + regSubScore.y;
			regMaxH = max(regMaxH, regT);
		
			regP = regH.y;					//save the old H value
			regH.y = max(regT, regE.y);  	//adjust vecH value with vecE and vecShift
			regH.y = max(regH.y, regF);
				
			regT = sub_sat(regH.y, cudaGapOE);	//calculate the new vecE
			regE.y = max(regE.y - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

			/*compute the vector segment starting from 2*/
			regT = regP + regSubScore.z;
			regMaxH = max(regMaxH, regT);
		
			regP = regH.z;					//save the old H value
			regH.z = max(regT, regE.z);  	//adjust vecH value with vecE and vecShift
			regH.z = max(regH.z, regF);
				
			regT = sub_sat(regH.z, cudaGapOE);	//calculate the new vecE
			regE.z = max(regE.z - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
			/*compute the vector segment starting from 3*/
			regT = regP + regSubScore.w;
			regMaxH = max(regMaxH, regT);
		
			regP = regH.w;					//save the old H value
			regH.w = max(regT, regE.w);  	//adjust vecH value with vecE and vecShift
			regH.w = max(regH.w, regF);
				
			regT = sub_sat(regH.w, cudaGapOE);	//calculate the new vecE
			regE.w = max(regE.w - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

			/*compute the vector segment starting from 4*/
			j += THREADS_PER_WARP;
			regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
			regT = regP + regSubScore.x;
			regMaxH = max(regMaxH, regT);
	
			regP = regH0.x;					//save the old H value
			regH0.x = max(regT, regE0.x);  	//adjust vecH value with vecE and vecShift
			regH0.x = max(regH0.x, regF);
				
			regT = sub_sat(regH0.x, cudaGapOE);	//calculate the new vecE
			regE0.x = max(regE0.x - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 5*/
			regT = regP + regSubScore.y;
			regMaxH = max(regMaxH, regT);
			
			regP = regH0.y;					//save the old H value
			regH0.y = max(regT, regE0.y);  	//adjust vecH value with vecE and vecShift
			regH0.y = max(regH0.y, regF);
					
			regT = sub_sat(regH0.y, cudaGapOE);	//calculate the new vecE
			regE0.y = max(regE0.y - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
			/*compute the vector segment starting from 6*/
			regT = regP + regSubScore.z;
			regMaxH = max(regMaxH, regT);
			
			regP = regH0.z;					//save the old H value
			regH0.z = max(regT, regE0.z);  	//adjust vecH value with vecE and vecShift
			regH0.z = max(regH0.z, regF);
				
			regT = sub_sat(regH0.z, cudaGapOE);	//calculate the new vecE
			regE0.z = max(regE0.z - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 7*/
			regT = regP + regSubScore.w;
			regMaxH = max(regMaxH, regT);
			
			regP = regH0.w;					//save the old H value
			regH0.w = max(regT, regE0.w);  	//adjust vecH value with vecE and vecShift
			regH0.w = max(regH0.w, regF);
					
			regT = sub_sat(regH0.w, cudaGapOE);	//calculate the new vecE
			regE0.w = max(regE0.w - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

			/*compute the vector segment starting from 8*/
			j += THREADS_PER_WARP;
			regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
			regT = regP + regSubScore.x;
			regMaxH = max(regMaxH, regT);
	
			regP = regH1.x;					//save the old H value
			regH1.x = max(regT, regE1.x);  	//adjust vecH value with vecE and vecShift
			regH1.x = max(regH1.x, regF);
				
			regT = sub_sat(regH1.x, cudaGapOE);	//calculate the new vecE
			regE1.x = max(regE1.x - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 9*/
			regT = regP + regSubScore.y;
			regMaxH = max(regMaxH, regT);
			
			regP = regH1.y;					//save the old H value
			regH1.y = max(regT, regE1.y);  	//adjust vecH value with vecE and vecShift
			regH1.y = max(regH1.y, regF);
					
			regT = sub_sat(regH1.y, cudaGapOE);	//calculate the new vecE
			regE1.y = max(regE1.y - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
			/*compute the vector segment starting from 10*/
			regT = regP + regSubScore.z;
			regMaxH = max(regMaxH, regT);
			
			regP = regH1.z;					//save the old H value
			regH1.z = max(regT, regE1.z);  	//adjust vecH value with vecE and vecShift
			regH1.z = max(regH1.z, regF);
				
			regT = sub_sat(regH1.z, cudaGapOE);	//calculate the new vecE
			regE1.z = max(regE1.z - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 11*/
			regT = regP + regSubScore.w;
			regMaxH = max(regMaxH, regT);
			
			regP = regH1.w;					//save the old H value
			regH1.w = max(regT, regE1.w);  	//adjust vecH value with vecE and vecShift
			regH1.w = max(regH1.w, regF);
					
			regT = sub_sat(regH1.w, cudaGapOE);	//calculate the new vecE
			regE1.w = max(regE1.w - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
			
			/*compute the vector segment starting from 12*/
			j += THREADS_PER_WARP;
			regSubScore = tex2D(InterPackedQueryPrf, j, res);		//to the upper-left direction
			regT = regP + regSubScore.x;
			regMaxH = max(regMaxH, regT);
	
			regP = regH2.x;					//save the old H value
			regH2.x = max(regT, regE2.x);  	//adjust vecH value with vecE and vecShift
			regH2.x = max(regH2.x, regF);
				
			regT = sub_sat(regH2.x, cudaGapOE);	//calculate the new vecE
			regE2.x = max(regE2.x - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 13*/
			regT = regP + regSubScore.y;
			regMaxH = max(regMaxH, regT);
			
			regP = regH2.y;					//save the old H value
			regH2.y = max(regT, regE2.y);  	//adjust vecH value with vecE and vecShift
			regH2.y = max(regH2.y, regF);
					
			regT = sub_sat(regH2.y, cudaGapOE);	//calculate the new vecE
			regE2.y = max(regE2.y - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
	
			/*compute the vector segment starting from 14*/
			regT = regP + regSubScore.z;
			regMaxH = max(regMaxH, regT);
			
			regP = regH2.z;					//save the old H value
			regH2.z = max(regT, regE2.z);  	//adjust vecH value with vecE and vecShift
			regH2.z = max(regH2.z, regF);
				
			regT = sub_sat(regH2.z, cudaGapOE);	//calculate the new vecE
			regE2.z = max(regE2.z - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;
		
			/*compute the vector segment starting from 15*/
			regT = regP + regSubScore.w;
			regMaxH = max(regMaxH, regT);
			
			regP = regH2.w;					//save the old H value
			regH2.w = max(regT, regE2.w);  	//adjust vecH value with vecE and vecShift
			regH2.w = max(regH2.w, regF);
					
			regT = sub_sat(regH2.w, cudaGapOE);	//calculate the new vecE
			regE2.w = max(regE2.w - cudaGapExtend, regT);
			regF = max(regF - cudaGapExtend, regT); //calculate the new vecShift;

			//the lazy-F loop
			for(j = 1; j < THREADS_PER_WARP; j++){
				//shift left by one element
				vecShift[tid] = regF;			//shift left
				regF = 0;
				if(tid > 0){
					regF = vecShift[tid - 1];
				}
				/*check the vector segment starting from 0*/
				cmpRes = regF <= sub_sat(regH.x, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH.x = max(regH.x, regF);
				//update vecE values from the new vecH
				regE.x = max(regE.x, regH.x - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 1*/
				cmpRes = regF <= sub_sat(regH.y, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH.y = max(regH.y, regF);
				//update vecE values from the new vecH
				regE.y = max(regE.y, regH.y - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 2*/
				cmpRes = regF <= sub_sat(regH.z, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH.z = max(regH.z, regF);
				//update vecE values from the new vecH
				regE.z = max(regE.z, regH.z - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 3*/
				cmpRes = regF <= sub_sat(regH.w, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH.w = max(regH.w, regF);
				//update vecE values from the new vecH
				regE.w = max(regE.w, regH.w - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
			
				/*check the vector segment starting from 4*/
				cmpRes = regF <= sub_sat(regH0.x, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH0.x = max(regH0.x, regF);
				//update vecE values from the new vecH
				regE0.x = max(regE0.x, regH0.x - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 5*/
				cmpRes = regF <= sub_sat(regH0.y, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH0.y = max(regH0.y, regF);
				//update vecE values from the new vecH
				regE0.y = max(regE0.y, regH0.y - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 6*/
				cmpRes = regF <= sub_sat(regH0.z, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH0.z = max(regH0.z, regF);
				//update vecE values from the new vecH
				regE0.z = max(regE0.z, regH0.z - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 7*/
				cmpRes = regF <= sub_sat(regH0.w, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH0.w = max(regH0.w, regF);
				//update vecE values from the new vecH
				regE0.w = max(regE0.w, regH0.w - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
				
				/*check the vector segment starting from 8*/
				cmpRes = regF <= sub_sat(regH1.x, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH1.x = max(regH1.x, regF);
				//update vecE values from the new vecH
				regE1.x = max(regE1.x, regH1.x - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 9*/
				cmpRes = regF <= sub_sat(regH1.y, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH1.y = max(regH1.y, regF);
				//update vecE values from the new vecH
				regE1.y = max(regE1.y, regH1.y - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 10*/
				cmpRes = regF <= sub_sat(regH1.z, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH1.z = max(regH1.z, regF);
				//update vecE values from the new vecH
				regE1.z = max(regE1.z, regH1.z - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 11*/
				cmpRes = regF <= sub_sat(regH1.w, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH1.w = max(regH1.w, regF);
				//update vecE values from the new vecH
				regE1.w = max(regE1.w, regH1.w - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
				
				/*check the vector segment starting from 12*/
				cmpRes = regF <= sub_sat(regH2.x, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH2.x = max(regH2.x, regF);
				//update vecE values from the new vecH
				regE2.x = max(regE2.x, regH2.x - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 13*/
				cmpRes = regF <= sub_sat(regH2.y, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH2.y = max(regH2.y, regF);
				//update vecE values from the new vecH
				regE2.y = max(regE2.y, regH2.y - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;

				/*check the vector segment starting from 14*/
				cmpRes = regF <= sub_sat(regH2.z, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH2.z = max(regH2.z, regF);
				//update vecE values from the new vecH
				regE2.z = max(regE2.z, regH2.z - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
	
				/*check the vector segment starting from 15*/
				cmpRes = regF <= sub_sat(regH2.w, cudaGapOE);
				if(__all(cmpRes)){
					break;
				}
				//re-compute the vecH value from vecShift
				regH2.w = max(regH2.w, regF);
				//update vecE values from the new vecH
				regE2.w = max(regE2.w, regH2.w - cudaGapOE);
				//update the vecShift value
				regF -= cudaGapExtend;
			}
			regP = regM;

			//save the H and F values of the last row of this segment
			regF = max(regF, 0);
			if(tid == THREADS_PER_WARP_MASK){
				regS.x = min(regH2.w, 0x0FFFF);
				regS.y = min(regF, 0x0FFFF);
				regS.y = max(regS.y, regS.x - cudaGapOE);
				globalHF[i] = regS;
			}
		}//for i
	}//for iter

	*queryPrfOff = queryBaseOff;
	
	return regMaxH;
}
__device__ int interSWPartitionInit(int tid, int db_cx, int db_cy, int dblen, volatile int* vecShift)
{
	int score;
	int segLength = cudaPrfLength >> THREADS_PER_WARP_SHIFT;
	switch(segLength){
		case 1: score = interSWCoreInitPart1(tid, db_cx, db_cy, dblen, vecShift); break;
		case 2: score = interSWCoreInitPart2(tid, db_cx, db_cy, dblen, vecShift); break;
		case 3: score = interSWCoreInitPart3(tid, db_cx, db_cy, dblen, vecShift); break;
		case 4: score = interSWCoreInitPart4(tid, db_cx, db_cy, dblen, vecShift); break;
		case 5: score = interSWCoreInitPart5(tid, db_cx, db_cy, dblen, vecShift); break;
		case 6: score = interSWCoreInitPart6(tid, db_cx, db_cy, dblen, vecShift); break;
		case 7: score = interSWCoreInitPart7(tid, db_cx, db_cy, dblen, vecShift); break;
		case 8: score = interSWCoreInitPart8(tid, db_cx, db_cy, dblen, vecShift); break;
		case 9: score = interSWCoreInitPart9(tid, db_cx, db_cy, dblen, vecShift); break;
		case 10: score = interSWCoreInitPart10(tid, db_cx, db_cy, dblen, vecShift); break;
		case 11: score = interSWCoreInitPart11(tid, db_cx, db_cy, dblen, vecShift); break;
		case 12: score = interSWCoreInitPart12(tid, db_cx, db_cy, dblen, vecShift); break;
		case 13: score = interSWCoreInitPart13(tid, db_cx, db_cy, dblen, vecShift); break;
		case 14: score = interSWCoreInitPart14(tid, db_cx, db_cy, dblen, vecShift); break;
		case 15: score = interSWCoreInitPart15(tid, db_cx, db_cy, dblen, vecShift); break;
		case 16: score = interSWCoreInitPart16(tid, db_cx, db_cy, dblen, vecShift); break;
	};
	//write the maximum value
    vecShift[tid] = score;

    //get the maximum score
    for(unsigned int s = THREADS_PER_HALF_WARP; s > 0; s >>= 1){
        if (tid < s){
            vecShift[tid] = max(vecShift[tid], vecShift[tid + s]);
        }
    }
    return vecShift[0];
}
__global__ void interSWUsingSIMD(ushort2* cudaHF, size_t cudaHFPitch,
			DatabaseHash* hash, SeqEntry* cudaResult, int numSeqs, int firstBlk)
{
	unsigned int tid, warpNum;
	unsigned int warpId, gwarpId;
	unsigned int seqidx;
	int score, queryBaseOff;
 
	volatile __shared__ int space[THREADS_PER_BLOCK];	
	__shared__ ushort2 shrHF[WARPS_PER_BLOCK * SUBJECT_SEQUENCE_LENGTH];

	//compute the thread id
	tid = threadIdx.x;
	//compute the warp index in this thread block
	warpId = tid >> THREADS_PER_WARP_SHIFT;
	tid &= THREADS_PER_WARP_MASK;

	//compute the number of warps in a thread block
	warpNum = blockDim.x >> THREADS_PER_WARP_SHIFT;
	//calculate the index of the current thread block;
	gwarpId = blockIdx.x * warpNum;
	//calculate the index of the current thread warp
	gwarpId += warpId;

	//compute the sequence index
	seqidx = gwarpId + firstBlk * warpNum;

	if(seqidx >= numSeqs){
		return;
	}
	//get the hash item
	DatabaseHash dbhash = hash[seqidx];
	int db_cx = dbhash.cx;
	int db_cy = dbhash.cy;
	int dblen = dbhash.length;
	
	volatile int* warpShift = space + (warpId << THREADS_PER_WARP_SHIFT);
	if(dbhash.length <= SUBJECT_SEQUENCE_LENGTH){
		ushort2* warpShrHF = shrHF + warpId * SUBJECT_SEQUENCE_LENGTH;
		score = interSWPartition(tid, warpShrHF, db_cx, db_cy, dblen, warpShift, &queryBaseOff);
		score = max(score, interSWPartitionStep2(tid, warpShrHF, db_cx, db_cy, dblen, warpShift, queryBaseOff));
	}else{
		ushort2* warpGlobalHF = (ushort2*)(((char*)cudaHF) + gwarpId * cudaHFPitch);
		score = interSWPartition(tid,  warpGlobalHF, db_cx, db_cy, dblen, warpShift, &queryBaseOff);
		score = max(score, interSWPartitionStep2(tid, warpGlobalHF, db_cx, db_cy, dblen, warpShift, queryBaseOff));
	}
    //write the maximum value
    warpShift[tid] = score;

    //get the maximum score
    for(unsigned int s = THREADS_PER_HALF_WARP; s > 0; s >>= 1){
        if (tid < s){
            warpShift[tid] = max(warpShift[tid], warpShift[tid + s]);
        }
    }
	if(tid == 0){
		cudaResult[seqidx].value = warpShift[0];
	}
}
__global__ void interSWUsingSIMDInit( DatabaseHash* hash, SeqEntry* cudaResult, int numSeqs, int firstBlk)
{
	unsigned int tid, warpNum;
	unsigned int warpId, gwarpId;
	unsigned int seqidx;
	int score;
 
	volatile __shared__ int space[THREADS_PER_BLOCK];	

	//compute the thread id
	tid = threadIdx.x;
	//compute the warp index in this thread block
	warpId = tid >> THREADS_PER_WARP_SHIFT;
	tid &= THREADS_PER_WARP_MASK;

	//compute the number of warps in a thread block
	warpNum = blockDim.x >> THREADS_PER_WARP_SHIFT;
	//calculate the index of the current thread block;
	gwarpId = blockIdx.x * warpNum;
	//calculate the index of the current thread warp
	gwarpId += warpId;

	//compute the sequence index
	seqidx = gwarpId + firstBlk * warpNum;

	if(seqidx >= numSeqs){
		return;
	}
	//get the hash item
	DatabaseHash dbhash = hash[seqidx];
	
	score = interSWPartitionInit(tid,
					dbhash.cx, dbhash.cy, dbhash.length,
            		space + (warpId << THREADS_PER_WARP_SHIFT));
	if(tid == 0){
		cudaResult[seqidx].value = score;
	}

}

void CFastaSWVec::InterRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstBlk)
{
	dim3 grid (blknum, 1, 1);
	dim3 blocks (threads, 1, 1);
	
	if(this->queryPrfLength > QUERY_SEGMENT_LENGTH){
		interSWUsingSIMD<<<grid, blocks>>>(
			cudaHF, cudaHFPitch, cudaSeqHash, cudaResult, numSeqs, firstBlk);	
	}else{
		interSWUsingSIMDInit<<<grid, blocks>>>(cudaSeqHash, cudaResult, numSeqs, firstBlk);	
	}
	CUERR
	//kernel-level synchronization
	cudaThreadSynchronize();
}
/*******************************************************************
		Smith-Waterman for intra-task parallelization
Note: this kernel function is the same as the kernel function "IntraGlobalSmithWatermanScalar"
in the file CFastaSWScalar.cu. We use different names because for CUDA toolkit 3.1,
all kernel functions seem to be global scope but for earlier version, they are file scope.
Yongchao Liu, July 9, 2010
********************************************************************/
__device__ int IntraGlobalSmithWatermanVec(int matrix[32][32], ushort*D_A, ushort*D_B, ushort*D_C, 
		ushort*H, ushort*V_O, ushort*V_C, int db_cx, int db_cy, int n, unsigned char* query, int m)
{
    
	int i,j;
	unsigned char a,b;
	int dd,h,v;
	int lx,mn;
	int score;

	//the maximum number of threads is 256
	extern __shared__ int maxHH[];

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y;

	score = 0;
	mn = m;
	for(i = tid; i <= mn; i += step){
		D_A[i] = 0;
		D_B[i] = 0;
		D_C[i] = 0;
		
		H[i] = 0;
		V_O[i] = 0;
		V_C[i] = 0;
	}
	__syncthreads();
	
	/*the horizontal sequence in the edit graph is the database sequence;
		the vertical sequence in the edit graph is the query sequence*/
	mn = min(m, n);
	for( i = 1; i <= m + n - 1; i++){

		lx = i;
		int sj = 0;
		int ej = min(min(i, m + n - i), mn);
		if(i > n){
			lx = n;
			sj = i- n;
		}
		ej += sj;
		sj ++;
	
		lx -= tid;
		__syncthreads();
			
		for( j = sj + tid; j <= ej ; j += step, lx -= step){

			//calculate the vetical value
			h = max(H[j] - cudaGapExtend, D_B[j] - cudaGapOE);
			h = max(h, 0);
			//calcuate the horizontal value
			v = max(V_O[j-1] - cudaGapExtend, D_B[j-1] - cudaGapOE);
			v = max(v, 0);

			//calculate the diagonal value
			a = tex2D(IntraSeqs, db_cx + lx, db_cy);
			b = query[j];
			dd = D_A[j-1] + matrix[a][b];
			dd = max(dd, h);
			dd = max(dd, v);
			dd = min(dd, 0x0FFFF);

			//save the values
			D_C[j] = dd;
			H[j] = min(h, 0x0FFFF);
			V_C[j] = min(v, 0x0FFFF);
		
			score = max(score, dd);
		}
		
		//swap the buffers A <- B; B <- C;
		ushort* tmp = D_A;
		D_A = D_B;
		D_B = D_C;
		D_C = tmp;

		tmp = V_C;
		V_C = V_O;
		V_O = tmp;
	
	}
	maxHH[tid] = score;
	__syncthreads();
	
	if(tid == 0){
		score = maxHH[0];
		for( i = 1; i < step; i++){
			score = max(score, maxHH[i]);
		}
	}
	return score;
}

#define MAX_SM_QUERY_LENGTH				11264	//11KB by default, actually up to 12192
__global__ void intraSWUsingSIMD(ushort* D_A, size_t daPitch, ushort* D_B, size_t dbPitch, 
				ushort* D_C, size_t dcPitch, ushort* H, size_t hPitch, ushort* V_O, size_t voPitch, ushort* V_C, size_t vcPitch, DatabaseHash* hash, 
				SeqEntry* cudaResult, int numSeqs, int firstSeq)
{
	int score;
	int blkid;
	unsigned int tid;
	int step;
	int seqidx;
	int i;

	extern __shared__ unsigned char smQuery[];
	__shared__ int matrix[32][32];

	tid = threadIdx.y * blockDim.x + threadIdx.x;
	step = blockDim.x * blockDim.y;
	blkid = blockIdx.y * gridDim.x + blockIdx.x;

	//load substitution matrix from constant memory to shared memory
#if 0
	if(tid == 0){
		for(i = 0; i < 32; i++){
			for(int j = 0; j <= i; j++){
				int score = cudaSubMatrix[i][j];
				matrix[i][j] = score;
				matrix[j][i] = score;
			}
		}	
	}
#else
	int x = tid & 0x1f;  // tid %THREADS_PER_WARP
	int y = tid >> 5; //tid / THREADS_PER_WARP
	for(i = 0;i < 32;i += 8){
		matrix[i + y][x] = cudaSubMatrix[i + y][x];
	}
#endif
	__syncthreads();
	
	//compute the index of the database sequence corresponding to the current thread block
	seqidx = blkid + firstSeq;

	if(seqidx >= numSeqs){
		return;
	}
	//get the hash item of the database sequence
	DatabaseHash dbhash = hash[seqidx];

	if(cudaQueryLen >= MAX_SM_QUERY_LENGTH){
	
		score = IntraGlobalSmithWatermanVec(matrix,
				(ushort*)(((char*)D_A) + blkid * daPitch), (ushort*)(((char*)D_B) + blkid * dbPitch),
				(ushort*)(((char*)D_C) + blkid * dcPitch), (ushort*)(((char*)H) + blkid * hPitch),
				(ushort*)(((char*)V_O) + blkid * voPitch), (ushort*)(((char*)V_C) + blkid * vcPitch),
				dbhash.cx, dbhash.cy, dbhash.length, cudaQuery, cudaQueryLen);
	}else{
		//load query sequence into the shared memory
		for( i = tid + 1; i <= cudaQueryLen; i += step){
			smQuery[i] = cudaQuery[i];
		}
		__syncthreads();

	  	score = IntraGlobalSmithWatermanVec(matrix,
                (ushort*)(((char*)D_A) + blkid * daPitch), (ushort*)(((char*)D_B) + blkid * dbPitch),
                (ushort*)(((char*)D_C) + blkid * dcPitch), (ushort*)(((char*)H) + blkid * hPitch),
                (ushort*)(((char*)V_O) + blkid * voPitch), (ushort*)(((char*)V_C) + blkid * vcPitch),
				dbhash.cx, dbhash.cy, dbhash.length, smQuery, cudaQueryLen);
	}

	if(tid == 0){
		cudaResult[seqidx].value = score;
	}
}
void CFastaSWVec::IntraRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstSeq)
{
    dim3 grid (blknum, 1, 1);
    dim3 blocks (threads, 1, 1);

   	intraSWUsingSIMD<<<grid, blocks, MAX_SM_QUERY_LENGTH>>>((ushort*)cudaDA, cudaDAPitch,
            (ushort*)cudaDB, cudaDBPitch, (ushort*)cudaDC, cudaDCPitch,
            (ushort*)cudaHH, cudaHHPitch, (ushort*)cudaVV_O, cudaVV_OPitch,
			(ushort*)cudaVV_C, cudaVV_CPitch, cudaSeqHash,
            cudaResult, numSeqs, firstSeq);

    CUERR
    //kernel-level synchronization
    cudaThreadSynchronize();
}

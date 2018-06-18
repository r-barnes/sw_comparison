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

#include "CFastaSW.h"
#define CUERR { cudaError_t err;			\
	if ((err = cudaGetLastError()) != cudaSuccess) {		\
  		printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }}


CFastaSW::CFastaSW()
{
	uchar_channelDesc = cudaCreateChannelDesc<unsigned char>();
	uchar4_channelDesc = cudaCreateChannelDesc<uchar4>();
	uint_channelDesc = cudaCreateChannelDesc<unsigned int>();
	char4_channelDesc = cudaCreateChannelDesc<char4>();
	sint_channelDesc = cudaCreateChannelDesc<int>();
  sint4_channelDesc = cudaCreateChannelDesc<int4>();
}
CFastaSW::~CFastaSW()
{
	//do nothing
}
//global functions
void CFastaSW::swMemcpyParameters(int matrix[32][32], int gapOpen, int gapExtend)
{
	//do nothing
}
void CFastaSW::swMemcpyQuery(unsigned char* query, int qlen, int qAlignedLen, int offset, int matrix[32][32])
{
	//do nothing
}
void CFastaSW::swMemcpyQueryLength( int qlen, int qAlignedLen)
{
	//do nothing
}
void CFastaSW::swCreateChannelFormatDesc()
{
	//do nothing
}
void* CFastaSW::swMallocArray(int width, int height, int type)
{
    cudaArray* cu_array;

    switch(type){
    case pChannelFormatKindUnsignedChar:
        cudaMallocArray( &cu_array, &uchar_channelDesc, width, height);
        break;
	case pChannelFormatKindUnsignedChar4:
		cudaMallocArray(&cu_array, &uchar4_channelDesc, width, height);
		break;
	case pChannelFormatKindUnsigned:
		cudaMallocArray(&cu_array, &uint_channelDesc, width, height);
		break;
	case pChannelFormatKindChar4:
		cudaMallocArray(&cu_array, &char4_channelDesc, width, height);
		break;
    case pChannelFormatKindSignedInt:
        cudaMallocArray( &cu_array, &sint_channelDesc, width, height);
        break;
 	case pChannelFormatKindSignedInt4:
        cudaMallocArray( &cu_array, &sint4_channelDesc, width, height);
        break;
		default:
			fprintf(stderr, "Unknown cuda array type\n");
			exit(0);
    }

    CUERR
    return cu_array;

}
void CFastaSW::swBindTextureToArray()
{
	//do nothing
}
void CFastaSW::swBindQueryProfile()
{
	//do nothing
}
void CFastaSW::swUnbindTexture()
{
	//do nothing
}
void CFastaSW::swUnbindQueryProfile()
{
	//do nothing
}
void CFastaSW::swInterMallocThreadSlots(int threads, int multiProcessors, int slotSize)
{
	//do nothing
}
void CFastaSW::swInterFreeThreadSlots()
{
	//do nothing
}

void CFastaSW::swIntraMallocThreadSlots(int multiProcessors, int slotSize)
{
	int slots;
	
	//calculate the number of slots to be allocated
	slots = multiProcessors;
	
	cudaDA = pMallocPitch(sizeof(ushort),slotSize,slots, &cudaDAPitch);
	cudaDB = pMallocPitch(sizeof(ushort),slotSize,slots, &cudaDBPitch);
	cudaDC = pMallocPitch(sizeof(ushort),slotSize, slots, &cudaDCPitch);
	cudaHH = pMallocPitch(sizeof(ushort),slotSize,slots,&cudaHHPitch);
	cudaVV_O = pMallocPitch(sizeof(ushort),slotSize,slots,&cudaVV_OPitch);
	cudaVV_C = pMallocPitch(sizeof(ushort),slotSize,slots, &cudaVV_CPitch);
}
void CFastaSW::swIntraFreeThreadSlots()
{
	pFree(cudaDA);
	pFree(cudaDB);
	pFree(cudaDC);
	pFree(cudaHH);
	pFree(cudaVV_O);
	pFree(cudaVV_C);
}
void CFastaSW::InterRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstBlk)
{
	//no nothing
}
void CFastaSW::IntraRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstSeq)
{
	//do nothing
}

void CFastaSW::transferResult(int numSeqs)
{
    cudaMemcpy(hostResult, cudaResult, numSeqs * sizeof(SeqEntry), cudaMemcpyDeviceToHost);
    CUERR;
}

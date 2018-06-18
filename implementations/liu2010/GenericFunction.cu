/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com#
* #
* # GPL 2.0 applies.
* ************************************************/

#include "GenericFunction_cu.h"
#include<stdio.h>
#include<string.h>
#include<stdlib.h>

#define CUERR { cudaError_t err;	\
if ((err = cudaGetLastError()) != cudaSuccess){printf("CUDA error: %s, function: %s  line %d\n", cudaGetErrorString(err), __FUNCTION__, __LINE__); }}
 
static const enum cudaMemcpyKind kinds[]={
	cudaMemcpyHostToDevice,
	cudaMemcpyDeviceToHost,
	cudaMemcpyDeviceToDevice
};
GPUInfo* gpuInfo = 0;

GPUInfo* pInitDevice(int argc, char* argv[])
{
	int i;
	gpuInfo = (GPUInfo*) malloc (sizeof(GPUInfo));
	if(!gpuInfo){
		fprintf(stderr, "memory allocation failed\n");
		exit(0);
	}
	//get the number of CUDA-enabled GPUs
	gpuInfo->n_device = 0;
	cudaGetDeviceCount(&gpuInfo->n_device);
	CUERR
	if(gpuInfo->n_device <= 0){
		fprintf(stderr, "There is no CUDA-enabled device avaiable\n");
		exit(0);
	}
	gpuInfo->devices = (int*) malloc (sizeof(int) * gpuInfo->n_device);
	gpuInfo->props = (cudaDeviceProp*) malloc ( sizeof(cudaDeviceProp) * gpuInfo->n_device);
	int realDevice = 0;
	for(i = 0; i < gpuInfo->n_device; i++){
		gpuInfo->devices[realDevice] = i;
		cudaGetDeviceProperties(&gpuInfo->props[realDevice], i);
		//check the compute capability of the device if it is >= 1.2
		if (gpuInfo->props[realDevice].regsPerBlock < 16384){
			continue;
		}
		CUERR
		printf("--------------------------------\n");
		printf("---------device(%d)-------------\n", i);
		printf("---------------------------------\n");
		printf("name:%s\n",gpuInfo->props[realDevice].name);
		printf("multiprocessor count:%d\n", gpuInfo->props[realDevice].multiProcessorCount);
		printf("clock rate:%d\n", gpuInfo->props[realDevice].clockRate);
		printf("shared memory:%ld\n", gpuInfo->props[realDevice].sharedMemPerBlock);
		printf("global  memory:%ld\n", gpuInfo->props[realDevice].totalGlobalMem);
		printf("registers per block:%d\n", gpuInfo->props[realDevice].regsPerBlock);

		realDevice ++;
	}
	gpuInfo->n_device = realDevice;
	printf("Only %d devices with compute capability >= 1.2\n", gpuInfo->n_device);
	
	return gpuInfo;
}
void pExitDevice(GPUInfo* info)
{
	if(!info)	return;
	if(info->devices) free(info->devices);
	if(info->props) free(info->props);
}
GPUInfo* pGetGPUInfo()
{
	return gpuInfo;
}
void pSetDevice(GPUInfo* info, int dev)
{
	cudaSetDevice(dev);
	CUERR
}
int pGetMultiProcessorCount(GPUInfo* info, int dev)
{
	return info->props[dev].multiProcessorCount;
}
int pGetRegistersPerBlock(GPUInfo* info, int dev)
{	
	return info->props[dev].regsPerBlock;
}
int pGetMultiProcessorCount()
{
	return gpuInfo->props[0].multiProcessorCount;
}
int pGetRegistersPerBlock()
{	
	return gpuInfo->props[0].regsPerBlock;
}
void* pMallocHost(size_t size)
{
	void* host;
#ifndef UNIX_EMU	
	cudaMallocHost(&host,size);
#else
	host = malloc(size);
#endif
	CUERR
	
	return host;
}
void pFreeHost(void*host)
{
#ifndef UNIX_EMU
	cudaFreeHost(host);
#else
	if(host) free(host);
#endif
	CUERR
}
void* pMallocPitch(size_t block_size, size_t width, size_t height,size_t* pitch)
{
	void* device;
	size_t devPitch;
	
	if(!pitch){
		pitch= &devPitch;
	}
	
	cudaMallocPitch((void**)&device,pitch,block_size*width, height);
	CUERR
	
	return device;
}
void pFree(void*device)
{
	cudaFree(device);
	CUERR
}
void pFreeArray(void*array)
{
	cudaFreeArray((cudaArray*)array);
	CUERR
}
void pMemcpy(void*dst, const void* src, size_t count, int kind)
{	
	cudaMemcpy(dst, src, count, kinds[kind]);
	CUERR
}
void pMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
			int kind)
{
	cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kinds[kind]);
	CUERR
}
void pMemcpyToArray(void*dst, int x,int y, const void* src, size_t count, int kind)
{
	
	cudaMemcpyToArray((cudaArray*)dst, x,y, src, count, kinds[kind]);
	CUERR
}
void pMemcpy2DToArray( void* dst, int dstx,int dsty, void*src, size_t src_pitch,
	size_t width, size_t height, int kind)
{
	cudaMemcpy2DToArray((cudaArray*)dst,dstx,dsty, src, src_pitch,width, height, kinds[kind]);
	CUERR
}
void pMemcpy2DFromArray(void*dst, size_t pitch,void*src, size_t srcx,size_t srcy,size_t width,size_t height,int kind)
{
	cudaMemcpy2DFromArray(dst,pitch,(cudaArray*)src,srcx,srcy,width,height,kinds[kind]);
	CUERR
}

/*
 * GPUBWT.cu
 *
 *  Created on: Jan 7, 2013
 *      Author: yongchao
 */
#include "GPUBWT.h"
#include "GPUSeeds.h"
#include "GPUVariables.h"

GPUBWT::GPUBWT(uint32_t* hostBwtPtr, size_t bwtSize, uint32_t dollar,
		uint32_t seqLen, uint32_t* ccounts) {

	Utils::log("load BWT (%u MB)\n", bwtSize * sizeof(uint32_t) / 1048576);

	/*calculate the width and height*/
	uint32_t bwtWidthShift = 16;
	uint32_t bwtWidthMask = (1 << bwtWidthShift) - 1;
	cudaMemcpyToSymbol(_cudaBwtWidthShift, &bwtWidthShift, sizeof(uint32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaBwtWidthMask, &bwtWidthMask, sizeof(uint32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	/*allocate space for bwt data*/
	uint32_t bwtWidth = 1 << bwtWidthShift;
	uint32_t bwtHeight = (bwtSize + bwtWidthMask) >> bwtWidthShift;
	cudaChannelFormatDesc channelDes = cudaCreateChannelDesc<uint32_t>();
	cudaMallocArray(&_bwtDevPtr, &channelDes, bwtWidth, bwtHeight);

	/*copy the data*/
	uint32_t* data;

	/*get a copy of the original data*/
	cudaMallocHost(&data, bwtWidth * bwtHeight * sizeof(uint32_t));
	myCheckCudaError;
	memcpy(data, hostBwtPtr, bwtSize * sizeof(uint32_t));

	/*copy to the CUDA array*/
	cudaMemcpy2DToArray(_bwtDevPtr, 0, 0, data,
			bwtWidth * sizeof(uint32_t), bwtWidth * sizeof(uint32_t), bwtHeight,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	/*release the temp data*/
	cudaFreeHost(data);
	myCheckCudaError;

	/*set texture parameters*/
	_texBWT.addressMode[0] = cudaAddressModeClamp;
	_texBWT.addressMode[1] = cudaAddressModeClamp;
	_texBWT.filterMode = cudaFilterModePoint;
	_texBWT.normalized = false;

	/*bind the texture memory*/
	cudaBindTextureToArray(_texBWT, _bwtDevPtr, channelDes);
	myCheckCudaError;

	/*other parameters*/
	cudaMemcpyToSymbol(_cudaBwtDollar, &dollar, sizeof(dollar), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaBwtSeqLength, &seqLen, sizeof(seqLen), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaBwtCCounts, ccounts, sizeof(uint32_t) * BWT_NUM_OCC,
			0, cudaMemcpyHostToDevice);
	myCheckCudaError;
}

GPUBWT::~GPUBWT() {
	/*unbind texture memory*/
	cudaUnbindTexture(_texBWT);
	myCheckCudaError;

	/*release device memory*/
	cudaFreeArray(_bwtDevPtr);
	myCheckCudaError;
}

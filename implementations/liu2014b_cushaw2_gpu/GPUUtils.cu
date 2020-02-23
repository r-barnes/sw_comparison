/*
 * GPUUtils.cu
 *
 *  Created on: Jan 15, 2013
 *      Author: yongchao
 */
#include "GPUUtils.h"

static __global__ void _kernelTransposeUint2(const __restrict uint2* idata,
		uint2* odata, int32_t width, int32_t height) {
	__shared__ uint2 tile[TILE_DIM][TILE_DIM + 1];

	// read the matrix tile into shared memory
	uint32_t xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	uint32_t yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	uint32_t index_in = yIndex * width + xIndex;
	tile[threadIdx.y][threadIdx.x] = idata[index_in];

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	uint32_t index_out = yIndex * height + xIndex;
	odata[index_out] = tile[threadIdx.x][threadIdx.y];
}
static __global__ void _kernelTransposeUint32(const __restrict uint32_t* idata,
		uint32_t* odata, int32_t width, int32_t height) {
	__shared__ uint32_t tile[TILE_DIM][TILE_DIM + 1];

	// read the matrix tile into shared memory
	uint32_t xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	uint32_t yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	uint32_t index_in = yIndex * width + xIndex;
	tile[threadIdx.y][threadIdx.x] = idata[index_in];

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	uint32_t index_out = yIndex * height + xIndex;
	odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

static __global__ void _kernelTransposeUint16(const __restrict uint16_t* idata,
		uint16_t* odata, int32_t width, int32_t height) {
	__shared__ uint32_t tile[TILE_DIM][TILE_DIM + 1];

	// read the matrix tile into shared memory
	uint32_t xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	uint32_t yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	uint32_t index_in = yIndex * width + xIndex;
	tile[threadIdx.y][threadIdx.x] = idata[index_in];

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	uint32_t index_out = yIndex * height + xIndex;
	odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

void GPUUtils::configKernels() {
	cudaFuncSetCacheConfig(_kernelTransposeUint2, cudaFuncCachePreferShared);
	myCheckCudaError;

	cudaFuncSetCacheConfig(_kernelTransposeUint32, cudaFuncCachePreferShared);
	myCheckCudaError;

	cudaFuncSetCacheConfig(_kernelTransposeUint16, cudaFuncCachePreferShared);
	myCheckCudaError;
}

void GPUUtils::transpose(uint2* idata, uint2* odata, int32_t width,
		int32_t height, cudaStream_t stream) {
	dim3 grid(width / TILE_DIM, height / TILE_DIM);
	dim3 blocks(TILE_DIM, TILE_DIM);

	/*shared-memory banking mode*/
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	myCheckCudaError;

	_kernelTransposeUint2<<<grid, blocks, 0, stream>>>(idata, odata, width, height);
	myCheckCudaError;

	/*shared-memory banking mode*/
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	myCheckCudaError;
}

void GPUUtils::transpose(uint32_t* idata, uint32_t* odata, int32_t width,
		int32_t height, cudaStream_t stream) {
	dim3 grid(width / TILE_DIM, height / TILE_DIM);
	dim3 blocks(TILE_DIM, TILE_DIM);

	_kernelTransposeUint32<<<grid, blocks, 0, stream>>>(idata, odata, width, height);
	myCheckCudaError;
}

void GPUUtils::transpose(uint16_t* idata, uint16_t* odata, int32_t width,
		int32_t height, cudaStream_t stream) {
	dim3 grid(width / TILE_DIM, height / TILE_DIM);
	dim3 blocks(TILE_DIM, TILE_DIM);

	_kernelTransposeUint16<<<grid, blocks, 0, stream>>>(idata, odata, width, height);
	myCheckCudaError;
}


/*
 * GPUBWT.cu
 *
 *  Created on: Jan 7, 2013
 *      Author: yongchao
 */
#include "GPUBWT.h"
#include "GPUVariables.h"

/*count the number of occurrences*/
static __device__  inline uint32_t _cudaBwtOccAux(uint64_t y, uint32_t c) {
	// reduce nucleotide counting to bits counting
	y = ((c & 2) ? y : ~y) >> 1 & ((c & 1) ? y : ~y) & 0x5555555555555555ull;
	// count the number of 1s in y
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);

	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

static __device__ inline uint32_t _cudaBwtGetElement(uint32_t position)
{
	return tex2D(_texBWT, position & _cudaBwtWidthMask, position >> _cudaBwtWidthShift);
}
/*get the number of occurrences of a base at a specific position*/
__device__ uint32_t _cudaBwtOcc(
		uint32_t base, uint32_t pos) {
	uint32_t n;
	uint64_t pack;
	uint32_t offset; /*the position of the current BWT pointer*/

	if (pos == (uint32_t) -1) {
		return 0;
	}
	if (pos == _cudaBwtSeqLength)
		return _cudaBwtCCounts[base + 1] - _cudaBwtCCounts[base];

	//for bases indexed greater equal than the $ symbol, the index is decreased by 1 because $ is removed from the final BWT
	if (pos >= _cudaBwtDollar) {
		--pos;
	}

	/*get the current offset*/
	offset = (pos >> BWT_OCC_INTERVAL_SHIFT) * BWT_OCC_PTR_OFFSET;

	/*get the cumulative occurrence of the base*/
	n = _cudaBwtGetElement(offset + base);

	/*skip the addresses of the interleaved cumulative occurrences of the four bases*/
	offset += BWT_NUM_NUCLEOTIDE;

	// calculate Occ up to the last pos/32
	uint32_t _pos = (pos >> 5) << 5;
	for (uint32_t ipos = (pos >> BWT_OCC_INTERVAL_SHIFT) * BWT_OCC_INTERVAL;
			ipos < _pos; ipos += 32) {
		pack = _cudaBwtGetElement(offset++);
		pack <<= 32;
		pack += _cudaBwtGetElement(offset++);
		n += _cudaBwtOccAux(pack, base);
	}
	// calculate Occ
	pack = _cudaBwtGetElement(offset++);
	pack <<= 32;
	pack += _cudaBwtGetElement(offset);
	n += _cudaBwtOccAux(pack & ~((1ull << ((~pos & 31) << 1)) - 1), base);
	if (base == 0)
		n -= ~pos & 31; // corrected for the masked bits

	return n;
}

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


/*
 * GPUBWT.h
 *
 *  Created on: Jan 4, 2013
 *      Author: yongchao
 */

#ifndef GPUBWT_H_
#define GPUBWT_H_
#include "GPUVariables.h"
#include "GPUMacros.h"

class GPUBWT
{
public:
	GPUBWT(uint32_t* hostBwtPtr, size_t bwtSize, uint32_t dollar,
			uint32_t seqLen, uint32_t* ccounts);
	~GPUBWT();

private:
	/*device memory for BWT*/
	cudaArray* _bwtDevPtr;
};

__device__ inline uint32_t _cudaBwtGetElement(uint32_t position)
{
	return tex2D(_texBWT, position & _cudaBwtWidthMask, position >> _cudaBwtWidthShift);
}

/*count the number of occurrences*/
__device__ inline uint32_t _cudaBwtOccAux(uint64_t y, uint32_t c) {
	// reduce nucleotide counting to bits counting
	y = ((c & 2) ? y : ~y) >> 1 & ((c & 1) ? y : ~y) & 0x5555555555555555ull;
	// count the number of 1s in y
	y = (y & 0x3333333333333333ull) + (y >> 2 & 0x3333333333333333ull);

	return ((y + (y >> 4)) & 0xf0f0f0f0f0f0f0full) * 0x101010101010101ull >> 56;
}

/*get the number of occurrences of a base at a specific position*/
__device__ inline uint32_t _cudaBwtOcc(
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

#endif /* GPUBWT_H_ */

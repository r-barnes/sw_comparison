/*
 * GPUUtils.h
 *
 *  Created on: Jan 15, 2013
 *      Author: yongchao
 */

#ifndef GPUUTILS_H_
#define GPUUTILS_H_
#include "GPUMacros.h"
#include "Sequence.h"

#define TILE_DIM		16

class GPUUtils
{
public:
	static void configKernels();

	/*transpose the matrix*/
	static void transpose(uint2* idata, uint2* odata, int32_t width,
			int32_t height, cudaStream_t stream);

	static void transpose(uint32_t* idata, uint32_t* odata, int32_t width,
			int32_t height, cudaStream_t stream);

	static void transpose(uint16_t* idata, uint16_t* odata, int32_t width,
			int32_t height, cudaStream_t stream);
};

#endif /* GPUUTILS_H_ */

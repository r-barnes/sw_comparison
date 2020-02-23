/*
 * GPUSW.h
 *
 *  Created on: Sep 3, 2012
 *      Author: yongchao
 */

#ifndef GPUSW_H_
#define GPUSW_H_
#include "GPUMacros.h"

class GPUSW
{
public:
	GPUSW() {
	}
	~GPUSW() {
	}

	/*configure kernels*/
	void configKernels();

	/*score-only SW algorithm*/
	void initAlignScore(uint2*devSeeds, size_t numSeeds);
	void finalizeAlignScore();
	void alignScore(uint32_t* devAlignScores, size_t numSeeds, int nthreads,
			cudaStream_t stream);

	/*SW alignment with traceback*/
	void initAlign(uint2* devSeeds, uint32_t *devReadIndices, size_t numSeeds);
	void finalizeAlign();
	void align(uint16_t* devCigars, uint32_t devCigarWidth,
			uint32_t maxNumCigars, uint8_t* devNumCigars, uint2* devAligns,
			float* devBasePortions, size_t numSeeds, int nthreads,
			cudaStream_t stream);
};
#endif /* GPUSW_H_ */

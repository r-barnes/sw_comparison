/*
 * GPUBWT.h
 *
 *  Created on: Jan 4, 2013
 *      Author: yongchao
 */

#ifndef GPUBWT_H_
#define GPUBWT_H_
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
extern __device__ uint32_t _cudaBwtOcc(uint32_t base, uint32_t pos);

#endif /* GPUBWT_H_ */

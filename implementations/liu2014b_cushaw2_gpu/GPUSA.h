/*
 * GPUSA.h
 *
 *  Created on: Jan 4, 2013
 *      Author: yongchao
 */

#ifndef GPUSA_H_
#define GPUSA_H_
#include "Options.h"
#include "Sequence.h"
#include "GPUMacros.h"
#include "Bitmap.h"
#include "GPUUtils.h"

class GPUSA
{
public:
	GPUSA(Options* options);
	~GPUSA();

	/*calculate SA intervals*/
	void getSAIntervals(int32_t maxNumSAsPerRead, int32_t numSeqs, int nthreads,
			cudaStream_t stream);

	/*write the results to files*/
	void save(Bitmap* membership, const string& saFileName);

	/*config kernels*/
	void configKernels();

	/*load parameters*/
	void loadParams(uint32_t maxSeedOcc, float minID, float minBaseRatio);

	/*load reads*/
	uint32_t loadReads(Sequence *seqs, int32_t numSeqs);
	void unloadReads();

private:
	Options *_options;

	/*for read batch*/
	int4* _devHash;
	uint32_t *_devReads;

	/*output buffers*/
	uint2 *_devSAs, *_transposedSAs;
	uint32_t _devSASize;
	uint32_t *_devSeeds, *_transposedSeeds;

	uint2 *_hostSAs;
	uint32_t *_hostSeeds;
	size_t _hostSAWidth;
	size_t _hostSAHeight;

	void _loadParams(uint32_t maxReadLength);
};

#endif /* GPUSA_H_ */

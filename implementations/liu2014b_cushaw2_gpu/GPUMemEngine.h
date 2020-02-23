/*
 * GPUMemEngine.h
 *
 *  Created on: Aug 29, 2012
 *      Author: yongchao
 */

#ifndef GPUMEMENGINE_H_
#define GPUMEMENGINE_H_
#include "Macros.h"
#include "GPUBWT.h"
#include "Options.h"
#include "Utils.h"
#include "Genome.h"
#include "Sequence.h"
#include "SAM.h"
#include "GPUSA.h"
#include "GPUSeeds.h"
#include "MemEngine.h"

#define MAX_READS_PER_BATCH			0x100000

class GPUMemEngine : public MemEngine
{
public:
	GPUMemEngine(Options* options, Genome* rgenome, SAM* sam);
	~GPUMemEngine();

	/*run the engine*/
	size_t run(Sequence* sequences, size_t numSequences,
			size_t maxNumSeedsPerBatch = 204800);

	/*init the engine*/
	void initialize(int32_t gpuID);
	void finalize();

private:
	/*for GPU computing*/
	GPUBWT * _gpuBWT;
	GPUSA* _gpuSA;
	GPUSeeds* _gpuSeeds;
	GPUInfo *_gpuInfo;
	uint32_t* _gpuReadOccs;

	/*for alignment*/
	string _saFileName;
	string _seedFileName;
	Bitmap *_membership, *_seedMembership;
	uint8_t* _gpuMapQuals;

	/*load and unload reads*/
	uint32_t _maxReadLength;
	bool _paired;
	bool _unique;

	/*load and unload BWT data*/
	inline void _loadBWT() {
		_gpuBWT = new GPUBWT(_rbwt->getBWTPtr(), _rbwt->getBWTSize(),
				_rbwt->getDollarPos(), _rbwt->getBwtSeqLength(),
				_rbwt->getCCounts());
	}
	inline void _unloadBWT() {
		if (_gpuBWT) {
			delete _gpuBWT;
			_gpuBWT = NULL;
		}
	}

	/*calculate suffix array intervals*/
	void _calcSAIntervals(size_t numSequences, Bitmap* membership);

	/*calculate top seed hits*/
	void _selectTopHits(Sequence* sequences, size_t numSequences,
			Bitmap* membership, Bitmap *seedMemship,
			size_t maxNumSeedsPerBatch);

	/*trace-back to find the alignments*/
	size_t _align(Sequence* sequences, size_t numSequences, Bitmap* membership,
			uint8_t *mapQuals, size_t maxNumSeedsPerBatch);

	/*output the alignments*/
	size_t _print(Sequence* sequences, size_t numSequences,
			uint32_t* readIndices, uint16_t* cigars, uint8_t *numCigars,
			uint2* aligns, uint8_t* mapQuals, size_t numSeeds,
			uint32_t *readOccs, float* basePortions);

	size_t _printPaired(Sequence* sequences, size_t numSequences,
			uint32_t* readIndices, uint16_t* cigars, uint8_t *numCigars,
			uint2* aligns, uint8_t* mapQuals, size_t numSeeds,
			uint32_t* readOccs, float* basePortions);
};

#endif /* GPUMEMENGINE_H_ */

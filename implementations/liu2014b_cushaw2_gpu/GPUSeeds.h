/*
 * GPUSeeds.h
 *
 *  Created on: Jan 7, 2013
 *      Author: yongchao
 */

#ifndef GPUSEEDS_H_
#define GPUSEEDS_H_
#include "GPUMacros.h"
#include "Bitmap.h"
#include "SuffixArray.h"
#include "GPUSW.h"
#include "Options.h"
#include "Genome.h"
#include "Sequence.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

class GPUSeeds
{
public:
	GPUSeeds(Options* options) {
		_options = options;
		_membership = NULL;
		_file = NULL;
		_maxNumSeeds = 0;
		_seqBaseIndex = 0;
		_numSeqs = 0;

		_buffer = NULL;
		_buffer2 = NULL;
		_bufferSize = 0;

		/*create GPU SW engine*/
		_gpuSW = new GPUSW();

		_multiAlign = _options->getMaxMultiAligns();
	}
	~GPUSeeds() {
		if (_buffer) {
			delete[] _buffer;
		}
		if (_buffer2) {
			delete[] _buffer2;
		}
	}

	/*reset all paramters*/
	inline void reset(Bitmap* membership, FILE* safile, size_t maxNumIndices,
			size_t numSeqs) {
		_membership = membership;
		_file = safile;

		/*other variables*/
		_maxNumSeeds = maxNumIndices;
		_seqBaseIndex = 0;
		_numSeqs = numSeqs;
	}

	inline size_t getSeqBaseIndex() {
		return _seqBaseIndex;
	}

	/*configure kernels*/
	void configKernels();

	/* suffix array*/
	void loadSuffixArray(SuffixArray* sa);
	void unloadSuffixArray();

	/*for genome sequence*/
	void loadGenome(Genome* genome);
	void unloadGenome();

	/*load Smith Waterman parameters*/
	void loadParameters(int32_t match, int32_t mismatch, int32_t gapOpen,
			int32_t gapExtend);

	/*read a batch of suffix array indices from the file*/
	void readSAindices(thrust::host_vector<uint2>& seeds,
			thrust::host_vector<uint32_t>& readIndices, uint32_t* readOccs);

	/*calculate mapping positions*/
	void calcMapPositions(uint2* devSeeds, size_t numSeeds, int32_t nthreads,
			cudaStream_t stream);

	/*perform SW algorithm*/
	void calcAlignScores(uint2 *devSeeds, uint32_t* devAlignScores,
			size_t numSeeds, int32_t nthreads, cudaStream_t stream);
	void calcAlign(uint2* devSeeds, uint32_t* devReadIndices,
			uint16_t* devCigar, uint32_t devCigarWidth, uint32_t maxNumCigars,
			uint8_t *devNumCigars, uint2 *devAligns, float* devBasePortions,
			size_t numSeeds, int32_t nthreads, cudaStream_t stream);

	/*save the top seeds*/
	size_t save(Sequence* sequences, uint2* seeds, uint32_t *alignScores,
			uint32_t *readOccs, uint8_t *mapQuals, size_t numSeeds,
			Bitmap* membership, FILE* file, uint32_t minAlignScore);

	/*read a batch of qualified seeds*/
	void readQualifiedSeeds(thrust::host_vector<uint2>& seeds,
			thrust::host_vector<uint32_t>& readIndices, uint32_t* readOccs);

private:
	Options* _options;
	Bitmap* _membership;
	FILE* _file;
	size_t _maxNumSeeds;
	size_t _seqBaseIndex;
	size_t _numSeqs;
	uint32_t _multiAlign;

	/*for suffix array*/
	uint32_t *_devSuffixArray;

	/*for genome*/
	cudaArray* _devPacGenome;
	uint32_t* _devBwtAnns;

	/*for Smith-Waterman algorithm*/
	GPUSW* _gpuSW;

	/*buffer*/
	inline void _resize(size_t newSize) {
		if (newSize <= _bufferSize) {
			return;
		}

		_bufferSize = newSize + 1024;
		if (_buffer) {
			delete[] _buffer;
		}
		if (_buffer2) {
			delete[] _buffer2;
		}
		_buffer = new uint2[_bufferSize];
		_buffer2 = new uint32_t[_bufferSize];
	}
	uint2* _buffer;
	uint32_t *_buffer2;
	size_t _bufferSize;

	void _initMapQualCalc(uint2*devSeeds, uint32_t* devReadIndices,
			uint32_t* devReadOccs, size_t numSequences, size_t numSeeds);
	void _finalizeMapQualCalc();
};

#endif /* GPUSEEDS_H_ */

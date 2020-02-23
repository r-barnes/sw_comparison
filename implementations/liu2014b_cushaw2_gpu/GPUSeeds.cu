/*
 * GPUSeeds.cu
 *
 *  Created on: Jan 7, 2013
 *      Author: yongchao
 */
#include "GPUSeeds.h"
#include "GPUBWT.h"
#include "GPUVariables.h"

static __device__              inline uint8_t _devBwtGetBase(uint32_t bwtPos) {
	uint32_t shift;
	uint32_t offset;

	offset = (bwtPos >> BWT_OCC_INTERVAL_SHIFT) * BWT_OCC_PTR_OFFSET;
	offset += BWT_NUM_NUCLEOTIDE;
	offset += ((bwtPos & BWT_OCC_INTERVAL_MASK) >> 4);
	shift = (~bwtPos & 0x0F) << 1;

	return (tex2D(_texBWT, offset & _cudaBwtWidthMask,
			offset >> _cudaBwtWidthShift) >> shift) & 3;
}
static __device__              inline uint32_t _devBwtLFCM(uint32_t bwtPos) {
	uint8_t base;
	if (bwtPos == _cudaBwtDollar)
		return 0;
	if (bwtPos < _cudaBwtDollar) {
		base = _devBwtGetBase(bwtPos);
		return _cudaBwtCCounts[base] + _cudaBwtOcc(base, bwtPos);
	}
	base = _devBwtGetBase(bwtPos - 1);

	return _cudaBwtCCounts[base] + _cudaBwtOcc(base, bwtPos);
}
static __device__              inline uint32_t _devBwtGetMarkedPos(uint32_t pos,
		uint32_t& mapOff) {
	uint32_t off = 0;
	while (pos % _cudaSaFactor != 0) {
		++off;
		pos = _devBwtLFCM(pos);
	}
	mapOff = off;

	return pos;
}

static __device__              inline uint32_t _devGetPosition(
		const __restrict uint32_t* cudaSuffixArray, uint32_t pos) {
	uint32_t markedPosition;
	uint32_t markedOff;

	//get the marked position and its offset
	markedPosition = _devBwtGetMarkedPos(pos, markedOff);

	//get the mapping position
	return markedOff + cudaSuffixArray[markedPosition / _cudaSaFactor];
}
static __global__ void kernelGetPositions(uint2* cudaSeeds, uint32_t numSeeds,
		const __restrict uint32_t* cudaSuffixArray) {
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t target;
	uint2 seed;

	if (gid >= numSeeds) {
		return;
	}

	/*read the suffix array index*/
	seed = cudaSeeds[gid];

	/*get the position*/
	target = _devGetPosition(cudaSuffixArray, seed.x);
	target = _cudaBwtSeqLength - target - GET_SEED_LENGTH(seed.y);
	//printf("gid: %d seedPosition %u seedLength %u strand %d target: %u\n", gid, GET_SEED_POS(seed.y), GET_SEED_LENGTH(seed.y), GET_SEED_STRAND(seed.y), target);

	/*save the index*/
	cudaSeeds[gid].x = target;
}
void GPUSeeds::loadParameters(int32_t match, int32_t mismatch, int32_t gapOpen,
		int32_t gapExtend) {
	int32_t gapOE = gapOpen + gapExtend;

	/*negative mismatch score*/
	mismatch = -mismatch;

	cudaMemcpyToSymbol(_cudaMatchScore, &match, sizeof(int32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaMismatchScore, &mismatch, sizeof(int32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaGapOE, &gapOE, sizeof(int32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaGapExtend, &gapExtend, sizeof(int32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

}
void GPUSeeds::configKernels() {
	_gpuSW->configKernels();
}
void GPUSeeds::loadSuffixArray(SuffixArray *sa) {
	uint32_t saFactor = sa->getFactor();
	uint32_t* saPtr = sa->getData();

	/*get the device pointer of the mapped memory*/
	cudaHostGetDevicePointer(&_devSuffixArray, saPtr, 0);
	myCheckCudaError;

	/*copy the factor*/
	cudaMemcpyToSymbol(_cudaSaFactor, &saFactor, sizeof(uint32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;
}
void GPUSeeds::unloadSuffixArray() {

#if 0
	cudaFreeHost(_hostSuffixArray);
	myCheckCudaError;
#endif
}

void GPUSeeds::loadGenome(Genome* genome) {

	uint8_t* pacGenome = genome->getPacGenome();
	uint32_t genomeSize = (genome->getGenomeLength() + 3) >> 2;
	Utils::log("load packed genome (%ld MB)\n", genomeSize / 1048576);

	/*allocate memory*/
	uint32_t genomeWidthShift = 16;
	uint32_t genomeWidthMask = (1 << genomeWidthShift) - 1;
	cudaMemcpyToSymbol(_cudaGenomeWidthShift, &genomeWidthShift,
			sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaMemcpyToSymbol(_cudaGenomeWidthMask, &genomeWidthMask, sizeof(uint32_t),
			0, cudaMemcpyHostToDevice);
	myCheckCudaError;

	/*allocate memory*/
	uint32_t genomeWidth = 1 << genomeWidthShift;
	uint32_t genomeHeight = (genomeSize + genomeWidthMask) >> genomeWidthShift;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();

	cudaMallocArray(&_devPacGenome, &channelDesc, genomeWidth, genomeHeight);
	myCheckCudaError;

	/*copy data*/
	uint8_t* data;
	cudaMallocHost(&data, genomeWidth * genomeHeight);
	myCheckCudaError;
	memcpy(data, pacGenome, genomeWidth * genomeHeight);

	cudaMemcpy2DToArray(_devPacGenome, 0, 0, data,
			genomeWidth * sizeof(uint8_t), genomeWidth * sizeof(uint8_t),
			genomeHeight, cudaMemcpyHostToDevice);
	myCheckCudaError;

	/*bind the texture*/
	_texPacGenome.addressMode[0] = cudaAddressModeClamp;
	_texPacGenome.addressMode[1] = cudaAddressModeClamp;
	_texPacGenome.filterMode = cudaFilterModePoint;
	_texPacGenome.normalized = false;

	cudaBindTextureToArray(_texPacGenome, _devPacGenome, channelDesc);
	myCheckCudaError;

	/*load the genome indices information*/
	BwtAnn* bwtAnns = genome->getBwtAnn();
	uint32_t numGenomicSeqs = genome->getNumSeqs();
	uint32_t* hostBwtAnns;

	/*allocate host buffer*/
	cudaMallocHost(&hostBwtAnns, numGenomicSeqs * sizeof(uint32_t));
	myCheckCudaError;

	for (uint32_t i = 0; i < numGenomicSeqs; ++i) {
		hostBwtAnns[i] = bwtAnns->_offset;
		++bwtAnns;
	}

	/*allocate device buffer*/
	cudaMalloc(&_devBwtAnns, numGenomicSeqs * sizeof(uint32_t));
	myCheckCudaError;

	/*copy data*/
	cudaMemcpy(_devBwtAnns, hostBwtAnns, numGenomicSeqs * sizeof(uint32_t),
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaFreeHost(hostBwtAnns);
	myCheckCudaError;

	/*bind texture*/
	_texBwtAnns.addressMode[0] = cudaAddressModeClamp;
	_texBwtAnns.filterMode = cudaFilterModePoint;
	_texBwtAnns.normalized = false;

	channelDesc = cudaCreateChannelDesc<uint32_t>();
	cudaBindTexture(NULL, _texBwtAnns, _devBwtAnns, channelDesc,
			numGenomicSeqs * sizeof(uint32_t));
	myCheckCudaError;

	/*copy other parameters*/
	cudaMemcpyToSymbol(_cudaNumGenomicSeqs, &numGenomicSeqs, sizeof(uint32_t),
			0, cudaMemcpyHostToDevice);
	myCheckCudaError;

}
void GPUSeeds::unloadGenome() {
	cudaUnbindTexture(_texPacGenome);
	myCheckCudaError;

	cudaFreeArray(_devPacGenome);
	myCheckCudaError;

	cudaUnbindTexture(_texBwtAnns);
	myCheckCudaError;

	cudaFree(_devBwtAnns);
	myCheckCudaError;
}

/*calculate mapping positions*/
void GPUSeeds::calcMapPositions(uint2* devSeeds, size_t numSeeds,
		int32_t nthreads, cudaStream_t stream) {
	int32_t nblocks = (numSeeds + nthreads - 1) / nthreads;
	dim3 grid(nblocks, 1);
	dim3 blocks(nthreads, 1);

	kernelGetPositions<<<grid, blocks, 0, stream>>>(devSeeds, numSeeds, _devSuffixArray);
	myCheckCudaError;

}
void GPUSeeds::readSAindices(thrust::host_vector<uint2>& seeds,
		thrust::host_vector<uint32_t>& readIndices, uint32_t *readOccs) {
	uint2 range;
	size_t numRanges;
	size_t numSeeds = 0;
	size_t numOccs = 0;
	uint32_t seed;

	/*clear the buffers*/
	seeds.clear();
	readIndices.clear();

	/*read the data from the file*/
	while (_seqBaseIndex < _numSeqs) {

		/*check the existence of SAs for the current read*/
		if (!_membership->test(_seqBaseIndex)) {
			readOccs[_seqBaseIndex++] = 0;
			continue;
		}

		/*read the number of SAs*/
		if (fread(&range, sizeof(uint2), 1, _file) != 1) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}
		if (range.x == 0) {
			Utils::exit(
					"Data inconsistent in the SA file at line %d in file %s\n",
					__LINE__, __FILE__);
		}

		/*check the constraint on the number of seeds*/
		if (numSeeds > 0 && numSeeds + range.x > _maxNumSeeds) {
			/*move back the file pointer*/
			fseek(_file, -sizeof(uint2), SEEK_CUR);
			break;
		}

		/*resize the buffer*/
		_resize(range.x);

		/*read the suffix array intervals*/
		if (fread(_buffer, sizeof(uint2), range.x, _file) != range.x) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}

		/*read the correspondong seeds*/
		if (fread(_buffer2, sizeof(uint32_t), range.x, _file) != range.x) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}

		/*save the suffix array intervals*/
		numOccs = 0;
		numRanges = range.x;
		for (size_t i = 0; i < numRanges; ++i) {
			range = _buffer[i];
			seed = _buffer2[i];

			/*Utils::log("ranges: %u %u\n", range.x, range.y);*/
			/*if(GET_SEED_LENGTH(seed) > 100){
			 Utils::log("seed: %u %u %u readindex %d numSeqs %d\n", GET_SEED_LENGTH(seed), GET_SEED_STRAND(seed), GET_SEED_POS(seed), _seqBaseIndex, _numSeqs);
			 }*/
			numSeeds += range.y - range.x + 1;
			numOccs += range.y - range.x + 1;
			for (; range.x <= range.y; ++range.x) {
				/*save the seed*/
				seeds.push_back(make_uint2(range.x, seed));

				/*save the read index*/
				readIndices.push_back(_seqBaseIndex);
			}
		}
		readOccs[_seqBaseIndex++] = numOccs;
	}
}

void GPUSeeds::calcAlignScores(uint2* devSeeds, uint32_t* devAlignScores,
		size_t numSeeds, int32_t nthreads, cudaStream_t stream) {
	/*init the data*/
	_gpuSW->initAlignScore(devSeeds, numSeeds);

	/*invode the SW engine*/
	_gpuSW->alignScore(devAlignScores, numSeeds, nthreads, stream);

	/*wait for the completion*/
	cudaDeviceSynchronize();

	/*finalize the data*/
	_gpuSW->finalizeAlignScore();
}
void GPUSeeds::calcAlign(uint2* devSeeds, uint32_t* devReadIndices,
		uint16_t* devCigar, uint32_t devCigarWidth, uint32_t maxNumCigars,
		uint8_t *devNumCigars, uint2 *devAligns, float* devBasePortions,
		size_t numSeeds, int32_t nthreads, cudaStream_t stream) {
	/*init the data*/
	_gpuSW->initAlign(devSeeds, devReadIndices, numSeeds);

	/*invode the SW engine*/
	_gpuSW->align(devCigar, devCigarWidth, maxNumCigars, devNumCigars,
			devAligns, devBasePortions, numSeeds, nthreads, stream);

	/*wait for the completion*/
	cudaDeviceSynchronize();

	/*finalize the data*/
	_gpuSW->finalizeAlign();
}

size_t GPUSeeds::save(Sequence* sequences, uint2* seeds, uint32_t* alignScores,
		uint32_t* readOccs, uint8_t* mapQuals, size_t numSeeds,
		Bitmap* membership, FILE* file, uint32_t minAlignScore) {
	uint32_t readIndex, bestScore, bestScore2;
	uint32_t numOccs, bestNumOccs;
	uint32_t *scoreStart, *scoreEnd, startIndex;
	uint2 *seedStart;
	int64_t targetPosition, targetPosition2;
	int32_t bestScoreDiff = 0, offset;

	size_t numQualifiedSeeds = 0;
	bool paired = _options->isPaired(), found;
	int32_t maxGapSize = 10, numErrors;

	/*save the data in reverse order*/
	for (int64_t seedIndex = numSeeds - 1; seedIndex >= 0;) {

		/*get the current read index*/
		readIndex = GET_READ_INDEX(alignScores[seedIndex]);

		/*get the number of seeds for the current read*/
		numOccs = readOccs[readIndex];

		/*get the highest alignment score*/
		startIndex = seedIndex + 1 - numOccs;
		bestScore = GET_ALIGN_SCORE(alignScores[startIndex]);
		//Utils::log("readIndex: %u numOccs: %ld\n", readIndex, numOccs);

		/*check the alignment score against the threshold*/
		if (bestScore < minAlignScore) {
			/*move to the next read*/
			seedIndex -= numOccs;
			continue;
		}

		/*set the existing bit*/
		membership->set(readIndex);

		/*calculate the mapping quality score*/
		scoreStart = alignScores + startIndex;
		seedStart = seeds + startIndex;
		scoreEnd = scoreStart + numOccs;

		bestNumOccs = 1;
		found = false;
		targetPosition = seedStart->x;
		targetPosition -= GET_SEED_POS(seedStart->y);
		numErrors = _options->getNumErrors(sequences[readIndex]._length);

		//Utils::log("readIndex %d bestScore %d\n", readIndex, bestScore);
		for (++scoreStart, ++seedStart; scoreStart < scoreEnd;
				++scoreStart, ++seedStart) {

			/*get the next best score*/
			bestScore2 = GET_ALIGN_SCORE(*scoreStart);
			//Utils::log("bestScore2 %d\n", bestScore2);
			if (bestScore2 != bestScore) {
				if (!found) {
					found = true;
					/*optimal local alignment score diff*/
					bestScoreDiff = bestScore - bestScore2;
				}
				break;
			}
			/*estimate the starting position of the alignment*/
			targetPosition2 = seedStart->x;
			targetPosition2 -= GET_SEED_POS(seedStart->y);
			if (!found
					&& (bestScore2 == bestScore
							&& labs(targetPosition - targetPosition2)
									>= maxGapSize + numErrors)) {
				found = true;
				/*optimal local aignment score diff*/
				bestScoreDiff = 0;
			}
			++bestNumOccs;
		}
		mapQuals[readIndex] =
				found == true ? DEFAULT_MAX_MAP_QUAL * bestScoreDiff / bestScore : DEFAULT_MAX_MAP_QUAL;

		//Utils::log("mapQual: %d bestNumOccs: %d\n", mapQuals[readIndex], bestNumOccs);
		/*confine the number of reported alignments*/
		bestNumOccs = paired ? bestNumOccs : min(bestNumOccs, _multiAlign);
		//Utils::log("seedIndex: %d alignScore: %d readIndex %d\n", seedIndex, alignScore, readIndex);

		/*resize the buffer*/
		_resize(bestNumOccs);

		/*iterate each seed*/
		offset = 0;
		seedStart = seeds + startIndex;
		scoreStart = alignScores + startIndex;
		for (scoreEnd = scoreStart + bestNumOccs; scoreStart < scoreEnd;
				++scoreStart, ++seedStart) {
			_buffer[offset] = *seedStart;
			_buffer2[offset] = *scoreStart;
			++offset;
		}
		//Utils::log("number of qualified seeds: %d\n", offset);

		/*write the number of occurences*/
		if (fwrite(&offset, sizeof(uint32_t), 1, file) != 1) {
			Utils::exit("File write failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}
		/*write all alignment scores*/
		if (fwrite(_buffer2, sizeof(uint32_t), offset, file) != offset) {
			Utils::exit("File write failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}
		/*write all seeds*/
		if (fwrite(_buffer, sizeof(uint2), offset, file) != offset) {
			Utils::exit("File write failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}

		/*calculate the total number of qualified seeds*/
		numQualifiedSeeds += offset;

		/*move to the next seeds*/
		seedIndex -= numOccs;
	}

	return numQualifiedSeeds;
}
void GPUSeeds::readQualifiedSeeds(thrust::host_vector<uint2>& seeds,
		thrust::host_vector<uint32_t>& alignScores, uint32_t* readOccs) {
	size_t numSeeds = 0;
	size_t numOccs = 0;
	size_t numReads = 0;
	bool paired = _options->isPaired();

	/*clear the buffers*/
	seeds.clear();
	alignScores.clear();

	/*read the data from the file*/
	while (_seqBaseIndex < _numSeqs) {

		/*check the existence of SAs for the current read*/
		if (!_membership->test(_seqBaseIndex)) {
			readOccs[_seqBaseIndex] = 0;

			/*move to the next read*/
			++_seqBaseIndex;
			++numReads;
			continue;
		}

		/*read the number of seeds*/
		if (fread(&numOccs, sizeof(uint32_t), 1, _file) != 1) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}
		if (numOccs == 0) {
			Utils::exit(
					"Data inconsistent in the SA file at line %d in file %s\n",
					__LINE__, __FILE__);
		}

		/*check the constraints*/
		if (numSeeds > 0 && numOccs + numSeeds > _maxNumSeeds) {
			/*for paired-end alignment, the number of reads must be even*/
			if (paired == false || (numReads & 1) == 0) {
				fseek(_file, -sizeof(uint32_t), SEEK_CUR);
				break;
			}
		}

		/*resize the buffer*/
		_resize(numOccs);

		/*read read indices and alignment scores*/
		if (fread(_buffer2, sizeof(uint32_t), numOccs, _file) != numOccs) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}

		/*read the corresponding seeds and mapping locations*/
		if (fread(_buffer, sizeof(uint2), numOccs, _file) != numOccs) {
			Utils::exit("File read failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}

		for (uint32_t i = 0; i < numOccs; ++i) {
			seeds.push_back(_buffer[i]);
			alignScores.push_back(_buffer2[i]);
		}
		readOccs[_seqBaseIndex] = numOccs;

		/*move to the next read*/
		++_seqBaseIndex;
		++numReads;

		/*increase the number of seeds*/
		numSeeds += numOccs;
	}

}

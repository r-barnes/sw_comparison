/*
 * GPUSA.cu
 *
 *  Created on: Jan 4, 2013
 *      Author: yongchao
 */
#include "GPUSA.h"
#include "GPUSeeds.h"
#include "GPUBWT.h"
#include "GPUVariables.h"

static __device__   inline uint32_t GET_BASE_FROM_PACKED_READ(int texOffset,
		int position) {
	uint32_t pacBase = tex1Dfetch(_texPacReads, texOffset + (position >> 2));
	return (pacBase >> ((position & 3) << 3)) & 0x0ff;
}

static __device__ uint32_t _devGetSAIntervals(uint32_t seqcx,
		uint32_t seqLength, uint32_t minSeedSize, uint32_t strand,
		uint2* localSAs, uint32_t* localSeeds) {
	uint32_t ch;
	uint32_t numSAs = 0;
	uint2 range, range2;
	int startPos = 0, endPos = 0, lastPos = 0;
	int numSuffixes = seqLength - minSeedSize + 1;

	//for each starting position in the query
	while (startPos < numSuffixes) {
		range = make_uint2(0, _cudaBwtSeqLength);
		range2 = make_uint2(1, 0);
		for (endPos = startPos; endPos < seqLength; endPos++) {
			/*get the base*/
			ch = GET_BASE_FROM_PACKED_READ(seqcx, endPos);
			if (ch == UNKNOWN_BASE) {
				range2 = make_uint2(1, 0);
				break;
			}

			//calculate the range
			range2.x = _cudaBwtCCounts[ch] + _cudaBwtOcc(ch, range.x - 1) + 1;
			range2.y = _cudaBwtCCounts[ch] + _cudaBwtOcc(ch, range.y);
			if (range2.x > range2.y) {
				break;
			}
			range = range2;
		}
		/*confirm the valid seed*/
		if (range.x <= range.y && endPos - startPos >= minSeedSize) {
			range.y = min(range.y, range.x + _cudaMaxSeedOcc - 1);
			if (lastPos != endPos) {
				/*save the suffix array interval*/
				localSAs[numSAs] = range;
				localSeeds[numSAs] = make_sa_seed(endPos - startPos, startPos,
						strand);
				//printf("%u %u %u %u %u\n", range.x, range.y, endPos - startPos, GET_SEED_LENGTH(localSeeds[numSAs]), GET_SEED_POS(localSeeds[numSAs]));
				++numSAs;
			}
			//update the last stop position;
			lastPos = max(lastPos, endPos);
		}
		/*re-calculate the starting position*/
		startPos = (range2.x <= range2.y) ? numSuffixes : (startPos + 1);
	}
	return numSAs;
}

static __global__ void _kernelGetSAIntervals(size_t numReads, uint2* cudaSAs,
		uint32_t* cudaSeeds, size_t cudaSAWidth) {
	int gid;
	uint2 localSAs[MAX_SEQ_LENGTH];
	uint32_t localSeeds[MAX_SEQ_LENGTH];
	uint2 *insa, *outsa;
	uint32_t *inseeds, *outseeds, numSAs;
	int4 hash;
	int32_t count = 0, minSeedSize;

	gid = threadIdx.x + blockIdx.x * blockDim.x; /*global id*/

	if (gid >= numReads) {
		return;
	}
	/*get the read information*/
	hash = tex1Dfetch(_texHash, gid);

  	/*compute the suffix array interval for the forward strand*/
  	minSeedSize = hash.w;
  	do{
    	numSAs = _devGetSAIntervals(hash.x, hash.y, minSeedSize, 0, localSAs,
      		localSeeds);

    	/*compute the suffix array interval for the reverse strand*/
   	 	numSAs = numSAs
      		+ _devGetSAIntervals(hash.x + (hash.z >> 2), hash.y, minSeedSize, 1,
          	localSAs + numSAs, localSeeds + numSAs);

    	/*recompute minSeedSize*/
    	minSeedSize = (minSeedSize + GLOBAL_MIN_SEED_SIZE) / 2;
  	}while(numSAs == 0 && ++count < 2);

	/*get the pointer*/
	outsa = cudaSAs + gid;

	/*write the number of SAs*/
	*outsa = make_uint2(numSAs, numSAs);

	/*write all SAs*/
	insa = localSAs;
	for (uint32_t i = 0; i < numSAs; ++i) {
		outsa += cudaSAWidth;
		*outsa = *insa++;
	}

	/*write all seeds*/
	inseeds = localSeeds;
	outseeds = cudaSeeds + gid;
	for (uint32_t i = 0; i < numSAs; ++i) {
		*outseeds = *inseeds++;
		/*move to the next row*/
		outseeds += cudaSAWidth;
	}
}

GPUSA::GPUSA(Options* options) {
	_options = options;
	_devSAs = _hostSAs = NULL;
	_transposedSAs = NULL;
	_transposedSeeds = NULL;
	_devSASize = 0;
	_devSeeds = _hostSeeds = NULL;
}
GPUSA::~GPUSA() {

	cudaFree(_devSAs);
	myCheckCudaError;

	cudaFree(_devSeeds);
	myCheckCudaError;

	cudaFree(_transposedSAs);
	myCheckCudaError;

	cudaFree(_transposedSeeds);
	myCheckCudaError;

	cudaFreeHost(_hostSAs);
	myCheckCudaError;

	cudaFreeHost(_hostSeeds);
	myCheckCudaError;
}

void GPUSA::configKernels() {

	/*kernel function configurations*/
	cudaFuncSetCacheConfig(_kernelGetSAIntervals, cudaFuncCachePreferL1);
	myCheckCudaError;

	/*configure utility kernels*/
	GPUUtils::configKernels();
}

void GPUSA::_loadParams(uint32_t maxReadLength) {
	cudaMemcpyToSymbol(_cudaMaxReadLength, &maxReadLength, sizeof(uint32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;
}

void GPUSA::loadParams(uint32_t maxSeedOcc, float minID, float minBaseRatio) {
	cudaMemcpyToSymbol(_cudaMaxSeedOcc, &maxSeedOcc, sizeof(uint32_t), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	minID *= 100;
	cudaMemcpyToSymbol(_cudaMinID, &minID, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	minBaseRatio *= 100;
	cudaMemcpyToSymbol(_cudaMinBaseRatio, &minBaseRatio, sizeof(float), 0,
			cudaMemcpyHostToDevice);
	myCheckCudaError;
}

/*compute the SA intervals of a batch of reads*/
void GPUSA::getSAIntervals(int32_t maxNumSAs, int32_t numSeqs, int nthreads,
		cudaStream_t stream) {

	uint32_t width, height;
	size_t memSize;
	uint32_t nblocks;

	/*maxtrix size on the device*/
	width = (numSeqs + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
	height = (maxNumSAs + TILE_DIM - 1) / TILE_DIM * TILE_DIM; /*aligned to TILE_DIM*/
	memSize = width * height;

	/*do we need to allocate memory?*/
	if (_devSASize < memSize) {
		if (_devSAs) {
			cudaFree(_devSAs);
			myCheckCudaError;
		}
		if (_devSeeds) {
			cudaFree(_devSeeds);
			myCheckCudaError;
		}
		if (_transposedSAs) {
			cudaFree(_transposedSAs);
			myCheckCudaError;
		}
		if (_transposedSeeds) {
			cudaFree(_transposedSeeds);
			myCheckCudaError;
		}
		/*release resources*/
		if (_hostSAs) {
			cudaFreeHost(_hostSAs);
			myCheckCudaError;
		}

		if (_hostSeeds) {
			cudaFreeHost(_hostSeeds);
			myCheckCudaError;
		}
		_devSASize = memSize;

		cudaMalloc(&_devSAs, _devSASize * sizeof(uint2));
		myCheckCudaError;

		cudaMalloc(&_devSeeds, _devSASize * sizeof(uint32_t));
		myCheckCudaError;

		cudaMalloc(&_transposedSAs, _devSASize * sizeof(uint2));
		myCheckCudaError;

		cudaMalloc(&_transposedSeeds, _devSASize * sizeof(uint32_t));
		myCheckCudaError;

		cudaMallocHost(&_hostSAs, _devSASize * sizeof(uint2));
		myCheckCudaError;

		cudaMallocHost(&_hostSeeds, _devSASize * sizeof(uint32_t));
		myCheckCudaError;
	}
	//Utils::log("memSize: %ld width %ld height %ld\n", memSize, width, height);

	/*launch the SA computation kernel*/
	nblocks = (numSeqs + nthreads - 1) / nthreads;
	dim3 grid(nblocks, 1);
	dim3 blocks(nthreads, 1);
	_kernelGetSAIntervals<<<grid, blocks, 0, stream>>>(numSeqs, _transposedSAs, _transposedSeeds, width);
	myCheckCudaError;

	/*transpose the output buffer*/
	GPUUtils::transpose(_transposedSAs, _devSAs, width, height, stream);
	myCheckCudaError;

	GPUUtils::transpose(_transposedSeeds, _devSeeds, width, height, stream);
	myCheckCudaError;

	/*copy to host*/
	cudaMemcpy(_hostSAs, _devSAs, memSize * sizeof(uint2),
			cudaMemcpyDeviceToHost);
	myCheckCudaError;

	cudaMemcpy(_hostSeeds, _devSeeds, memSize * sizeof(uint32_t),
			cudaMemcpyDeviceToHost);
	myCheckCudaError;

	/*save the matrix size on the host*/
	_hostSAWidth = height;
	_hostSAHeight = numSeqs;
}
void GPUSA::save(Bitmap* membership, const string& saFileName) {
	/*traverse the matrix to get the data*/
	uint32_t numSAs;
	uint2 *ranges = _hostSAs;
	uint32_t *seeds = _hostSeeds;
	size_t readIndex = 0;

	if (!_hostSAs) {
		return;
	}

	/*open the file*/
	FILE* safile = fopen(saFileName.c_str(), "wb");
	if (!safile) {
		Utils::exit("Failed to open file %s at line %d in file %s\n",
				saFileName.c_str(), __LINE__, __FILE__);
	}

	for (size_t row = 0; row < _hostSAHeight; ++row) {
		/*check the number of SAs*/
		numSAs = ranges->x;
		//Utils::log("row (%d) numSAs: %d\n", row, numSAs);
		if (numSAs > 0) {

			/*set the existing bit*/
			membership->set(readIndex);

			/*write the SAs*/
			if (fwrite(ranges, sizeof(uint2), numSAs + 1, safile)
					!= numSAs + 1) {
				Utils::exit("File writes failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}

#if 0
			/*for debugging*/
			for(uint32_t i = 0; i < numSAs; ++i) {
				uint32_t seed = seeds[i];
				if(GET_SEED_LENGTH(seed) > 100) {
					Utils::log("seed: %u %u %u readindex %d numSeqs %d numSAs %d\n", GET_SEED_LENGTH(seed), GET_SEED_STRAND(seed), GET_SEED_POS(seed), row, _hostSAHeight, numSAs);
				}
			}
#endif

			/*write all seeds*/
			if (fwrite(seeds, sizeof(uint32_t), numSAs, safile) != numSAs) {
				Utils::exit("File writes failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
		}
		/*move to the next row*/
		ranges += _hostSAWidth;
		seeds += _hostSAWidth;
		++readIndex;
	}
	/*close the file*/
	fclose(safile);
}
uint32_t GPUSA::loadReads(Sequence* seqs, int32_t numSeqs) {
	uint32_t* hostReads;
	int4* hostHash;
	int32_t offset = 0;
	cudaChannelFormatDesc channelDes;

	/*create host buffer*/
	cudaMallocHost(&hostHash, numSeqs * sizeof(hostHash[0]));
	myCheckCudaError;

	/*build hash tables*/
	int4* hashPtr = hostHash;
	uint32_t length, alnLength;

	uint32_t maxReadLength = 0;
	for (int32_t index = 0; index < numSeqs; ++index) {
		/*get the sequence length*/
		length = seqs[index]._length;
		alnLength = ((length + 3) >> 2) << 2; /*aligned to 4*/

		/*create hash data*/
		hashPtr->x = offset;
		hashPtr->y = length;
		hashPtr->z = alnLength;
		hashPtr->w = _options->getMinSeedSize(length);
		offset += alnLength >> 1; /*two strands: alnLength * 2 / 4*/

		/*calculate the maximum sequence length*/
		maxReadLength = max(maxReadLength, length);

		/*increase the hash entry*/
		++hashPtr;
	}
	/*load read hash tables*/
	cudaMalloc(&_devHash, numSeqs * sizeof(hostHash[0]));
	myCheckCudaError;

	cudaMemcpy(_devHash, hostHash, numSeqs * sizeof(hostHash[0]),
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaFreeHost(hostHash);
	myCheckCudaError;

	/*set texture parameters*/
	_texHash.addressMode[0] = cudaAddressModeClamp;
	_texHash.filterMode = cudaFilterModePoint;
	_texHash.normalized = false;

	/*bind texture memory*/
	channelDes = cudaCreateChannelDesc<int4>();
	cudaBindTexture(NULL, _texHash, _devHash, channelDes,
			numSeqs * sizeof(hostHash[0]));
	myCheckCudaError;

	/*create read base buffer*/
	cudaMallocHost(&hostReads, offset * sizeof(hostReads[0]));
	myCheckCudaError;

	uint32_t* bases = hostReads;
	uint32_t base, pacBase;
	for (Sequence *p = seqs, *q = seqs + numSeqs; p < q; ++p) {
		length = p->_length;
		alnLength = ((length + 3) >> 2) << 2;	/*aligned to 4*/

		/*forward strand*/
		uint8_t *src = p->_bases;
		for (uint32_t i = 0; i < (length >> 2); ++i) {
			pacBase = *src++;

			base = *src++;
			pacBase |= base << 8;

			base = *src++;
			pacBase |= base << 16;

			base = *src++;
			pacBase |= base << 24;

			*bases++ = pacBase;
		}
		for (uint32_t i = (length >> 2) << 2; i < alnLength;) {
			pacBase = (i++ < length) ? *src++ : DUMMY_BASE;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 8;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 16;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 24;

			*bases++ = pacBase;
		}

		/*reverse strand*/
		src = p->_rbases;
		for (uint32_t i = 0; i < (length >> 2); ++i) {
			pacBase = *src++;

			base = *src++;
			pacBase |= base << 8;

			base = *src++;
			pacBase |= base << 16;

			base = *src++;
			pacBase |= base << 24;

			*bases++ = pacBase;
		}
		for (uint32_t i = (length >> 2) << 2; i < alnLength;) {
			pacBase = (i++ < length) ? *src++ : DUMMY_BASE;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 8;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 16;

			base = (i++ < length) ? *src++ : DUMMY_BASE;
			pacBase |= base << 24;

			*bases++ = pacBase;
		}
	}

	/*load reads*/
	cudaMalloc(&_devReads, offset * sizeof(hostReads[0]));
	myCheckCudaError;

	cudaMemcpy(_devReads, hostReads, offset * sizeof(hostReads[0]),
			cudaMemcpyHostToDevice);
	myCheckCudaError;

	cudaFreeHost(hostReads);
	myCheckCudaError;

	/*set texture parameters*/
	_texPacReads.addressMode[0] = cudaAddressModeClamp;
	_texPacReads.filterMode = cudaFilterModePoint;
	_texPacReads.normalized = false;

	channelDes = cudaCreateChannelDesc<uint32_t>();
	cudaBindTexture(NULL, _texPacReads, _devReads, channelDes,
			offset * sizeof(uint32_t));
	myCheckCudaError;

	/*load parameters*/
	_loadParams(maxReadLength);

	return maxReadLength;
}
void GPUSA::unloadReads() {
	/*release read hash*/
	cudaUnbindTexture(_texHash);
	myCheckCudaError;

	cudaFree(_devHash);
	myCheckCudaError;

	/*release reads*/
	cudaUnbindTexture(_texPacReads);
	myCheckCudaError;

	cudaFree(_devReads);
	myCheckCudaError;
}


#include "GPUMemEngine.h"
#include "GPUBWT.h"
#include "SeqFileParser.h"
#include <unistd.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#ifndef TMP_FILE_BASE
#define TMP_FILE_BASE "."
#endif

GPUMemEngine::GPUMemEngine(Options* options, Genome* rgenome, SAM* sam) :
		MemEngine(options, rgenome, sam) {

	/*create engines*/
	_gpuSA = new GPUSA(_options);
	_gpuSeeds = new GPUSeeds(_options);
	_gpuInfo = GPUInfo::getGPUInfo();
	_gpuReadOccs = new uint32_t[MAX_READS_PER_BATCH];
	if (!_gpuReadOccs) {
		Utils::exit("Memory allocation failed at line %d in file %s\n",
				__LINE__, __FILE__);
	}

	/*get file name*/
	char buffer[1024];
	int pid = getpid();
	sprintf(buffer, "%s/tmp#cushaw2#%d.sa", TMP_FILE_BASE, pid);
	_saFileName = buffer;

	sprintf(buffer, "%s/tmp#cushaw2#%d.seed", TMP_FILE_BASE, pid);
	_seedFileName = buffer;

	_membership = new Bitmap(MAX_READS_PER_BATCH);
	_seedMembership = new Bitmap(MAX_READS_PER_BATCH);
	_gpuMapQuals = new uint8_t[MAX_READS_PER_BATCH];
	if (!_gpuMapQuals) {
		Utils::exit("Memory allocation failed at line %d in file %s\n",
				__LINE__, __FILE__);
	}
	_paired = _options->isPaired();
	_unique = options->outOnlyUnique();
}

GPUMemEngine::~GPUMemEngine() {
	if (_gpuSA) {
		delete _gpuSA;
	}
	if (_gpuSeeds) {
		delete _gpuSeeds;
	}
	if (_gpuReadOccs) {
		delete[] _gpuReadOccs;
	}
	if (_membership) {
		delete _membership;
	}
	if (_seedMembership) {
		delete _seedMembership;
	}
	if (_gpuMapQuals) {
		delete[] _gpuMapQuals;
	}

	/*delete files*/
	unlink(_saFileName.c_str());
	unlink(_seedFileName.c_str());
}
/*align a batch of reads*/
size_t GPUMemEngine::run(Sequence* sequences, size_t numSequences,
		size_t maxNumSeedsPerBatch) {
	size_t numAligned;

	/*load the sequences into device memory*/
	_maxReadLength = _gpuSA->loadReads(sequences, numSequences);

	/*calculate the suffix array intervals*/
	_membership->reset();
	_calcSAIntervals(numSequences, _membership);

	/*select top hits*/
	//Utils::log("select top hits for each read\n");
	_seedMembership->reset();
	_selectTopHits(sequences, numSequences, _membership, _seedMembership,
			maxNumSeedsPerBatch);

	/*Get the final alignment*/
	//Utils::log("compute final alignments: %ld\n", numQualifiedSeeds);
	numAligned = _align(sequences, numSequences, _seedMembership, _gpuMapQuals,
			maxNumSeedsPerBatch);

	/*unload the sequences*/
	_gpuSA->unloadReads();

	return numAligned;
}
void GPUMemEngine::initialize(int32_t gpuID) {

	/*set GPU device*/
	_gpuInfo->setDevice(gpuID);

	/*load GPU resources*/
	_loadBWT();
	_gpuSeeds->loadGenome(_rgenome);
	_gpuSeeds->loadSuffixArray(_rsa);

	/*configure kernels*/
	_gpuSA->configKernels();

	_gpuSeeds->configKernels();

	/*load parameterss*/
	_gpuSA->loadParams(_options->getMaxSeedOcc(), _options->getMinIdentity(),
			_options->getMinRatio());
	_gpuSeeds->loadParameters(_options->getMatch(), _options->getMismatch(),
			_options->getGapOpen(), _options->getGapExtend());
}
void GPUMemEngine::finalize() {

	/*unload BWT*/
	_unloadBWT();

	/*unload suffix array*/
	_gpuSeeds->unloadSuffixArray();

	/*unload reference genome*/
	_gpuSeeds->unloadGenome();
}
void GPUMemEngine::_calcSAIntervals(size_t numSequences, Bitmap* membership) {
	uint32_t maxNumSAs;
	/*create a stream*/
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	/*calculate the suffix array intervals for the read batch*/
	maxNumSAs = 2 * (_maxReadLength - _options->getLowerMinSeedSize() + 2); /*for two strands*/
	_gpuSA->getSAIntervals(maxNumSAs, numSequences, 64, stream);

	/*save all SAs to an intermediate file*/
	_gpuSA->save(membership, _saFileName);

	/*destroy the stream*/
	cudaStreamDestroy(stream);
}
void GPUMemEngine::_selectTopHits(Sequence* sequences, size_t numSequences,
		Bitmap *membership, Bitmap *seedMembership,
		size_t maxNumSeedsPerBatch) {
	/*calculate the maximal number of indices*/
	thrust::host_vector<uint2> seeds;
	thrust::host_vector<uint32_t> alignScores;

	/*reserve memory*/
	seeds.reserve(maxNumSeedsPerBatch);
	alignScores.reserve(maxNumSeedsPerBatch);

	/*open the file*/
	FILE *safile = fopen(_saFileName.c_str(), "rb");
	if (!safile) {
		Utils::exit("Failed to open file %s at line %d in file %s\n",
				_saFileName.c_str(), __LINE__, __FILE__);
	}
	FILE *seedfile = fopen(_seedFileName.c_str(), "wb");
	if (!seedfile) {
		Utils::exit("Failed to open file %s at line %d in file %s\n",
				_seedFileName.c_str(), __LINE__, __FILE__);
	}

	/*create a stream*/
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	/*reset the engine*/
	_gpuSeeds->reset(membership, safile, maxNumSeedsPerBatch, numSequences);

	/*find the seeds*/
	size_t numSeeds = 0;
	thrust::device_vector<uint2> devSeeds;
	thrust::device_vector<uint32_t> devAlignScores;

	/*clear mapping quality scores*/
	do {
		/*read suffix arary intervals from the file*/
		_gpuSeeds->readSAindices(seeds, alignScores, _gpuReadOccs);

		numSeeds = seeds.size(); /*get the number of seeds*/
		//Utils::log("numSeeds: %d seqBaseIndex %ld numSequences %ld\n", numSeeds, _gpuSeeds->getSeqBaseIndex(), numSequences);
		if (numSeeds == 0) {
			break;
		}

		/*calculate the mapping positions on the genome*/
		/*initialize the device vector*/
		devSeeds.resize(numSeeds);
		thrust::copy(seeds.begin(), seeds.end(), devSeeds.begin());
		_gpuSeeds->calcMapPositions(thrust::raw_pointer_cast(devSeeds.data()),
				numSeeds, 64, stream);

		/*perform SW algorithm to obtain the optimal local alignment scores*/
		devAlignScores.resize(numSeeds);/*initialize the device vector*/
		thrust::copy(alignScores.begin(), alignScores.end(),
				devAlignScores.begin());
		_gpuSeeds->calcAlignScores(thrust::raw_pointer_cast(devSeeds.data()),
				thrust::raw_pointer_cast(devAlignScores.data()), numSeeds, 64,
				stream);

		/*sort and select the best seeds*/
		thrust::sort_by_key(devAlignScores.begin(), devAlignScores.end(),
				devSeeds.begin(), thrust::greater<uint32_t>());

		/*copy the sorted list to the host*/
		thrust::copy(devSeeds.begin(), devSeeds.end(), seeds.begin());
		thrust::copy(devAlignScores.begin(), devAlignScores.end(),
				alignScores.begin());

		/*save all seeds that is not less than the minimal score threshold*/
		_gpuSeeds->save(sequences, seeds.data(), alignScores.data(),
				_gpuReadOccs, _gpuMapQuals, numSeeds, seedMembership, seedfile,
				_minAlignScore);
	} while (1);

	/*release resources*/
	devSeeds.clear();
	devAlignScores.clear();

	/*close the file*/
	fclose(safile);
	fclose(seedfile);

	/*destroy stream*/
	cudaStreamDestroy(stream);
}
/*trace-back to find the alignments*/
size_t GPUMemEngine::_align(Sequence* sequences, size_t numSequences,
		Bitmap* membership, uint8_t* mapQuals, size_t maxNumSeedsPerBatch) {
	/*calculate the maximal number of indices*/
	thrust::host_vector<uint2> seeds;
	thrust::host_vector<uint32_t> alignScores;
	thrust::host_vector<uint16_t> cigars;
	thrust::host_vector<uint8_t> numCigars;
	thrust::host_vector<uint2> aligns;
	thrust::host_vector<float> basePortions;

	/*reserve memory*/
	seeds.reserve(maxNumSeedsPerBatch);
	alignScores.reserve(maxNumSeedsPerBatch);

	/*open the file*/
	FILE *seedfile = fopen(_seedFileName.c_str(), "rb");
	if (!seedfile) {
		Utils::exit("Failed to open file %s at line %d in file %s\n",
				_seedFileName.c_str(), __LINE__, __FILE__);
	}

	/*create a stream*/
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	/*reset the engine*/
	_gpuSeeds->reset(membership, seedfile, maxNumSeedsPerBatch, numSequences);

	/*find the seeds*/
	size_t numSeeds = 0;
	size_t numAligned = 0;
	uint32_t devCigarWidth;
	thrust::device_vector<uint2> devSeeds;
	thrust::device_vector<uint32_t> devAlignScores;
	thrust::device_vector<uint16_t> devCigars;
	thrust::device_vector<uint8_t> devNumCigars;
	thrust::device_vector<uint16_t> transposedDevCigars;
	thrust::device_vector<uint2> devAligns;
	thrust::device_vector<float> devBasePortions;

	/*main loop*/
	do {
		/*read seeds from the file*/
		_gpuSeeds->readQualifiedSeeds(seeds, alignScores, _gpuReadOccs);

		/*get the number of seeds for this round*/
		numSeeds = seeds.size();
		//Utils::log("number of seeds: %ld\n", numSeeds);
		if (numSeeds == 0) {
			break;
		}

		/*allocate memory on the GPU*/
		devSeeds.resize(numSeeds);
		thrust::copy(seeds.begin(), seeds.end(), devSeeds.begin());

		devAlignScores.resize(numSeeds);/*initialize the device vector*/
		thrust::copy(alignScores.begin(), alignScores.end(),
				devAlignScores.begin());

		/*Width and height of the array on the device*/
		devCigarWidth = (numSeeds + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
		devCigars.resize(devCigarWidth * MAX_NUM_CIGAR_ENTRIES);
		transposedDevCigars.resize(devCigarWidth * MAX_NUM_CIGAR_ENTRIES);
		devNumCigars.resize(devCigarWidth);
		devAligns.resize(devCigarWidth);
		devBasePortions.resize(devCigarWidth);

		/*perform the Smith-Waterman algorithm*/
		_gpuSeeds->calcAlign(thrust::raw_pointer_cast(devSeeds.data()),
				thrust::raw_pointer_cast(devAlignScores.data()),
				thrust::raw_pointer_cast(transposedDevCigars.data()),
				devCigarWidth, MAX_NUM_CIGAR_ENTRIES,
				thrust::raw_pointer_cast(devNumCigars.data()),
				thrust::raw_pointer_cast(devAligns.data()),
				thrust::raw_pointer_cast(devBasePortions.data()), numSeeds, 64,
				stream);

		/*transpose the cigar array*/
		GPUUtils::transpose(
				thrust::raw_pointer_cast(transposedDevCigars.data()),
				thrust::raw_pointer_cast(devCigars.data()), devCigarWidth,
				MAX_NUM_CIGAR_ENTRIES, stream);

		/*copy the transposed data*/
		cigars.resize(devCigarWidth * MAX_NUM_CIGAR_ENTRIES);
		numCigars.resize(devCigarWidth);
		aligns.resize(devCigarWidth);
		basePortions.resize(devCigarWidth);

		thrust::copy(devCigars.begin(), devCigars.end(), cigars.begin());
		thrust::copy(devNumCigars.begin(), devNumCigars.end(),
				numCigars.begin());
		thrust::copy(devAligns.begin(), devAligns.end(), aligns.begin());
		thrust::copy(devBasePortions.begin(), devBasePortions.end(),
				basePortions.begin());

		/*output the alignment*/
		numAligned +=
				_paired ?
						_printPaired(sequences, numSequences,
								alignScores.data(), cigars.data(),
								numCigars.data(), aligns.data(), mapQuals,
								numSeeds, _gpuReadOccs, basePortions.data()) :
						_print(sequences, numSequences, alignScores.data(),
								cigars.data(), numCigars.data(), aligns.data(),
								mapQuals, numSeeds, _gpuReadOccs,
								basePortions.data());

	} while (1);

	/*release resources*/
	devSeeds.clear();
	devAlignScores.clear();
	devCigars.clear();
	devNumCigars.clear();
	transposedDevCigars.clear();
	devAligns.clear();
	devBasePortions.clear();

	/*close the file*/
	fclose(seedfile);

	/*print out unaligned reads*/
	for (size_t i = 0; i < numSequences; ++i) {
		if (_gpuReadOccs[i] == 0) {
			_options->lock();
			_sam->print(sequences[i]);
			_options->unlock();
		}
	}

	/*destroy stream*/
	cudaStreamDestroy(stream);

	return numAligned;
}

size_t GPUMemEngine::_print(Sequence* sequences, size_t numSequences,
		uint32_t *alignScores, uint16_t* cigars, uint8_t *numCigars,
		uint2* aligns, uint8_t* mapQuals, size_t numSeeds, uint32_t *readOccs,
		float* basePortions) {

	uint32_t qual;
	uint32_t readIndex;
	size_t numAligned = 0;
	uint32_t numOccs;
	bool aligned;

	for (uint32_t row = 0; row < numSeeds;) {

		/*get the read index*/
		readIndex = GET_READ_INDEX(*alignScores);
		numOccs = readOccs[readIndex];

		/*check all alignment instances of the read*/
		aligned = false;
		for (uint32_t start = row, end = row + numOccs; start < end; ++start) {
			if (*numCigars > 0) {
				aligned = true;

				/*generate a SAM line*/
				qual = mapQuals[readIndex];
				qual *= *basePortions;
				if (_unique == false || qual > 0) {
					_options->lock();
					_sam->print(sequences[readIndex], cigars, *numCigars,
							aligns->x >> 1, aligns->y, aligns->x & 1, qual);
					_options->unlock();
				}
			}

			/*move to the next row*/
			++numCigars;
			++alignScores;
			++aligns;
			++basePortions;
			cigars += MAX_NUM_CIGAR_ENTRIES;
		}
		/*check if it is aligned*/
		if (aligned) {
			++numAligned;
		} else {
			/*generate a SAM line*/
			_options->lock();
			_sam->print(sequences[readIndex]);
			_options->unlock();
		}

		/*updates the row*/
		row += numOccs;
	}
	return numAligned;
}

size_t GPUMemEngine::_printPaired(Sequence* sequences, size_t numSequences,
		uint32_t *alignScores, uint16_t* cigars, uint8_t *numCigars,
		uint2* aligns, uint8_t* mapQuals, size_t numSeeds, uint32_t* readOccs,
		float* basePortions) {

	uint32_t leftReadIndex, rightReadIndex;
	size_t numAligned = 0;
	uint8_t *leftNumCigars, *leftNumCigarsEnd, *rightNumCigars,
			*rightNumCigarsEnd;
	uint2 *leftAlign, *rightAlign;
	uint32_t leftStrand, rightStrand;
	uint32_t leftGenomeIndex, rightGenomeIndex;
	uint16_t *leftCigars, *rightCigars;
	uint32_t leftOccs, numOccs;
	int64_t leftMapPos, rightMapPos;
	uint32_t leftLength, rightLength;
	float* leftBasePortions, *rightBasePortions;
	uint32_t leftMapQual, rightMapQual;
	bool isPaired, good, good2;
	int insertSize;

	for (uint32_t row = 0; row < numSeeds; row += numOccs, alignScores +=
			numOccs, aligns += numOccs, numCigars += numOccs, basePortions +=
			numOccs, cigars += numOccs * MAX_NUM_CIGAR_ENTRIES) {

		/*get the read indices for the current sequence pair*/
		leftReadIndex = GET_READ_INDEX(*alignScores);
		if (leftReadIndex & 1) {
			rightReadIndex = leftReadIndex;
			--leftReadIndex; /*this reads does not have any qualified seed*/
		} else {
			rightReadIndex = leftReadIndex + 1;
		}

		/*get the number of top seeds for each end*/
		leftOccs = readOccs[leftReadIndex];
		numOccs = leftOccs + readOccs[rightReadIndex];

		/*get the sequence length*/
		leftLength = sequences[leftReadIndex]._length;
		rightLength = sequences[rightReadIndex]._length;

		/*for the left end*/
		leftNumCigarsEnd = numCigars + leftOccs;
		rightNumCigarsEnd = numCigars + numOccs;

		/*get mapping quality scores*/
		leftMapQual = mapQuals[leftReadIndex];
		rightMapQual = mapQuals[rightReadIndex];

		/*start the core loop*/
		isPaired = false;
		leftAlign = aligns;
		leftCigars = cigars;
		leftBasePortions = basePortions;
		for (leftNumCigars = numCigars; leftNumCigars < leftNumCigarsEnd;
				++leftNumCigars, ++leftAlign, ++leftBasePortions, leftCigars +=
						MAX_NUM_CIGAR_ENTRIES) {

			/*check the availability of the alignment*/
			//Utils::log("leftNumCigars: %d\n", *leftNumCigars);
			if (*leftNumCigars == 0) {
				continue;
			}

			/*get the alignment strand*/
			leftGenomeIndex = leftAlign->x >> 1;
			leftStrand = leftAlign->x & 1;
			leftMapPos = leftAlign->y;

			/*for the right end*/
			rightAlign = aligns + leftOccs;
			rightCigars = cigars + leftOccs * MAX_NUM_CIGAR_ENTRIES;
			rightBasePortions = basePortions + leftOccs;
			for (rightNumCigars = numCigars + leftOccs;
					rightNumCigars < rightNumCigarsEnd;
					++rightNumCigars, ++rightAlign, ++rightBasePortions, rightCigars +=
							MAX_NUM_CIGAR_ENTRIES) {

				/*check the availability of the alignment*/
				//Utils::log("rightNumCigars: %d\n", *rightNumCigars);
				if (*rightNumCigars == 0) {
					continue;
				}

				/*get the alignment strand*/
				rightGenomeIndex = rightAlign->x >> 1;
				rightStrand = rightAlign->x & 1;
				rightMapPos = rightAlign->y;

				/*check the strand*/
				//Utils::log("strand: %d %d\n", leftStrand, rightStrand);
				if (leftStrand == rightStrand
						|| leftGenomeIndex != rightGenomeIndex) {
					continue;
				}

				/*check the insert size contraint*/
				if (labs(leftMapPos - rightMapPos) <= _maxDistance) {
					isPaired = true;
					if (leftStrand == 0) {
						insertSize = leftMapPos - rightMapPos
								- leftLength;
					} else {
						insertSize = leftMapPos + leftLength
								- rightMapPos;
					}

					_options->lock();
					_sam->print(sequences[leftReadIndex], leftCigars,
							*leftNumCigars, leftGenomeIndex, leftMapPos,
							leftStrand, leftMapQual * (*leftBasePortions),
							rightGenomeIndex, rightMapPos, insertSize, SAM_FR1);

					_sam->print(sequences[rightReadIndex], rightCigars,
							*rightNumCigars, rightGenomeIndex, rightMapPos,
							rightStrand, rightMapQual * (*rightBasePortions),
							leftGenomeIndex, leftMapPos, -insertSize, SAM_FR2);
					_options->unlock();
					break;
				}
			}

			/*if the reads have been paired*/
			if (isPaired) {
				break;
			}
		}
		if (isPaired) {
			numAligned += 2;
			continue;
		}

		/*get the best alignment of the left end*/
		good = false;
		leftCigars = cigars;
		leftAlign = aligns;
		leftBasePortions = basePortions;
		leftNumCigars = numCigars;
		if (leftOccs > 0 && *leftNumCigars) {
			leftMapQual *= *leftBasePortions;
			good = leftMapQual >= _mapQualReliable;
		}

		good2 = false;
		rightCigars = cigars + leftOccs * MAX_NUM_CIGAR_ENTRIES;
		rightAlign = aligns + leftOccs;
		rightBasePortions = basePortions + leftOccs;
		rightNumCigars = numCigars + leftOccs;
		if (numOccs > leftOccs && *rightNumCigars) {
			rightMapQual *= *rightBasePortions;
			good2 = rightMapQual >= _mapQualReliable;
		}

		/*Utils::log("good %d %d good2 %d %d %d\n", good, *leftNumCigars, good2,
		 *rightNumCigars, _mapQualReliable);*/

		int64_t genomeStart1, genomeEnd1, genomeStart2, genomeEnd2;
		if (_options->rescueMate()) {
			if (good) { /*the left end has a reliable alignment*/

				/*calcualte the mapping region of the mate read*/
				leftGenomeIndex = leftAlign->x >> 1;
				leftStrand = leftAlign->x & 1;
				leftMapPos = leftAlign->y;
				_getMateRegion(genomeStart1, genomeEnd1,
						_rgenome->getGenomeOffset(leftGenomeIndex) + leftMapPos
								- 1, leftStrand, leftLength, rightLength);

				/*refine the region*/
				_rgenome->refineRegionRange(leftGenomeIndex, genomeStart1,
						genomeEnd1);

				/*perform the alignment*/
				Mapping* mateMapping = _getAlignment(_aligner,
						sequences[rightReadIndex], 1 - leftStrand,
						_options->getNumErrors(rightLength) * 2,
						leftGenomeIndex, genomeStart1,
						genomeEnd1 - genomeStart1 + 1, _minRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * leftMapQual);

				/*succeeded in finding an alignment*/
				if (mateMapping) {
					isPaired = true;
					if (leftStrand == 0) {
						insertSize = leftMapPos - rightMapPos
								- leftLength;
					} else {
						insertSize = leftMapPos + leftLength
								- rightMapPos;
					}
					_options->lock();
					_sam->print(sequences[leftReadIndex], leftCigars,
							*leftNumCigars, leftGenomeIndex, leftMapPos,
							leftStrand, leftMapQual, mateMapping->_genomeIndex,
							mateMapping->_gposition, insertSize, SAM_FR1);
					_sam->print(sequences[rightReadIndex], *mateMapping,
							leftGenomeIndex, leftMapPos, -insertSize, SAM_FR2);
					_options->unlock();

					delete mateMapping;

					/*indicating that the read is rescued*/
					readOccs[rightReadIndex] = 1;
					numAligned += 2;
					continue;
				}
			}
			if (good2) { /*the right end has a reliable alignment*/
				rightGenomeIndex = rightAlign->x >> 1;
				rightStrand = rightAlign->x & 1;
				rightMapPos = rightAlign->y;
				/*calcualte the mapping region of the mate read*/
				_getMateRegion(genomeStart2, genomeEnd2,
						_rgenome->getGenomeOffset(rightGenomeIndex)
								+ rightMapPos - 1, rightStrand, rightLength,
						leftLength);

				/*refine the region*/
				_rgenome->refineRegionRange(rightGenomeIndex, genomeStart2,
						genomeEnd2);

				/*perform the alignment*/
				Mapping* mateMapping = _getAlignment(_aligner,
						sequences[leftReadIndex], 1 - rightStrand,
						_options->getNumErrors(leftLength) * 2,
						rightGenomeIndex, genomeStart2,
						genomeEnd2 - genomeStart2 + 1, _minRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * rightMapQual);

				/*succeeded in finding an alignment*/
				if (mateMapping) {
					isPaired = true;
					if (rightStrand == 0) {
						insertSize = rightMapPos - leftMapPos
								- rightLength;
					} else {
						insertSize = rightMapPos + rightLength
								- leftMapPos;
					}
					_options->lock();
					_sam->print(sequences[leftReadIndex], *mateMapping,
							rightGenomeIndex, rightMapPos, 0-insertSize,
							SAM_FR1);
					_sam->print(sequences[rightReadIndex], rightCigars,
							*rightNumCigars, rightGenomeIndex, rightMapPos,
							rightStrand, rightMapQual,
							mateMapping->_genomeIndex, mateMapping->_gposition,
							insertSize, SAM_FR2);
					_options->unlock();

					delete mateMapping;

					/*indicating that the read is rescued*/
					readOccs[leftReadIndex] = 1;
					numAligned += 2;

					continue;
				}
			}

#ifdef HAVE_TWICE_RESCUE
			/*twice rescuing*/
			if (_options->rescueTwice()) {
				if(good) {
					/*perform the alignment*/
					Mapping* mateMapping = _getAlignment(_swaligner,
							sequences[rightReadIndex], 1 - leftStrand,
							_options->getNumErrors(rightLength) * 2,
							leftGenomeIndex, genomeStart1,
							genomeEnd1 - genomeStart1 + 1, _minRatio, _minIdentity,
							SW_MAP_QUALITY_SCORE * leftMapQual);

					/*succeeded in finding an alignment*/
					if (mateMapping) {
						isPaired = true;
						if (leftStrand == 0) {
							insertSize = leftMapPos - rightMapPos
							- leftLength;
						} else {
							insertSize = leftMapPos + leftLength
							- rightMapPos;
						}
						_options->lock();
						_sam->print(sequences[leftReadIndex], leftCigars,
								*leftNumCigars, leftGenomeIndex, leftMapPos,
								leftStrand, leftMapQual, mateMapping->_genomeIndex,
								mateMapping->_gposition, insertSize, SAM_FR1);
						_sam->print(sequences[rightReadIndex], *mateMapping,
								leftGenomeIndex, leftMapPos, 0-insertSize, SAM_FR2);
						_options->unlock();

						delete mateMapping;

						/*indicating that the read is rescued*/
						readOccs[rightReadIndex] = 1;
						numAligned += 2;
						continue;
					}
				}
				if (good2) {
					/*perform the alignment*/
					Mapping* mateMapping = _getAlignment(_swaligner,
							sequences[leftReadIndex], 1 - rightStrand,
							_options->getNumErrors(leftLength) * 2,
							rightGenomeIndex, genomeStart2,
							genomeEnd2 - genomeStart2 + 1, _minRatio, _minIdentity,
							SW_MAP_QUALITY_SCORE * rightMapQual);

					/*succeeded in finding an alignment*/
					if (mateMapping) {
						isPaired = true;
						if (rightStrand == 0) {
							insertSize = rightMapPos - leftMapPos
							- rightLength;
						} else {
							insertSize = rightMapPos + rightLength
							- leftMapPos;
						}
						_options->lock();
						_sam->print(sequences[leftReadIndex], *mateMapping,
								rightGenomeIndex, rightMapPos, 0-insertSize,
								SAM_FR1);
						_sam->print(sequences[rightReadIndex], rightCigars,
								*rightNumCigars, rightGenomeIndex, rightMapPos,
								rightStrand, rightMapQual,
								mateMapping->_genomeIndex, mateMapping->_gposition,
								insertSize, SAM_FR2);
						_options->unlock();

						delete mateMapping;

						/*indicating that the read is rescued*/
						readOccs[leftReadIndex] = 1;
						numAligned += 2;

						continue;
					}
				}
			}
#endif
		}

		/*if not paired, output in single-end mode*/
		if (!isPaired) {
			_options->lock();
			if (leftOccs > 0 && *leftNumCigars > 0) {
				_sam->print(sequences[leftReadIndex], leftCigars,
						*leftNumCigars, leftAlign->x >> 1, leftAlign->y,
						leftAlign->x & 1, leftMapQual, SAM_FR1);
				++numAligned;

			} else {
				/*indicating that the read is unaligned*/
				readOccs[leftReadIndex] = 0;
			}
			if (numOccs > leftOccs && *rightNumCigars > 0) {
				_sam->print(sequences[rightReadIndex], rightCigars,
						*rightNumCigars, rightAlign->x >> 1, rightAlign->y,
						rightAlign->x & 1, rightMapQual, SAM_FR2);
				++numAligned;
			} else {
				/*indicating that the read is unaligned*/
				readOccs[rightReadIndex] = 0;
			}
			_options->unlock();
		}
	}

	return numAligned;
}


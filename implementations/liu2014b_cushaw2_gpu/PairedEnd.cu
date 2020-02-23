/*
 * PairedEnd.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: yongchao
 */

#include "PairedEnd.h"
#include "SeqFileParser.h"
#include "Options.h"
#include "GPUMemEngine.h"
#include <omp.h>

PairedEnd::PairedEnd(Options* options, Genome * genome, SAM* sam) {
	_options = options;
	_genome = genome;
	_sam = sam;

	/*get the number of threads*/
	_numThreads = _options->getNumThreads();
	_numCPUs = _options->getNumCPUs();
	_numGPUs = _options->getNumGPUs();

	/*create parameters for threads*/
	if (_options->estimateInsertSize()) {
		_threads.resize(max(_numCPUs, 1));
	} else {
		_threads.resize(_numCPUs); /*at least on thread reserved for insert-size estimation*/
	}
	_engines.resize(_threads.size());
	for (size_t tid = 0; tid < _threads.size(); ++tid) {
		_engines[tid] = new MemEngine(_options, _genome, _sam);
		_threads[tid] = new ThreadParams(tid, _options, _sam, _engines[tid],
				this);
	}

	/*create parameters for GPU threads*/
	_gpuThreads.resize(_numGPUs);
	_gpuEngines.resize(_numGPUs);
	for (size_t tid = 0; tid < _gpuThreads.size(); ++tid) {
		_gpuEngines[tid] = new GPUMemEngine(_options, _genome, _sam);

		/*initialize GPU*/
		_gpuEngines[tid]->initialize(options->getGPUIndex());

		_gpuThreads[tid] = new ThreadParams(tid, _options, _sam,
				_gpuEngines[tid], this);
	}

	/*create a global mutex*/
	pthread_mutex_init(&_mutex, NULL);

#ifdef FREE_UNUSED_GENOME_MEMORY
	/*If CPU threads are not used, free more host memory*/
	uint32_t flags = _threads.size() == 0 ? 1 : 0; /*free BWT*/
	flags |= _options->rescueMate() == false ? 2 : 0; /*free Genome*/
	freeGenomeMemory(flags);
#endif
}

PairedEnd::~PairedEnd() {
	/*destory the global mutex*/
	pthread_mutex_destroy(&_mutex);

	for (size_t i = 0; i < _threads.size(); ++i) {
		delete _threads[i];
	}
	_threads.clear();

	for (size_t i = 0; i < _gpuThreads.size(); ++i) {
		delete _gpuThreads[i];
	}
	_gpuThreads.clear();

	for (size_t i = 0; i < _engines.size(); ++i) {
		delete _engines[i];
	}
	_engines.clear();

	GPUInfo* gpuInfo = GPUInfo::getGPUInfo();
	for (size_t i = 0; i < _gpuEngines.size(); ++i) {
		/*set device*/
		gpuInfo->setDevice(_options->getGPUIndex());

		/*release GPU*/
		_gpuEngines[i]->finalize();
		delete _gpuEngines[i];
	}
	_gpuEngines.clear();
}
void PairedEnd::estimateInsertSize(int minAlignedPairs, int maxReadBatchSize,
		int numReadBatchs) {
	size_t nreads;
	int mapQualReliable = _options->getMapQualReliable();
	int numReadPairs = 0;
	SeqFileParser *parser1, *parser2;
	Sequence *seqs1, *seqs2;
	vector<int> globalInsertSizes;
	int* localInsertSizes;
	bool viaFifo = _options->isViaFifo();
	double stime = Utils::getSysTime();

	Utils::log("Start estimating insert size using the top 0x%x read pairs\n",
			maxReadBatchSize * numReadBatchs);
	/*allocate memory*/
	seqs1 = new Sequence[maxReadBatchSize];
	if (!seqs1) {
		Utils::exit("Memory allocation failed in line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	seqs2 = new Sequence[maxReadBatchSize];
	if (!seqs2) {
		Utils::exit("Memory allocation failed in line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	localInsertSizes = new int[maxReadBatchSize];
	if (!localInsertSizes) {
		Utils::exit("Memory allocation failed in line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	/*reserved space for global insert sizes*/
	globalInsertSizes.reserve(maxReadBatchSize * numReadBatchs);

	/*set the number of threads for OpenMP runtime*/
	omp_set_num_threads(_threads.size());

	/*for paired-end alignment*/
	bool done = false;
	vector<pair<string, int> > &inputs = _options->getInputFileList();
	for (size_t file = 0; file < inputs.size(); file += 2) {

		/*open the file*/
		parser1 = new SeqFileParser(inputs[file].first.c_str(), false,
				inputs[file].second);
		if (viaFifo) {
			/*read a batch of paired-end reads*/
			for (nreads = 0; nreads < maxReadBatchSize; ++nreads) {
				if (!parser1->getSeq(seqs1[nreads], seqs2[nreads])) {
					break;
				}
			}
		} else {
			/*open the file for the right sequences*/
			parser2 = new SeqFileParser(inputs[file + 1].first.c_str(), false,
					inputs[file].second);

			/*read a batch of paired-end reads*/
			if ((nreads = parser1->getSeqLockFree(seqs1, maxReadBatchSize))
					== 0) {
				Utils::log("Empty input file\n");
			}
			if (parser2->getSeqLockFree(seqs2, maxReadBatchSize) != nreads) {
				Utils::exit("The two files have different number of reads\n");
			}
		}

		/*start the main loop*/
		do {
			size_t index;
			/*get the single-end alignments*/
#pragma omp parallel for private(index) default(shared) schedule(dynamic, 1)
			for (index = 0; index < nreads; ++index) {

				/*get the thread ID*/
				int tid = omp_get_thread_num();

				/*get the engine for the thread*/
				MemEngine *engine = (MemEngine*) _threads[tid]->_engine;

				/*perform alignment*/
				Mapping *mapping1, *mapping2;
				engine->align(seqs1[index], mapping1);
				engine->align(seqs2[index], mapping2);

				/*calcualte the insert size*/
				int insertSize = -1; /*dummy insert size*/
				if (mapping1 && mapping2) {
					if (mapping1->_strand != mapping2->_strand
							&& mapping1->_genomeIndex == mapping2->_genomeIndex
							&& mapping1->_mapQual >= mapQualReliable
							&& mapping2->_mapQual >= mapQualReliable) {
						if (mapping1->_strand == 0) {
							insertSize = ((long) mapping1->_position)
									- mapping2->_position
									- seqs2[index]._length;
						} else {
							insertSize = ((long) mapping1->_position)
									- mapping2->_position
									+ seqs1[index]._length;
						}
						if (insertSize < 0)
							insertSize = -insertSize;
					}
				}
				/*save the insert size for the current read pair*/
				localInsertSizes[index] = insertSize;
				/*release the mapping*/
				if (mapping1)
					delete mapping1;
				if (mapping2)
					delete mapping2;
			}

			/*merge all local insert sizes*/
			for (size_t i = 0; i < nreads; ++i) {
				if (localInsertSizes[i] > 0) {
					globalInsertSizes.push_back(localInsertSizes[i]);
				}
			}

			/*statistical information*/
			numReadPairs += nreads;
			if (numReadPairs >= maxReadBatchSize * numReadBatchs) {
				Utils::log("#read pairs read from the input: %d\n",
						numReadPairs);
				done = true;
				break;
			}

			/*re-load a batch of read pairs*/
			if (viaFifo) {
				for (nreads = 0; nreads < maxReadBatchSize; ++nreads) {
					if (!parser1->getSeq(seqs1[nreads], seqs2[nreads])) {
						done = true;
						break;
					}
				}
				if (done) {
					break;
				}
			} else {
				if ((nreads = parser1->getSeqLockFree(seqs1, maxReadBatchSize))
						== 0) {
					break;
				}
				if (parser2->getSeqLockFree(seqs2, maxReadBatchSize)
						!= nreads) {
					Utils::exit(
							"The two files have different number of reads\n");
				}
			}
		} while (1);

		/*release the file parser*/
		delete parser1;
		if (!viaFifo) {
			delete parser2;
		}

		/*check if sufficient aligned pairs have got*/
		if (done) {
			break;
		}
	}
	/*release resources*/
	delete[] localInsertSizes;
	delete[] seqs2;
	delete[] seqs1;

	/*check the number of insert sizes*/
	int numInsertSizes = globalInsertSizes.size();
	if (numInsertSizes < minAlignedPairs) {
		Utils::exit(
				"#qualified reads pairs (%d) is less than %d and please specify the insert size through parameters\n",
				numInsertSizes, minAlignedPairs);
	}

	/*sort the insert sizes*/
	sort(globalInsertSizes.begin(), globalInsertSizes.end());

	/*get the inset size for percentile 25, 50 and 75*/
	int* insertSizes = &globalInsertSizes[0];
	int p25 = insertSizes[(int) (numInsertSizes * 0.25 + 0.499)];
	int p50 = insertSizes[(int) (numInsertSizes * 0.5 + 0.499)];
	int p75 = insertSizes[(int) (numInsertSizes * 0.75 + 0.499)];

	/*estimate the mean using mean value instead of the median? FIXME*/
	double dvariance = 0;
	long mean, variance;
	int low, high, count;
	low = p25 - 2 * (p75 - p25);
	if (low < 0)
		low = 0;

	high = p75 + 2 * (p75 - p25);
	if (high > 5 * p50)
		high = 5 * p50;

	/*calculate the mean value*/
	mean = 0;
	count = 0;
	for (int i = 0; i < numInsertSizes; ++i) {
		if (insertSizes[i] > low && insertSizes[i] < high) {
			mean += insertSizes[i];
			++count;
		}
	}
	if (count > 0) {
		mean = (long) (static_cast<double>(mean) / count + 0.499);
	} else {
		Utils::exit(
				"Failed to estimate the insert size. Please specify this information through parameters\n");
	}

	/*calculate the variance*/
	variance = 0;
	count = 0;
	for (int i = 0; i < numInsertSizes; ++i) {
		if (insertSizes[i] > low && insertSizes[i] < high) {
			variance += (insertSizes[i] - mean) * (insertSizes[i] - mean);
			++count;
		}
	}
	if (count > 0) {
		dvariance = sqrt(static_cast<double>(variance) / count);
	}
	if (variance == 0)
		dvariance = 1;

	/*further limit the variance*/
	if (dvariance > 0.2 * mean) {
		dvariance = 0.2 * mean;
	}
	variance = static_cast<int>(dvariance + 0.499);

	/*set the insert size as well as the standard deviation*/
	_options->setInsertSize(mean);
	_options->setStdInsertSize(variance);
	Utils::log("Estimated insert size: %ld +/- %ld from %d effective samples\n",
			mean, variance, count);

	/*release resource*/
	globalInsertSizes.clear();

	/*update distance information*/
	for (size_t tid = 0; tid < _threads.size(); ++tid) {
		((MemEngine*) _threads[tid]->_engine)->updateDistance();
	}
	for (size_t tid = 0; tid < _gpuThreads.size(); ++tid) {
		((MemEngine*) _gpuThreads[tid]->_engine)->updateDistance();
	}

	double etime = Utils::getSysTime();
	Utils::log(
			"Finish estimating insert size (taken %f seconds using %d threads)\n",
			etime - stime, _threads.size());
}
void PairedEnd::execute() {
	SeqFileParser *parser1, *parser2 = NULL;
	vector<pthread_t> threadIDs;
	bool viaFifo = _options->isViaFifo();

	/*estimate the insert size?*/
	if (_options->estimateInsertSize()) {
		estimateInsertSize(100, INS_SIZE_EST_MULTIPLE,
				_options->getTopReadsEstIns() / INS_SIZE_EST_MULTIPLE);
	}
	/*initialize variables*/
	threadIDs.resize(_numThreads);

	/*for paired-end alignment*/
	vector<pair<string, int> > &inputs = _options->getInputFileList();
	for (size_t file = 0; file < inputs.size(); file += 2) {
		/*open the file for the left sequences*/
		parser1 = new SeqFileParser(inputs[file].first.c_str(), false,
				inputs[file].second);

		/*open the file for the right sequences*/
		if(!viaFifo){
			parser2 = new SeqFileParser(inputs[file + 1].first.c_str(), false,
				inputs[file].second);
		}

		/*create CPU threads*/
		size_t gid = 0;
		for (size_t tid = 0; tid < _numCPUs; ++tid) {
			/*set file parser*/
			_threads[tid]->setFileParser(parser1, parser2);

			/*create thread entities*/
			if (pthread_create(&threadIDs[gid++], NULL, _threadFunc,
					_threads[tid]) != 0) {
				Utils::exit("Thread creating failed\n");
			}
		}
		for (size_t tid = 0; tid < _numGPUs; ++tid) {

			/*set file parser*/
			_gpuThreads[tid]->setFileParser(parser1, parser2);

			/*create thread entities*/
			if (pthread_create(&threadIDs[gid++], NULL, _gpuThreadFunc,
					_gpuThreads[tid]) != 0) {
				Utils::exit("Thread creating failed\n");
			}
		}
		/*wait for the completion of all threads*/
		for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
			pthread_join(threadIDs[tid], NULL);
		}
		/*release the file parser*/
		delete parser1;
		if(!viaFifo){
			delete parser2;
		}
	}
	threadIDs.clear();
	SeqFileParser::finalize();

	/*report the alignment information*/
	size_t numReads = 0, numAligned = 0, numPaired = 0;
	for (size_t tid = 0; tid < _threads.size(); ++tid) {
		numReads += _threads[tid]->_numReads;
		numAligned += _threads[tid]->_numAligned;
		numPaired += _threads[tid]->_numPaired;
	}
	for (size_t tid = 0; tid < _gpuThreads.size(); ++tid) {
		numReads += _gpuThreads[tid]->_numReads;
		numAligned += _gpuThreads[tid]->_numAligned;
		numPaired += _gpuThreads[tid]->_numPaired;
	}

	Utils::log("#Reads aligned: %ld / %ld (%.2f%%)\n", numAligned, numReads,
			((double) (numAligned)) / numReads * 100);

#if 0
	Utils::log("#Reads paired: %ld / %ld (%.2f%%)\n", numPaired, numReads,
			((double) (numAligned)) / numReads * 100);
#endif
}
void* PairedEnd::_threadFunc(void* arg) {
	Sequence seq1, seq2;
	Mapping *mapping1, *mapping2;
	int64_t numReads = 0, numAligned = 0, numPaired = 0;
	ThreadParams *params = (ThreadParams*) arg;
	SAM* sam = params->_sam;
	Options* options = params->_options;
	SeqFileParser *parser1 = params->_parser1;
	SeqFileParser *parser2 = params->_parser2;
	PairedEnd* aligner = (PairedEnd*) params->_aligner;
	MemEngine *engine = (MemEngine*) params->_engine;
	double stime = Utils::getSysTime();
	double etime;
	bool viaFifo = options->isViaFifo();
	bool unique = options->outOnlyUnique();
	int multi = options->getMaxMultiAligns();
	vector<Mapping*> mapv1, mapv2;

	/*reserve space for the vectors*/
	mapv1.reserve(multi);
	mapv2.reserve(multi);
	bool paired;
	int flags1 = SAM_FPD | SAM_FR1; /*the read is initially paired*/
	int flags2 = SAM_FPD | SAM_FR2;

	while (1) {
		/*read a sequence pair*/
		aligner->lock();
		if (viaFifo) {
			if (!parser1->getSeqLockFree(seq1, seq2)) {
				aligner->unlock();
				break;
			}
		} else {
			if (!parser1->getSeqLockFree(seq1)) {
				aligner->unlock();
				break;
			}
			if (!parser2->getSeqLockFree(seq2)) {
				Utils::log(
						"The two files have different number of sequences\n");
				aligner->unlock();
				break;
			}
		}
		aligner->unlock();

		/*paired-end alignment by outputing multiple equivalent alignments*/
		if (multi > 1) {
			/*invoke the engine to get the paired-end alignments*/
			if ((paired = engine->align(seq1, seq2, mapv1, mapv2))) {
				numPaired++;
			}
			/*output the alignment*/
			options->lock();
			if (paired) { /*paired alignments found*/
				if (mapv1.size() != mapv2.size()) {
					Utils::exit(
							"Error occured for paired-end alignments in line %d in funciton %s\n",
							__LINE__, __FUNCTION__);
				}
				for (size_t i = 0; i < mapv1.size(); ++i) {
					mapping1 = mapv1[i];
					mapping2 = mapv2[i];
					if (unique == false || mapping1->_mapQual > 0
							|| mapping2->_mapQual > 0) {
						sam->print(seq1, *mapping1, *mapping2, SAM_FR1);
						sam->print(seq2, *mapping2, *mapping1, SAM_FR2);
					}
				}
			} else {
				if (mapv1.size() >= 2 || mapv2.size() >= 2) {
					Utils::exit(
							"Error occured for paired-end alignments in line %d in funciton %s\n",
							__LINE__, __FUNCTION__);
				}
				if (mapv1.size() > 0) {
					mapping1 = mapv1[0];
					if (unique == false || mapping1->_mapQual > 0) {
						sam->print(seq1, *mapping1, flags1);
					}
					sam->print(seq2, flags2);
				} else if (mapv2.size() > 0) {
					mapping2 = mapv2[0];
					sam->print(seq1, flags1);
					if (unique == false || mapping2->_mapQual > 0) {
						sam->print(seq2, *mapping2, flags2);
					}
				} else {
					sam->print(seq1, flags1);
					sam->print(seq2, flags2);
				}
			}
			options->unlock();

			/*statistical information*/
			numAligned += (mapv1.size() != 0) + (mapv2.size() != 0);

			/*release the resources*/
			for (size_t i = 0; i < mapv1.size(); ++i) {
				delete mapv1[i];
			}
			mapv1.clear();
			for (size_t i = 0; i < mapv2.size(); ++i) {
				delete mapv2[i];
			}
			mapv2.clear();
		} else {
			/*invoke the engine*/
			if (engine->align(seq1, seq2, mapping1, mapping2)) {
				numPaired++;
			}
			/*output the alignment*/
			options->lock();
			if (mapping1 && mapping2) {
				if (unique == false || mapping1->_mapQual > 0
						|| mapping2->_mapQual > 0) {
					sam->print(seq1, *mapping1, *mapping2, SAM_FR1);
					sam->print(seq2, *mapping2, *mapping1, SAM_FR2);
				}
			} else if (mapping1) {
				if (unique == false || mapping1->_mapQual > 0) {
					sam->print(seq1, *mapping1, flags1);
				}
				sam->print(seq2, flags2);
			} else if (mapping2) {
				sam->print(seq1, flags1);
				if (unique == false || mapping2->_mapQual > 0) {
					sam->print(seq2, *mapping2, flags2);
				}
			} else {
				sam->print(seq1, flags1);
				sam->print(seq2, flags2);
			}
			options->unlock();
			numAligned += (mapping1 != NULL) + (mapping2 != NULL);
			if (mapping1)
				delete mapping1;
			if (mapping2)
				delete mapping2;
		}

		/*statistical information*/
		numReads++;
		if (numReads % 100000 == 0) {
			etime = Utils::getSysTime();
			Utils::log(
					"processed %ld read pairs by thread %d in %.2f seconds\n",
					numReads, params->_tid, etime - stime);
		}
	}
	/*return the alignment results*/
	params->_numAligned += numAligned;
	params->_numReads += numReads * 2;
	params->_numPaired += numPaired * 2;

	return NULL;
}
void* PairedEnd::_gpuThreadFunc(void* arg) {
	size_t numReads = 0, numAligned = 0;
	ThreadParams *params = (ThreadParams*) arg;
	Options* options = params->_options;
	SeqFileParser *parser1 = params->_parser1;
	SeqFileParser *parser2 = params->_parser2;
	PairedEnd* aligner = (PairedEnd*) params->_aligner;
	GPUMemEngine* engine = (GPUMemEngine*) params->_engine;
	bool viaFifo = options->isViaFifo();
	size_t nseqs;

	double stime = Utils::getSysTime();
	double etime;

	GPUInfo *gpuInfo = GPUInfo::getGPUInfo();
	/*set the device*/
	gpuInfo->setDevice(options->getGPUIndex());

	/*get the property*/
	cudaDeviceProp* prop = gpuInfo->getDeviceProp(options->getGPUIndex());

	/*calculate the system-wide configurations*/
	size_t numReadsPerBatch = NUM_SM_SCALE * prop->multiProcessorCount
			* prop->maxThreadsPerMultiProcessor; /*must be multiples of 2*/
	if (numReadsPerBatch > MAX_READS_PER_BATCH) {
		numReadsPerBatch = MAX_READS_PER_BATCH;
	}
	size_t maxNumSeedPerBatch = EST_NUM_SEEDS_PER_READ * numReadsPerBatch;
	Utils::log("Max number of reads per batch: %ld\n", numReadsPerBatch);

	/*allocate space for read batches*/
	Sequence* sequences = new Sequence[numReadsPerBatch];
	if (!sequences) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}

	/*run the GPU computation*/
	for (;;) {

		/*read a batch of read pairs in an interleaved way*/
		aligner->lock();
		if (viaFifo) {
			for (nseqs = 0; nseqs < numReadsPerBatch; nseqs += 2) {
				if (!parser1->getSeqLockFree(sequences[nseqs],
						sequences[nseqs + 1])) {
					aligner->unlock();
					break;
				}
			}
		} else {
			for (nseqs = 0; nseqs < numReadsPerBatch; nseqs += 2) {
				if (!parser1->getSeqLockFree(sequences[nseqs])) {
					aligner->unlock();
					break;
				}
				if (!parser2->getSeqLockFree(sequences[nseqs + 1])) {
					Utils::log(
							"The two files have different number of sequences\n");
					aligner->unlock();
					break;
				}
			}
		}
		aligner->unlock();

		/*check the number of reads*/
		if (nseqs == 0) {
			break;
		}

		/*run the alignment*/
		numAligned += engine->run(sequences, nseqs, maxNumSeedPerBatch);

		/*clear the sequence data*/
		if (viaFifo) {
			/*If clearing each sequence here, the following "delete[]sequences" will failed when using FIFO;
			 * DO NOT KNOW WHY AT THE MOMENT. FIXME
			 */
			for (size_t i = 0; i < nseqs; ++i) {
				sequences[i].clear();
			}
		}

		/*calcualte total number of sequences*/
		numReads += nseqs;
		if (numReads % (numReadsPerBatch * 10) == 0) {
			etime = Utils::getSysTime();
			Utils::log("processed %ld reads by the GPU in %.2f seconds\n",
					numReads, etime - stime);
		}
	}
	delete[] sequences;

	/*return the alignment results*/
	params->_numAligned += numAligned;
	params->_numReads += numReads;

	return NULL;
}

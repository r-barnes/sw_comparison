/*
 * SingleEnd.cpp
 *
 *  Created on: Jan 18, 2012
 *      Author: yongchao
 */

#include "SingleEnd.h"
#include "SeqFileParser.h"
#include "GPUMemEngine.h"

SingleEnd::SingleEnd(Options* options, Genome * genome, SAM* sam) {
	int tid, gid;
	_options = options;
	_genome = genome;
	_sam = sam;

	/*get the number of threads*/
	_numThreads = _options->getNumThreads();
	_numCPUs = _options->getNumCPUs();
	_numGPUs = _options->getNumGPUs();

	/*create parameters for CPU threads*/
	_threads.resize(_numThreads);
	_engines.resize(_numCPUs);
	for (tid = 0; tid < _numCPUs; ++tid) {
		_engines[tid] = new MemEngine(_options, _genome, _sam);
		_threads[tid] = new ThreadParams(tid, _options, _sam, _engines[tid],
				this);
	}

	/*create parameters for GPU threads*/
	_gpuEngines.resize(_numGPUs);
	for (gid = 0; tid < _numThreads; ++tid, ++gid) {
		_gpuEngines[gid] = new GPUMemEngine(_options, _genome, _sam);
		
		/*initialize GPU*/
		_gpuEngines[gid]->initialize(options->getGPUIndex());

		_threads[tid] = new ThreadParams(tid, _options, _sam, _gpuEngines[gid],
				this);
	}

#ifdef FREE_UNUSED_GENOME_MEMORY
	/*If CPU threads are not used, free more host memory*/
	if (_threads.size() == 0) {
		freeGenomeMemory();
	}
#endif
}

SingleEnd::~SingleEnd() {
	for (size_t i = 0; i < _threads.size(); ++i) {
		delete _threads[i];
	}
	_threads.clear();

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

void SingleEnd::execute() {
	SeqFileParser *parser;
	vector<pthread_t> threadIDs(_numThreads, 0);

	/*for each input file*/
	vector<pair<string, int> > &inputs = _options->getInputFileList();
	for (size_t file = 0; file < inputs.size(); ++file) {
		parser = new SeqFileParser(inputs[file].first.c_str(), _numThreads > 1,
				inputs[file].second);

		/*create CPU threads*/
		size_t tid;
		for (tid = 0; tid < _engines.size(); ++tid) {
			/*set file parser*/
			_threads[tid]->setFileParser(parser);

			/*create thread entities*/
			if (pthread_create(&threadIDs[tid], NULL, _threadFunc,
					_threads[tid]) != 0) {
				Utils::exit("Thread creating failed\n");
			}
		}
		for (; tid < _threads.size(); ++tid) {

			/*set file parser*/
			_threads[tid]->setFileParser(parser);

			/*create thread entities*/
			if (pthread_create(&threadIDs[tid], NULL, _gpuThreadFunc,
					_threads[tid]) != 0) {
				Utils::exit("Thread creating failed\n");
			}
		}
		/*wait for the completion of all threads*/
		for (size_t tid = 0; tid < _threads.size(); ++tid) {
			pthread_join(threadIDs[tid], NULL);
		}
		/*release the file parser*/
		delete parser;
	}
	threadIDs.clear();
	SeqFileParser::finalize();

	/*report the alignment information*/
	size_t numReads = 0, numAlignedReads = 0;
	for (size_t tid = 0; tid < _threads.size(); ++tid) {
		numReads += _threads[tid]->_numReads;
		numAlignedReads += _threads[tid]->_numAligned;
	}
	Utils::log("#Reads aligned: %ld / %ld (%.2f%%)\n", numAlignedReads,
			numReads, ((double) (numAlignedReads)) / numReads * 100);
}

/*thread function for CPU threads*/
void* SingleEnd::_threadFunc(void* arg) {
	Sequence seq;
	Mapping* mapping;
	int64_t numReads = 0, numAligned = 0;
	ThreadParams* params = (ThreadParams*) arg;
	Options *options = params->_options;
	SAM *sam = params->_sam;
	SeqFileParser* parser = params->_parser;
	MemEngine* engine = (MemEngine*) params->_engine;
	double stime = Utils::getSysTime();
	double etime;
	bool unique = options->outOnlyUnique();
	int multi = options->getMaxMultiAligns();
	/*reserve space for the vector*/
	vector<Mapping*> mappings;
	mappings.reserve(multi);

	while (1) {
		/*read a sequence*/
		if (!parser->getSeq(seq)) {
			break;
		}

		/*output multiple equivalent alignments*/
		if (multi > 1) {
			/*invoke the engine*/
			if (engine->align(seq, mappings)) {
				++numAligned;
			}

			/*output the alignment*/
			options->lock();
			if (mappings.size() > 0) {
				for (size_t i = 0; i < mappings.size(); ++i) {
					mapping = mappings[i];
					if (unique == false /*select the random one*/
					|| mapping->_mapQual > 0) {
						sam->print(seq, *mapping);
					} else {
						sam->print(seq); /*as unaligned*/
					}
				}
			} else { /*unaligned*/
				sam->print(seq);
			}
			options->unlock();

			/*release the mapping*/
			for (size_t i = 0; i < mappings.size(); ++i) {
				delete mappings[i];
			}
			mappings.clear();
		} else {
			/*invoke the engine*/
			if (engine->align(seq, mapping)) {
				++numAligned;
			}

			/*output the alignment*/
			options->lock();
			if (mapping) {
				if (unique == false /*select the random one*/
				|| mapping->_mapQual > 0) {
					sam->print(seq, *mapping);
				} else {
					sam->print(seq); /*as unaligned*/
				}
			} else {
				sam->print(seq);
			}
			options->unlock();

			/*release the mapping*/
			if (mapping) {
				delete mapping;
			}
		}

		/*statistical information*/
		++numReads;
		if (numReads % 100000 == 0) {
			etime = Utils::getSysTime();
			Utils::log("processed %ld reads by CPU thread (%d) in %.2f seconds\n",
					numReads, params->_tid, etime - stime);
		}
	}

	/*return the alignment results*/
	params->_numAligned += numAligned;
	params->_numReads += numReads;

	return NULL;
}
/*thread function for CPU threads*/
void* SingleEnd::_gpuThreadFunc(void* arg) {
	Sequence seq;
	int64_t numReads = 0, numAligned = 0;
	ThreadParams* params = (ThreadParams*) arg;
	Options *options = params->_options;
	SeqFileParser* parser = params->_parser;
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
			* prop->maxThreadsPerMultiProcessor;
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
		/*read a batch of reads from the input*/
		if ((nseqs = parser->getSeq(sequences, numReadsPerBatch)) == 0) {
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

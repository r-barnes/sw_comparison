/*
 * PairedEnd.h
 *
 *  Created on: Jan 18, 2012
 *      Author: yongchao
 */

#ifndef PAIREDEND_H_
#define PAIREDEND_H_
#include "Macros.h"
#include "Options.h"
#include "MemEngine.h"
#include "Thread.h"
#include "GPUMemEngine.h"

class PairedEnd
{
public:
	PairedEnd(Options* options, Genome * genome, SAM* sam);
	~PairedEnd();

	/*execute the paired-end alignment*/
	void execute();
	void estimateInsertSize(int minAlignedPairs, int maxReadBatchSize,
			int numReadBatchs);
	inline void lock() {
		if (_numThreads > 1) {
			pthread_mutex_lock(&_mutex);
		}
	}
	inline void unlock() {
		if (_numThreads > 1) {
			pthread_mutex_unlock(&_mutex);
		}
	}

private:
	/*private member variables*/
	Options* _options;
	Genome* _genome;
	SAM* _sam;

	/*thread parameters*/
	int _numCPUs;
	vector<MemEngine*> _engines;
	vector<ThreadParams*> _threads;
	static void* _threadFunc(void*);

	/*GPU threads*/
	int32_t _numGPUs;	/*the number of GPUs*/
	vector<GPUMemEngine*> _gpuEngines;
	vector<ThreadParams*> _gpuThreads;
	static void* _gpuThreadFunc(void* arg);	/*thread functions for GPU threads*/

	pthread_mutex_t _mutex;

	/*all threads*/
	int _numThreads;

	/*free unused genome*/
	inline void freeGenomeMemory(uint32_t flags)
	{
		_genome->freeUnusedMemory(flags);
	}
};

#endif /* PAIREDEND_H_ */

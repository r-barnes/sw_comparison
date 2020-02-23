/*
 * SingleEnd.h
 *
 *  Created on: Jan 18, 2012
 *      Author: yongchao
 */

#ifndef SINGLEEND_H_
#define SINGLEEND_H_
#include "Macros.h"
#include "Options.h"
#include "Genome.h"
#include "SAM.h"
#include "Thread.h"
#include "GPUMemEngine.h"

class SingleEnd
{
public:
	SingleEnd(Options* options, Genome * genome, SAM* sam);
	~SingleEnd();

	/*execute the paired-end alignment*/
	void execute();

private:
	/*private member variables*/
	Options* _options;
	Genome* _genome;
	SAM* _sam;

	/*CPU and GPU threads information*/
	int32_t _numCPUs;
	vector<MemEngine*> _engines;
	static void* _threadFunc(void* arg);/*thread functions for CPU threads*/

	/*GPU threads*/
	int32_t _numGPUs;	/*the number of GPUs*/
	vector<GPUMemEngine*> _gpuEngines;
	static void* _gpuThreadFunc(void* arg);	/*thread functions for GPU threads*/

	/*All threads*/
	int32_t _numThreads;	/*the total number of threads = numCPUThreasd + numGPUThreads*/
	vector<ThreadParams*> _threads;

	/*free unused genome*/
	inline void freeGenomeMemory()
	{
		_genome->freeUnusedMemory();
	}
};

#endif /* SINGLEEND_H_ */

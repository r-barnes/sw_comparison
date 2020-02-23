/*
 * Options.h
 *
 *  Created on: Jan 10, 2012
 *      Author: yongchao
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_
#include "Macros.h"

class Options
{
public:
	Options();
	Options(int argc, char* argv[]);
	~Options();
	//get the file name of the suffix array index
	inline string& getSamFileName()
	{
		return _samFileName;
	}

	//get the prefix of BWT file name
	inline string& getBwtFileBase()
	{
		return _bwtFileBase;
	}
	//get the input file list
	inline vector< pair<string, int> >& getInputFileList()
	{
		return _readsFileNames;
	}
	inline bool outOnlyUnique()
	{
		return _onlyUnique;
	}
	inline int getMaxMultiAligns()
	{
		return _maxMultiAligns;
	}
	inline bool isPaired()
	{
		return _paired;
	}
	inline bool rescueMate()
	{
		return _rescueMate;
	}
#ifdef HAVE_TWICE_RESCUE
	inline bool rescueTwice()
	{
		return _rescueTwice;
	}
#endif
	inline void setPaired(bool pe)
	{
		_paired = pe;
	}
	inline bool isViaFifo()
	{
		return _viaFifo;
	}
	inline uint32_t getMaxSeedOcc()
	{
		return _maxSeedOcc;
	}
	inline float getMinRatio()
	{
		return _minRatio;
	}
	inline float getMinIdentity()
	{
		return _minIdentity;
	}
	inline int getMinAlignScore()
	{
		return _minAlginScore;
	}
	inline bool estimateInsertSize()
	{
		return _estInsertSize;
	}
	inline int getInsertSize()
	{
		return _insertSize;
	}
	inline void setInsertSize(int insertSize)
	{
		_insertSize = insertSize;
	}
	inline int getStdInsertSize()
	{
		return _stdInsertsize;
	}
	inline void setStdInsertSize(int stdInsertSize)
	{
		_stdInsertsize = stdInsertSize;
	}
	inline void setTopReadsEstIns(int nreads)
	{
		_topReadsEstIns = nreads;
	}
	inline int getTopReadsEstIns()
	{
		return _topReadsEstIns;
	}
	inline void setMapQualEstIns(int mapQ)
	{
		_mapQualReliable = mapQ;
	}
	inline int getMapQualReliable()
	{
		return _mapQualReliable;
	}
	inline int getNumThreads()
	{
		return _numThreads;
	}
	inline int getNumCPUs()
	{
		return _numCPUs;
	}
	inline int getNumGPUs()
	{
		return _numGPUs;
	}
	inline int getGPUIndex()
	{
		return _gpuIndex;
	}
	inline int getGapOpen()
	{
		return _gapOpen;
	}
	inline int getGapExtend()
	{
		return _gapExtend;
	}
	inline int getMismatch()
	{
		return _mismatch;
	}
	inline int getMatch()
	{
		return _match;
	}
	inline int getNumErrors(uint32_t length)
	{
		if(length > MAX_SEQ_LENGTH){
			return estimateNumErrors(length, _missProb);
		}
		return _numErrorTable[length];
	}
	inline int getLowerMinSeedSize()
	{
		return _lowerMinSeedSize;
	}
	inline int getMinSeedSize(uint32_t length)
	{
		if(length > MAX_SEQ_LENGTH){
			int numErrors = estimateNumErrors(length, _missProb);
			_numErrorTable[length] = numErrors;

			/*estimate the minimal seed size according to dove hole principle*/
			int seedSize = length / (numErrors + 1);
			if (seedSize < _lowerMinSeedSize)
			{
				seedSize = _lowerMinSeedSize;
			}
			if(seedSize > _upperMinSeedSize){
				seedSize = _upperMinSeedSize;
			}
			return seedSize;
		}
		return _minSeedSizeTable[length];
	}
	/*parse the command line*/
	bool parse(int argc, char* argv[]);

	inline void lock(){
		if(_numThreads > 1){
			pthread_mutex_lock(&globalMutex);
		}
	}
	inline void unlock()
	{
		if(_numThreads > 1){
			pthread_mutex_unlock(&globalMutex);
		}
	}
	void printUsage();
private:
	/*private member variables*/
	string _bwtFileBase;
	string _samFileName;
	vector< pair<string, int> > _readsFileNames;
	uint32_t _maxSeedOcc;
	float _minRatio;
	float _minIdentity;
	float _missProb;
	bool _estInsertSize;
	int _insertSize;
	int _stdInsertsize;
	int _topReadsEstIns;
	int _mapQualReliable;
	bool _onlyUnique;
	int _maxMultiAligns;
	bool _paired;
	bool _viaFifo;
	bool _rescueMate;
#ifdef HAVE_TWICE_RESCUE
	bool _rescueTwice;
#endif
	int _numThreads;
	int _numCPUs;
	int _numGPUs;
	int _gpuIndex;
	int _lowerMinSeedSize;
	int _upperMinSeedSize;

	int _gapOpen;
	int _gapExtend;
	int _mismatch;
	int _match;
	int _minAlginScore;

	int *_numErrorTable;
	int *_minSeedSizeTable;

	/*read group information*/
	string _rgID;	/*read group identifier*/
	string _rgSM;  /*sample name. Use pool name where a pool is being sequence*/
	string _rgLB;	/*library*/
	string _rgPL;  /*platform/technology used to produce the reads*/
	string _rgPU;	/*platform unit*/
	string _rgCN;	/*name of sequencing center produced the read*/
	string _rgDS;	/*description*/
	string _rgDT;	/*date the run was produced*/
	string _rgPI;	/*predicted median insert size*/

	/*global file lock*/
	pthread_mutex_t globalMutex;
private:
	/*private member functions*/
	void _setDefaults();
	inline int estimateNumErrors(size_t length, float errorRate)
	{
		float uniformErrorRate = 0.02;
		double elambda = exp(-length * uniformErrorRate);
		double sum, y = 1.0;
		int k, x = 1;
		for (k = 1, sum = elambda; k < 1000; ++k)
		{
			y *= length * uniformErrorRate;
			x *= k;
			sum += elambda * y / x;
			if (1.0 - sum < errorRate){
				return k;
			}
		}
		return 2;
	}

	friend class SAM;
};

#endif /* OPTIONS_H_ */

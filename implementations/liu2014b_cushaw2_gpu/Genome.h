/*
 * Genome.h
 *
 *  Created on: Dec 28, 2011
 *      Author: yongchao
 */

#ifndef GENOME_H_
#define GENOME_H_
#include "Macros.h"
#include "BWT.h"
#include "SuffixArray.h"
#include "Utils.h"

struct BwtAnn
{
	BwtAnn()
	{
		_name = _anno = NULL;
	}
	~BwtAnn()
	{
		if (_name)
			delete[] _name;
		if (_anno)
			delete[] _anno;
	}

	int64_t _offset;
	int32_t _length;
	int32_t _numAmbs;
	uint32_t _gi;
	char* _name;
	char* _anno;
};

struct BwtAmb
{
	int64_t _offset;
	int32_t _length;
	char _amb;
};

class Genome
{
public:
	Genome(string bwtBase, int dir)
	{
		string bwtFileName, saFileName;
		string annFileName, ambFileName, pacFileName;

		//initialize
		_bwt = NULL;
		_sa = NULL;
		_anns = NULL;
		_ambs = NULL;

		if (dir == 0)
		{
			bwtFileName = bwtBase + ".bwt";
			saFileName = bwtBase + ".sa";
		}
		else
		{
			bwtFileName = bwtBase + ".rbwt";
			saFileName = bwtBase + ".rsa";
		}
		annFileName = bwtBase + ".ann";
		ambFileName = bwtBase + ".amb";
		pacFileName = bwtBase + ".pac";

		init(bwtFileName, saFileName, annFileName, ambFileName, pacFileName);
	}
	~Genome()
	{
		if (_bwt)
			delete _bwt;
		if (_sa)
			delete _sa;
		if (_anns)
			delete[] _anns;
		if (_ambs)
			delete[] _ambs;
		if (_pacGenome)
		{
			delete[] _pacGenome;
		}
	}

	inline BWT* getBWT()
	{
		return _bwt;
	}
	inline SuffixArray* getSuffixArray()
	{
		return _sa;
	}
	inline uint8_t* getPacGenome()
	{
		return _pacGenome;
	}
	bool bridging(int genomeIndex, uint32_t position, uint32_t length)
	{
		int64_t left, right;

		//refine the range
		left = _anns[genomeIndex]._offset;
		right = left + _anns[genomeIndex]._length;

		/*check the range*/
		if (position < left || position + length > right)
		{
			return true;
		}
		return false;
	}
	inline void refineRegionRange(int genomeIndex, int64_t& lowerBound,
			int64_t& upperBound)
	{
		int64_t left, right;

		//refine the range
		left = _anns[genomeIndex]._offset;
		right = left + _anns[genomeIndex]._length - 1;

		if (lowerBound < left)
		{
			lowerBound = left;
		}else if (lowerBound > right)
		{
			lowerBound = right;
		}
		if (upperBound < left)
		{
			upperBound = left;
		}else if (upperBound > right)
		{
			upperBound = right;
		}
	}
	/*get the genome sequence index in the packed genome*/
	void getGenomeIndex(uint32_t position, int& genomeIndex);

	/*check if the alignment lies in a random filed*/
	int getRandomField(uint32_t position, int length);

	/*get the offset of the genome*/
	inline int64_t getGenomeOffset(int genomeIndex)
	{
		return _anns[genomeIndex]._offset;
	}

	/*get the genome sequence name*/
	inline char* getGenomeName(int genomeIndex)
	{
		return _anns[genomeIndex]._name;
	}

	/*print out all genome names in SAM format*/
	inline void genomeNamesOut(FILE* file)
	{
		if (file == NULL)
		{
			Utils::exit("Invalid file pointer in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		for (int i = 0; i < _numSeqs; ++i)
		{
			fprintf(file, "@SQ\tSN:%s\tLN:%d\n", _anns[i]._name,
					_anns[i]._length);
		}
	}
	inline void genomeNamesOut(gzFile file)
	{
		if (file == NULL)
		{
			Utils::exit("Invalid file pointer in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		for (int i = 0; i < _numSeqs; ++i)
		{
			gzprintf(file, "@SQ\tSN:%s\tLN:%d\n", _anns[i]._name,
					_anns[i]._length);
		}
	}

	inline uint64_t getGenomeLength()
	{
		return _genomeLength;
	}

	inline uint32_t getNumSeqs()
	{
		return _numSeqs;
	}
	inline BwtAnn* getBwtAnn()
	{
		return _anns;
	}

	inline void freeUnusedMemory(uint32_t flags = 3)
	{
		if(flags & 1){
			/*free BWT*/
			_bwt->freeUnusedMemory();
		}

		if(flags & 2){
			/*free packed genome*/
			if(_pacGenome){
				delete [] _pacGenome;
				_pacGenome = NULL;
			}
		}
	}
private:
	/*private member variables*/
	BWT* _bwt;
	SuffixArray* _sa;

	/*genome information*/
	int64_t _genomeLength; /*the original genome length*/
	uint8_t* _pacGenome;
	uint32_t _seed;
	int _numSeqs;
	BwtAnn* _anns; // _numSeqs elements
	int _numHoles;
	BwtAmb* _ambs; // _numHoles elements

private:
	/*private member functions*/
	void init(string bwtFileName, string saFileName, string annFileName,
			string ambFileName, string pacFileName);
};
#endif /* GENOME_H_ */

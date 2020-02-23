/*
 * Genome.cpp
 *
 *  Created on: Dec 28, 2011
 *      Author: yongchao
 */

#include "Genome.h"
#include "Utils.h"

void Genome::init(string bwtFileName, string saFileName, string annFileName,
		string ambFileName, string pacFileName)
{
	FILE* file;
	char buffer[4096];
	int intVar;
	long long longVar;
	BwtAnn* ann;
	BwtAmb* amb;
	size_t length;

	//read .bwt file
	_bwt = new BWT(bwtFileName.c_str());

	//read .sa file
	_sa = new SuffixArray(saFileName.c_str());

	//read .ann file
	file = fopen(annFileName.c_str(), "r");
	if (file == NULL)
	{
		Utils::exit("Failed to open file %s\n", annFileName.c_str());
	}
	fscanf(file, "%lld%d%u", &longVar, &_numSeqs, &_seed);

	_genomeLength = longVar;
	_anns = new BwtAnn[_numSeqs];
	if (_anns == NULL)
	{
		Utils::exit("Memory allocation failed in function %s in line %d\n",
				__FUNCTION__, __LINE__);
	}
	for (int i = 0; i < _numSeqs; ++i)
	{
		//get the pointer address

		ann = _anns + i;
		//read gi and sequence name
		fscanf(file, "%u%s", &(ann->_gi), buffer);
		length = strlen(buffer);
		ann->_name = new char[length + 1];
		if (ann->_name == NULL)
		{
			Utils::exit("Memory allocation failed in function %s in line %d\n",
					__FUNCTION__, __LINE__);
		}
		strcpy(ann->_name, buffer);

		/*read comments*/
		int c;
		char* p = buffer;
		while ((c = fgetc(file)) != '\n' && c != EOF)
			*p++ = c;
		*p = '\0';

		if (p - buffer > 1)
		{
			length = strlen(buffer);
			ann->_anno = new char[length];
			if (ann->_anno == NULL)
			{
				Utils::exit(
						"Memory allocation failed in function %s in line %d\n",
						__FUNCTION__, __LINE__);
			}
			strcpy(ann->_anno, buffer + 1);
		}
		else
		{
			ann->_anno = new char[1];
			if (ann->_anno == NULL)
			{
				Utils::exit(
						"Memory allocation failed in function %s in line %d\n",
						__FUNCTION__, __LINE__);
			}
			ann->_anno[0] = '\0';
		}

		/*read the remaining part*/
		fscanf(file, "%lld%d%d", &longVar, &ann->_length, &ann->_numAmbs);
		ann->_offset = longVar;
	}
	fclose(file);

	/*read .amb file*/
	file = fopen(ambFileName.c_str(), "r");
	if (file == NULL)
	{
		Utils::exit("Failed to open file %s\n", ambFileName.c_str());
	}
	/*read the packed genome length and the number of sequences in the genome*/
	fscanf(file, "%lld%d%d", &longVar, &intVar, &_numHoles);
	if (!(longVar == _genomeLength && intVar == _numSeqs))
	{
		Utils::exit("Inconsistent .ann and .amb files\n");
	}
	_ambs = new BwtAmb[_numHoles];
	if (_ambs == NULL)
	{
		Utils::exit("Memory allocation failed in function %s in line %d\n",
				__FUNCTION__, __LINE__);
	}
	for (int i = 0; i < _numHoles; ++i)
	{
		amb = _ambs + i;
		fscanf(file, "%lld%d%s", &longVar, &amb->_length, buffer);
		amb->_offset = longVar;
		amb->_amb = buffer[0];
	}
	fclose(file);

#if 0
	/*print out the sequence information in the genome*/
	for (int i = 0; i < _numSeqs; ++i)
		Utils::log("@SQ\tSN:%s\tLN:%d\n", _anns[i]._name,
			_anns[i]._length);
#endif

	/*read .pac file*/
	file = fopen(pacFileName.c_str(), "r");
	if (file == NULL)
	{
		Utils::exit("Failed to open file %s\n", ambFileName.c_str());
	}

	/*calculate the paced genome size*/
	size_t _pacGenomeBytes = (size_t) ((_genomeLength + 3) >> 2);

	/*allocate space*/
	_pacGenome = new uint8_t[_pacGenomeBytes];
	if (_pacGenome == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	//read the file
	size_t bytes = fread(_pacGenome, 1, _pacGenomeBytes, file);
	if (bytes != (size_t) _pacGenomeBytes)
	{
		Utils::exit("Incomplete file %s (read %ld != %ld )\n",
				pacFileName.c_str(), bytes, _pacGenomeBytes);
	}

	fclose(file);
}

void Genome::getGenomeIndex(uint32_t position, int& genomeIndex)
{
	int left, mid, right;
	if (position >= _genomeLength)
	{
		Utils::exit(
				"getGenomeIndex: Mapping position is longer than sequence length (%lld >= %lld)",
				position, _genomeLength);
	}
	// binary search for the sequence ID. Note that this is a bit different from the following one...

	left = 0;
	mid = 0;
	right = _numSeqs;
	while (left < right)
	{
		mid = (left + right) >> 1;
		if (position >= _anns[mid]._offset)
		{
			if (mid == _numSeqs - 1)
				break;
			if (position < _anns[mid + 1]._offset)
				break;

			left = mid + 1;
		}
		else
		{
			right = mid;
		}
	}
	genomeIndex = mid;
}
int Genome::getRandomField(uint32_t position, int length)
{
	int left, right;
	if (position >= _genomeLength)
	{
		Utils::exit(
				"getRandomField: Mapping position is longer than sequence length (%lld >= %lld)",
				position, _genomeLength);
	}

	// binary search for holes
	left = 0;
	right = _numHoles;
	int nn = 0;
	while (left < right)
	{
		int64_t mid = (left + right) >> 1;
		if (position >= _ambs[mid]._offset + _ambs[mid]._length)
		{
			left = mid + 1;
		}
		else if (position + length <= _ambs[mid]._offset)
		{
			right = mid;
		}
		else
		{ // overlap
			if (position >= _ambs[mid]._offset)
			{
				nn += _ambs[mid]._offset + _ambs[mid]._length
				< position + length ?
				_ambs[mid]._offset + _ambs[mid]._length - position :
				length;
			}
			else
			{
				nn += _ambs[mid]._offset + _ambs[mid]._length
				< position + length ?
				_ambs[mid]._length :
				length - (_ambs[mid]._offset - position);
			}
			break;
		}
	}
	return nn;
}


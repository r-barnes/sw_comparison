/*
 * SAM.h
 *
 *  Created on: Jan 5, 2012
 *      Author: yongchao
 */

#ifndef SAM_H_
#define SAM_H_
#include "Macros.h"
#include "Genome.h"
#include "CigarAlign.h"
#include "Sequence.h"
#include <pthread.h>
#include "Options.h"
#include "Mapping.h"

class SAM
{
public:
	SAM(Options* options, Genome* genome);
	~SAM();

	/*open the output file*/
	void open();
	/*close the output file*/
	void close();
	void print(Sequence& seq, int flags = 0);
	void print(Sequence& seq, Mapping& mapping, int flags = 0);
	void print(Sequence& seq, Mapping& mapping, int32_t mateGenomeIndex,
			int32_t matePosition, int32_t insertSize, int flags = 0);
	void print(Sequence& seq, Mapping& selft, Mapping& mate, int flags = 0);

	/*for GPU computing*/
	void print(Sequence& seq, uint16_t* cigars, uint32_t numCigars,
			int32_t genomeIndex, int64_t mapPosition, uint32_t strand,
			int32_t mapQual, int flags = 0);
	void print(Sequence& seq, uint16_t* cigars, uint32_t numCigars,
			int32_t genomeIndex, int64_t mapPosition, uint32_t strand,
			int32_t mapQual, int32_t mateGenomeIndex, int32_t matePosition,
			int32_t insertSize, int flags = 0);
private:
	/*private member variables*/
	int _numThreads;
	string _fileName;
	Genome* _genome;

	/*output file*/
	FILE* _file;
};

#endif /* SAM_H_ */

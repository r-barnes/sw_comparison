/*
 * Aligner.h
 *
 *  Created on: Dec 29, 2011
 *      Author: yongchao
		We have used and modified some code from the open-source SWIPE algorithm in this file.
		The following details the copyright and licence information of SWIPE.

    SWIPE
    Smith-Waterman database searches with Inter-sequence Parallel Execution

    Copyright (C) 2008-2012 Torbjorn Rognes, University of Oslo, 
    Oslo University Hospital and Sencel Bioinformatics AS

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Contact: Torbjorn Rognes <torognes@ifi.uio.no>, 
    Department of Informatics, University of Oslo, 
    PO Box 1080 Blindern, NO-0316 Oslo, Norway
*/

#ifndef ALIGNER_H_
#define ALIGNER_H_
#include "Macros.h"
#include "CigarAlign.h"
#include "Options.h"
#include "Seed.h"

typedef unsigned int UINT32;
typedef unsigned short WORD;
typedef unsigned char BYTE;
typedef BYTE VECTOR[16];


class Aligner
{
public:
	Aligner(Options* options);
#ifdef HAVE_TWICE_RESCUE
	Aligner(Options* options, int32_t match, int32_t mismatch, int32_t gopen, int32_t gext);
#endif
	~Aligner();

	/*local alignment score*/
	void lalignScore(uint8_t* query, int32_t qlen, uint8_t* sequences,
			int32_t* seqOffsets, int32_t numSeqs, AlignScore* scores);

	int32_t lalignScore(uint8_t* s1, uint8_t * s2, int32_t s1Length,
			int32_t s2Length, int32_t low, int32_t up);

	/*local alignment path*/
	vector<CigarAlign*> lalignPath(uint8_t* s1, uint8_t * s2, int32_t s1Length,
			int32_t s2Length, int32_t low, int32_t up, int which);

	inline int32_t estMinAlignScore(uint32_t length)
	{
		int32_t numErrors = _options->getNumErrors(length);
		int32_t penalty = min(_mismatch, -_gopen);

		return (length - numErrors) * _match + penalty * numErrors;
	}
private:
	Options* _options;
	/*private member variables*/
	/*scoring matrix and gap penalties*/
	int8_t _smat[5][8]; /*score matrix*/
	int32_t _gopen; /*gap-open penalty*/
	int32_t _gext; /*gap-ext penalty*/
	int32_t _goe;	/*sum of gap-open and -ext penality*/
	int32_t _mismatch;
	int32_t _match;
	int8_t *_profile;
	int _profileSize;
	int2* _heBuffer;
	int _heBufferSize;
	int8_t* _tbtable;
	int _tbtableSize;

private:
	/*private member functions*/
	/*convert the alignment to cigar format*/
	CigarAlign* align2cigar(uint8_t* s1, uint8_t* s2, int32_t s1Length,
			int32_t s2Length, int8_t* tbtble, int32_t bandBufferSize,
			int32_t cibest, int32_t cjbest, int32_t low, int32_t alignScore,
			bool rev);

	/*print out the alignment*/
	int32_t traceback(uint8_t* s1, uint8_t* s2, int32_t s1Length,
			int32_t s2Length, int8_t* tbtable, int32_t bandBufferSize,
			int32_t cibest, int32_t cjbest, int32_t alignScore, int32_t low);

private:

#ifndef USE_FULL_SW_64
	/*score matrix for SSE2*/
	int _scorelimit7;
	char* _score_matrix_7;
	int16_t* _score_matrix_16;
	BYTE* _dprofile;
	BYTE** _qtable;

	int32_t search7(BYTE** qtable, BYTE gap_open_penalty,
			BYTE gap_extend_penalty, BYTE * score_matrix, BYTE * dprofile,
			BYTE * hearray, int32_t qlen, int32_t numSeqs, int32_t* seqOffsets,
			uint8_t* sequences, AlignScore* scores);
	void search16(WORD** qtable, WORD gap_open_penalty,
			WORD gap_extend_penalty, WORD* score_matrix, WORD* dprofile,
			WORD* hearray, int32_t qlen, int32_t numSeqs, int32_t *seqOffsets,
			uint8_t *sequences, AlignScore* scores);
#else
	long fullsw(long gap_open_penalty, long gap_extend_penalty, long* score_matrix, long* hearray,
			uint8_t* dseq, uint8_t * dend, uint8_t* qseq, uint8_t* qend);
	long* _score_matrix_64;
#endif
	BYTE* _hearray;
	int32_t _maxQuerySize;
	bool _haveSSSE3;
};
#endif /* Aligner_H_ */

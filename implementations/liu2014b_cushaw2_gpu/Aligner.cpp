/*
 * Aligner.cpp
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

#include "Aligner.h"
#include "Utils.h"
#include "SeqFileParser.h"

/*for SSE2 implementations*/
#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h>
#endif
/*
 * Aligner.cpp
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

#include "Aligner.h"
#include "Utils.h"
#include "SeqFileParser.h"

//#define ALIGN_DEBUG
#define cpuid(f,a,b,c,d) asm("cpuid": "=a" (a), "=b" (b), "=c" (c), "=d" (d) : "a" (f));
Aligner::Aligner(Options* options)
{
	/*check the SSSE3 and SSE2*/
	unsigned int a __attribute__ ((unused));
	unsigned int b __attribute__ ((unused));
	unsigned int c, d;

	cpuid(1, a, b, c, d);
	//  printf("cpuid: %08x %08x %08x %08x\n", a, b, c, d);

	/*check sse2*/
	if (!((d >> 26) & 1))
	{
		Utils::exit("!!!!Requiring a CPU with SSE2 support!!!!\n");
	}
	/*check SSSE3*/
	if ((c >> 9) & 1)
	{
		_haveSSSE3 = true;
	}
	else
	{
		_haveSSSE3 = false;
		Utils::log("!!!!DO NOT have SSSE3 support on the CPU, resulting in lower speed\n");
	}

	_options = options;
	/*get gap open and extension penalties*/
	_gopen = options->getGapOpen();
	_gext = options->getGapExtend();
	_goe = _gopen + _gext;

	/*score matrix*/
	_match = options->getMatch();
	_mismatch = -options->getMismatch();

	/*for mismatches*/
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			_smat[i][j] = (i == j) ? _match : _mismatch;
		}
	}

	_profile = NULL;
	_profileSize = 0;
	_heBuffer = NULL;
	_heBufferSize = 0;
	_tbtable = NULL;
	_tbtableSize = 0;
	_maxQuerySize = MAX_SEQ_LENGTH;

#ifndef USE_FULL_SW_64
	/*scoring matrix for SSE2*/
	_score_matrix_7 = new char[32 * 32];
	if (_score_matrix_7 == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	_score_matrix_16 = new int16_t[32 * 32];
	if (_score_matrix_16 == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	/*initialize the matrix with _smat*/
	int hi = -100;
	for (int i = 1; i <= 5; i++)
	{
		for (int j = 1; j <= 5; ++j)
		{
			char score = _smat[i - 1][j - 1];
			_score_matrix_7[(i << 5) + j] = score;
			_score_matrix_16[(i << 5) + j] = score;
			if (score > hi)
			{
				hi = score;
			}
		}
	}
	_scorelimit7 = 128 - hi;

	_dprofile = new uint8_t[4 * 16 * 32];
	if (_dprofile == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	_qtable = new BYTE*[_maxQuerySize];
	if (_qtable == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
#else
	_score_matrix_64 = new long [32 * 32];
	if(_score_matrix_64 == NULL){
    Utils::exit("Memory allocation failed in function %s line %d\n",
        __FUNCTION__, __LINE__);
	}
	for (int i = 1; i <= 5; i++)
  {
    for (int j = 1; j <= 5; ++j)
    {
    	_score_matrix_64[(i << 5) + j] = _smat[i - 1][j - 1];
    }
  }
#endif
	_hearray = new BYTE[_maxQuerySize * 32];
	if (_hearray == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
}
#ifdef HAVE_TWICE_RESCUE
Aligner::Aligner(Options* options, int32_t match, int32_t mismatch, int32_t gopen, int32_t gext)
{
	/*check the SSSE3 and SSE2*/
	unsigned int a __attribute__ ((unused));
	unsigned int b __attribute__ ((unused));
	unsigned int c, d;

	cpuid(1, a, b, c, d);
	//  printf("cpuid: %08x %08x %08x %08x\n", a, b, c, d);

	/*check sse2*/
	if (!((d >> 26) & 1))
	{
		Utils::exit("!!!!Requiring a CPU with SSE2 support!!!!\n");
	}
	/*check SSSE3*/
	if ((c >> 9) & 1)
	{
		_haveSSSE3 = true;
	}
	else
	{
		_haveSSSE3 = false;
		Utils::log("!!!!DO NOT have SSSE3 support on the CPU, resulting in lower speed\n");
	}

	_options = options;
	/*get gap open and extension penalties*/
	_gopen = gopen;
	_gext = gext;
	_goe = _gopen + _gext;

	/*score matrix*/
	_match = match;
	_mismatch = -mismatch;

	/*for mismatches*/
	for (int i = 0; i < 5; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			_smat[i][j] = (i == j) ? _match : _mismatch;
		}
	}

	_profile = NULL;
	_profileSize = 0;
	_heBuffer = NULL;
	_heBufferSize = 0;
	_tbtable = NULL;
	_tbtableSize = 0;
	_maxQuerySize = MAX_SEQ_LENGTH;

#ifndef USE_FULL_SW_64
	/*scoring matrix for SSE2*/
	_score_matrix_7 = new char[32 * 32];
	if (_score_matrix_7 == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	_score_matrix_16 = new int16_t[32 * 32];
	if (_score_matrix_16 == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	/*initialize the matrix with _smat*/
	int hi = -100;
	for (int i = 1; i <= 5; i++)
	{
		for (int j = 1; j <= 5; ++j)
		{
			char score = _smat[i - 1][j - 1];
			_score_matrix_7[(i << 5) + j] = score;
			_score_matrix_16[(i << 5) + j] = score;
			if (score > hi)
			{
				hi = score;
			}
		}
	}
	_scorelimit7 = 128 - hi;

	_dprofile = new uint8_t[4 * 16 * 32];
	if (_dprofile == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	_qtable = new BYTE*[_maxQuerySize];
	if (_qtable == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
#else
	_score_matrix_64 = new long [32 * 32];
	if(_score_matrix_64 == NULL){
    Utils::exit("Memory allocation failed in function %s line %d\n",
        __FUNCTION__, __LINE__);
	}
	for (int i = 1; i <= 5; i++)
  {
    for (int j = 1; j <= 5; ++j)
    {
    	_score_matrix_64[(i << 5) + j] = _smat[i - 1][j - 1];
    }
  }
#endif
	_hearray = new BYTE[_maxQuerySize * 32];
	if (_hearray == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
}
#endif

Aligner::~Aligner()
{
#ifndef USE_FULL_SW_64
	if (_profile)
	{
		delete[] _profile;
	}
	if (_heBuffer)
	{
		delete[] _heBuffer;
	}
	if (_tbtable)
	{
		delete[] _tbtable;
	}
	delete[] _qtable;
	delete[] _dprofile;
	delete[] _score_matrix_16;
	delete[] _score_matrix_7;
#else
	delete [] _score_matrix_64;
#endif
	delete[] _hearray;
}
//#define ONLY_8CHANNEL
void Aligner::lalignScore(uint8_t* query, int qlen, uint8_t* sequences,
		int* seqOffsets, int numSeqs, AlignScore* scores)
{

	if (qlen > _maxQuerySize)
	{
		_maxQuerySize = qlen * 2;
#ifndef USE_FULL_SW_64
		if (_qtable)
		{
			delete[] _qtable;
		}
		_qtable = new BYTE*[_maxQuerySize];
		if (_qtable == NULL)
		{
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
#endif
		if (_hearray)
		{
			delete[] _hearray;
		}
		_hearray = new BYTE [_maxQuerySize * 32];
		if (_hearray == NULL)
		{
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}

	for (int32_t i = 0; i < qlen; ++i)
	{
		query[i] += 1; /*increase each base for the query sequence*/
#ifndef USE_FULL_SW_64
		_qtable[i] = _dprofile + 64 * query[i];
#endif
	}

#ifndef USE_FULL_SW_64
	/*perform the search*/
#ifdef ONLY_8CHANNEL
	search16((WORD**) _qtable, _goe, _gext, (WORD*) _score_matrix_16,
			(WORD*) _dprofile, (WORD*) _hearray, qlen, numSeqs, seqOffsets,
			sequences, scores);
#else
	if (search7(_qtable, _goe, _gext, (BYTE*) _score_matrix_7, _dprofile,
			_hearray, qlen, numSeqs, seqOffsets, sequences, scores)
			>= _scorelimit7)
	{
		search16((WORD**) _qtable, _goe, _gext, (WORD*) _score_matrix_16,
				(WORD*) _dprofile, (WORD*) _hearray, qlen, numSeqs, seqOffsets,
				sequences, scores);
	}
#endif
#else
	for(int seq = 0; seq < numSeqs; ++seq){
		uint8_t* dseq = sequences + seqOffsets[seq];
		uint8_t* dend = sequences + seqOffsets[seq + 1] - 1;
		scores[seq]._score = fullsw((long)_goe, (long)_gext, (long*)_score_matrix_64, (long*)_hearray, dseq, dend, query,  query + qlen);
	}
#endif

	/*recovery the query sequence*/
	for (int32_t i = 0; i < qlen; ++i)
	{
		query[i] -= 1;
	}
}
int32_t Aligner::lalignScore(uint8_t* s1, uint8_t* s2, int32_t s1Length,
		int32_t s2Length, int32_t low, int32_t up)
{
	/******************************************
	 * refer to the paper: Kun-Mao Chao, William R. Pearson and Webb Miller (1992)
	 * Aligner two sequences within a specified diagonal band".
	 * Comput Appl Biosci, 8(5): 481-487 */

	int32_t band;
	int32_t leftd, rightd;
	int32_t lowrow, hirow;
	int32_t score, h, e, f;
	int32_t itrans, jtrans;
	int8_t* mat;
	int2* heBuffer;

	band = up - low + 1;
	if (band < 1)
	{
		Utils::log(
				"low > up is unacceptable for banded local CigarAlign (%d > %d)\n",
				low, up);
		return 0;
	}
	/*CigarAlign buffer*/
	heBuffer = new int2[band + 2];
	if (heBuffer == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	/*initialize the diagonals*/
	if (low > 0)
	{
		leftd = 1;
	}
	else if (up < 0)
	{
		leftd = band;
	}
	else
	{
		leftd = 1 - low;
	}
	rightd = band;

	/*initialize the rows*/
	lowrow = max(0, -up); /* start index -1 */
	hirow = min(s1Length, s2Length - low); /* end index */

	/*calculate profile*/
	int8_t* p;
	int32_t psize = (s1Length + 1) * 5;
	if (psize > _profileSize)
	{
		if (_profile)
		{
			delete[] _profile;
		}
		_profileSize = psize * 2;
		_profile = new int8_t[_profileSize];
	}
	p = _profile + 5;
	for (int32_t i = 0; i < s1Length; ++i)
	{
		mat = _smat[s1[i]];
		for (int32_t j = 0; j < 5; ++j)
		{
			p[j] = mat[j];
		}
		p += 5;
	}
	/*initialize the vectors*/
	for (int32_t j = leftd; j <= rightd; ++j)
	{
		heBuffer[j].x = 0;
		heBuffer[j].y = -_gopen;
	}

	heBuffer[rightd + 1].x = ALIGN_MIN_SCORE;
	heBuffer[rightd + 1].y = ALIGN_MIN_SCORE;

	heBuffer[leftd - 1].x = ALIGN_MIN_SCORE;
	heBuffer[leftd].y = -_gopen;

	score = 0;
	for (int i = lowrow + 1; i <= hirow; ++i)
	{
		if (leftd > 1)
		{
			--leftd;
		}

		if (i > s2Length - up)
		{
			--rightd;
		}

		/*calculate the E value*/
		mat = _profile + i * 5;
		if ((h = heBuffer[leftd + 1].x - _goe)
				> (e = heBuffer[leftd + 1].y - _gext))
		{
			e = h;
		}

		/*convert to the original CigarAlign matrix*/
		if ((itrans = leftd + low + i - 1) > 0)
		{
			h = heBuffer[leftd].x + mat[s2[itrans - 1]];
		}
		if (e > h)
		{
			h = e;
		}
		if (h < 0)
		{
			h = 0;
		}

		f = h - _gopen;
		heBuffer[leftd].x = h;
		heBuffer[leftd].y = e;
		if (h > score)
		{
			score = h;
		}

		for (int32_t curd = leftd + 1; curd <= rightd; ++curd)
		{
			if ((h = h - _goe) > (f = f - _gext))
			{
				f = h;
			}
			if ((h = heBuffer[curd + 1].x - _goe)
					> (e = heBuffer[curd + 1].y - _gext))
			{
				e = h;
			}
			jtrans = curd + low + i - 1;
			h = heBuffer[curd].x + mat[s2[jtrans - 1]];
			if (f > h)
			{
				h = f;
			}
			if (e > h)
			{
				h = e;
			}
			if (h < 0)
			{
				h = 0;
			}
			heBuffer[curd].x = h;
			heBuffer[curd].y = e;
			if (h > score)
			{
				score = h;
			}
		}
	}
	delete[] heBuffer;

	return score;
}

vector<CigarAlign*> Aligner::lalignPath(uint8_t* s1, uint8_t* s2,
		int32_t s1Length, int32_t s2Length, int32_t low, int32_t up, int which)
{
	/******************************************
	 * refer to the paper: Kun-Mao Chao, William R. Pearson and Webb Miller (1992)
	 * Aligner two sequences within a specified diagonal band".
	 * Comput Appl Biosci, 8(5): 481-487 */

	int32_t band;
	int32_t leftd, rightd;
	int32_t lowrow, hirow;
	int32_t score, h, e, f;
	int32_t jtrans;
	int8_t* mat;
	int8_t *table; /*trace-back table*/
	int32_t cibest, cjbest;
	int32_t bandBufferSize;
	int8_t dir = ALIGN_DIR_STOP;
	vector<CigarAlign*> aligns;
	int bufferSize;

	band = up - low + 1;
	if (band < 1)
	{
		Utils::log(
				"low > up is unacceptable for banded local CigarAlign (%d > %d)\n",
				low, up);
		return aligns;
	}
	bandBufferSize = band + 2;

	/*CigarAlign buffer*/
	if (bandBufferSize > _heBufferSize)
	{
		if (_heBuffer)
		{
			delete[] _heBuffer;
		}
		_heBufferSize = bandBufferSize * 2;
		_heBuffer = new int2[_heBufferSize];
		if (_heBuffer == NULL)
		{
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}
	/*trace-back table*/
	bufferSize = bandBufferSize * (s1Length + 1);
	if (bufferSize > _tbtableSize)
	{
		if (_tbtable)
		{
			delete[] _tbtable;
		}
		_tbtableSize = bufferSize * 2;
		_tbtable = new int8_t[_tbtableSize];
		if (_tbtable == NULL)
		{
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}

	/*initialize the diagonals*/
	if (low > 0)
	{
		leftd = 1;
	}
	else if (up < 0)
	{
		leftd = band;
	}
	else
	{
		leftd = 1 - low;
	}
	rightd = band;

	/*initialize the rows*/
	lowrow = max(0, -up); /* start index -1 */
	hirow = min(s1Length, s2Length - low); /* end index */

	/*initialize the CigarAlign table*/

	table = _tbtable;
	for (int32_t i = 0; i < bandBufferSize; ++i)
	{
		table[i] = ALIGN_DIR_STOP;
	}
	table = _tbtable + s1Length * bandBufferSize;
	for (int32_t i = 0; i < bandBufferSize; ++i)
	{
		table[i] = ALIGN_DIR_STOP;
	}
	table = _tbtable;
	for (int32_t i = 0; i <= s1Length; ++i)
	{
		*table = ALIGN_DIR_STOP;
		table += bandBufferSize;
	}
	table = _tbtable + bandBufferSize - 1;
	for (int32_t i = 0; i <= s1Length; ++i)
	{
		*table = ALIGN_DIR_STOP;
		table += bandBufferSize;
	}
	/*compute profile*/
	int8_t* p;
	bufferSize = (s2Length + 1) * 5;
	if (bufferSize > _profileSize)
	{
		if (_profile)
		{
			delete[] _profile;
		}
		_profileSize = bufferSize * 2;
		_profile = new int8_t[_profileSize];
		if (_profile == NULL)
		{
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}
	p = _profile;
	for (int32_t i = 0; i < 5; ++i)
	{ /*row*/
		mat = _smat[i];
		p++;
		for (int32_t j = 0; j < s2Length; j++)
		{ /*column*/
			*p++ = mat[s2[j]];
		}
	}

	/*initialize the vectors*/
	for (int32_t i = leftd; i <= rightd; ++i)
	{
		_heBuffer[i].x = 0;
		_heBuffer[i].y = -_gopen;
	}

	_heBuffer[rightd + 1].x = ALIGN_MIN_SCORE;
	_heBuffer[rightd + 1].y = ALIGN_MIN_SCORE;
	_heBuffer[leftd - 1].x = ALIGN_MIN_SCORE;
	_heBuffer[leftd - 1].y = -_gopen;

	/*start the core CigarAlign loop*/
	score = 0; /*the best score*/
	cibest = cjbest = 0; /*the INVALID coordinate for the best score*/
	for (int i = lowrow + 1; i <= hirow; ++i)
	{
		if (leftd > 1)
		{
			--leftd;
		}

		if (i > s2Length - up)
		{
			--rightd;
		}

		/*calculate the E value*/
		h = _heBuffer[leftd + 1].x - _goe;
		e = _heBuffer[leftd + 1].y - _gext;
		if (h > e)
		{
			e = h;
		}
		//mat = _smat[s1[i - 1]];
		mat = _profile + s1[i - 1] * (s2Length + 1);
		if ((jtrans = leftd + low + i - 1) > 0)
		{
			h = _heBuffer[leftd].x + mat[jtrans];
			dir = ALIGN_DIR_DIAGONAL;
		}

		if (e > h)
		{
			h = e;
			dir = ALIGN_DIR_UP;
		}
		if (h < 0)
		{
			h = 0;
			dir = ALIGN_DIR_STOP;
		}

		f = h - _gopen;
		_heBuffer[leftd].x = h;
		_heBuffer[leftd].y = e;

		/*save the score and its coordinate*/
		if (h > score)
		{
			score = h;
			cibest = i;
			cjbest = leftd;
		}

		/*save the trace-back coordinate*/
		table = _tbtable + i * bandBufferSize;
		table[leftd] = dir;

		/*for the other cells in the i-th row*/
		for (int32_t curd = leftd + 1; curd <= rightd; ++curd)
		{
			if ((h = h - _goe) > (f = f - _gext))
			{
				f = h;
			}
			if ((h = _heBuffer[curd + 1].x - _goe)
					> (e = _heBuffer[curd + 1].y - _gext))
			{
				e = h;
			}

			dir = ALIGN_DIR_DIAGONAL;
			jtrans = curd + low + i - 1;
			h = _heBuffer[curd].x + mat[jtrans];
			if (f > h)
			{
				h = f;
				dir = ALIGN_DIR_LEFT;
			}
			if (e > h)
			{
				h = e;
				dir = ALIGN_DIR_UP;
			}
			if (h < 0)
			{
				h = 0;
				dir = ALIGN_DIR_STOP;
			}

			_heBuffer[curd].x = h;
			_heBuffer[curd].y = e;
			if (h > score)
			{
				score = h;
				cibest = i;
				cjbest = curd;
			}
			//Utils::log("%d ", score);
			/*save the trace-back coordinate*/
			table[curd] = dir;
			//Utils::log("i: %d j: %d dir:%d mat: %d h %d score %d\n", i, curd + low + i - 1, dir, mat[s2[jtrans - 1]], h, score);
		}
	}

	//Utils::log("cibest: %d cjbest: %d score: %d\n", cibest, cjbest + low + cibest - 1, score);
	//check the availability of the local CigarAlign*/
	if (cibest == 0 && cjbest == 0)
	{
		return aligns;
	}

	/*get the alignment*/
	aligns.reserve(2);
	switch (which)
	{
	case 1: /*the first sequence*/
		aligns.push_back(
				align2cigar(s1, s2, s1Length, s2Length, _tbtable,
						bandBufferSize, cibest, cjbest, score, low, true));
#ifdef ALIGN_DEBUG
		traceback(s1, s2, s1Length, s2Length, _tbtable, bandBufferSize, cibest, cjbest, score, low);
#endif
		break;
	case 2: /*the second sequence*/
		aligns.push_back(
				align2cigar(s1, s2, s1Length, s2Length, _tbtable,
						bandBufferSize, cibest, cjbest, score, low, false));
#ifdef ALIGN_DEBUG
		traceback(s1, s2, s1Length, s2Length, _tbtable, bandBufferSize, cibest, cjbest, score, low);
#endif
		break;

	default: /*both the first and the second sequences*/
		aligns.push_back(
				align2cigar(s1, s2, s1Length, s2Length, _tbtable,
						bandBufferSize, cibest, cjbest, score, low, true));
#ifdef ALIGN_DEBUG
		traceback(s1, s2, s1Length, s2Length, _tbtable, bandBufferSize, cibest, cjbest, score, low);
#endif
		aligns.push_back(
				align2cigar(s1, s2, s1Length, s2Length, _tbtable,
						bandBufferSize, cibest, cjbest, score, low, false));
#ifdef ALIGN_DEBUG
		traceback(s1, s2, s1Length, s2Length, _tbtable, bandBufferSize, cibest, cjbest, score, low);
#endif
		break;
	}

	return aligns;
}
int32_t Aligner::traceback(uint8_t* s1, uint8_t* s2, int32_t s1Length,
		int32_t s2Length, int8_t* tbtable, int32_t bandBufferSize,
		int32_t cibest, int32_t cjbest, int32_t alignScore, int32_t low)
{
	bool done = false;
	int32_t dir;
	uint8_t c1, c2;
	int32_t ti, tj;
	int32_t row = cibest;
	;
	int32_t column = cjbest;
	int32_t alignLength = 0;
	int32_t numMismatches = 0;
	int32_t numGaps = 0;
	uint8_t *s1align, *s2align, *smalign;

	/*trace-back from the cell with the best score to get the banded optimal local CigarAlign*/
	s1align = new uint8_t[s1Length + bandBufferSize];
	if (s1align == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	s2align = new uint8_t[s2Length + bandBufferSize];
	if (s2align == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	smalign = new uint8_t[s2Length + bandBufferSize];
	if (smalign == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
#if 0
	Utils::log("-------------------------s1-----------------\n");
	for(int i = 0; i < s1Length; ++i)
	{
		Utils::log("%c", "ACGTN"[s1[i]]);
	}
	Utils::log("\n");

	Utils::log("-------------------------s2-----------------\n");
	for(int i = 0; i < s2Length; ++i)
	{
		Utils::log("%c", "ACGTN"[s2[i]]);
	}
	Utils::log("\n");
	Utils::log("------------------------------------------\n");
#endif

	/*trace back to get the CigarAlign*/
	while (!done)
	{
		/*get the transformed coordinate in the CigarAlign matrix*/
		ti = row - 1;
		tj = (column + low + row - 1) - 1;

		/*check the direction of the alignment*/
		dir = tbtable[row * bandBufferSize + column];
		switch (dir)
		{
		case ALIGN_DIR_STOP:
			done = true;
			break;

		case ALIGN_DIR_DIAGONAL:
			/*save the aligned bases*/
			c1 = s1[ti];
			c2 = s2[tj];
			s1align[alignLength] = c1;
			s2align[alignLength] = c2;

			/*increase the number of mismatches*/
			if (c1 != c2)
			{
				++numMismatches;
				smalign[alignLength] = ' ';
			}
			else
			{
				smalign[alignLength] = '|';
			}
			++alignLength;

			/*adjusting the row and column*/
			--row;
			break;

		case ALIGN_DIR_UP: /*a deletion in s1*/
			c1 = s1[ti];
			s1align[alignLength] = c1;
			s2align[alignLength] = GAP_BASE;
			smalign[alignLength] = ' ';
			++alignLength;

			/*increase the number gaps*/
			++numGaps;

			/*adjust the row and column*/
			--row;
			++column;

			break;
		case ALIGN_DIR_LEFT: /*an insertion in s2*/
			c2 = s2[tj];
			s1align[alignLength] = GAP_BASE;
			s2align[alignLength] = c2;
			smalign[alignLength] = ' ';
			++alignLength;

			/*increase the number of gaps*/
			++numGaps;

			/*adjust the row and column*/
			--column;
			break;

		default:
			Utils::exit(
					"Unexpected value (%d) while tracing back the CigarAlign\n",
					dir);
			break;
		}
	}

	/*NO local CigarAlign*/
	if (alignLength == 0)
	{
		delete[] smalign;
		delete[] s2align;
		delete[] s1align;
		return 0;
	}

	/*print out the CigarAlign*/
	int32_t numLines = (alignLength + 59) / 60;
	for (int32_t line = 0, index = alignLength - 1; line < numLines;
			++line, index -= 60)
	{
		for (int32_t i = 0; i < 60 && index - i >= 0; ++i)
		{
			fputc(decode(s1align[index - i]), stderr);
		}
		fputc('\n', stderr);
		for (int32_t i = 0; i < 60 && index - i >= 0; ++i)
		{
			fputc(smalign[index - i], stderr);
		}
		fputc('\n', stderr);
		for (int32_t i = 0; i < 60 && index - i >= 0; ++i)
		{
			fputc(decode(s2align[index - i]), stderr);
		}
		fputc('\n', stderr);
	}

	delete[] smalign;
	delete[] s2align;
	delete[] s1align;

	Utils::log("#mismatches: %d #gaps: %d #alignScore: %d #alignLength: %d\n",
			numMismatches, numGaps, alignScore, alignLength);
	return numMismatches + numGaps;
}
CigarAlign* Aligner::align2cigar(uint8_t* s1, uint8_t* s2, int32_t s1Length,
		int32_t s2Length, int8_t* tbtable, int32_t bandBufferSize,
		int32_t cibest, int32_t cjbest, int32_t alignScore, int32_t low,
		bool rowwiseQuery)
{
	bool done = false;
	int32_t dir;
	uint8_t c1, c2;
	int32_t ti, tj;
	int32_t row = cibest, prow = cibest;
	int32_t column = cjbest, pcolumn = cjbest;
	int32_t alignLength = 0;
	int32_t numMismatches = 0;
	int32_t numGaps = 0;
	uint8_t op = ' ';
	uint8_t lastOp = ALIGN_DIR_STOP;
	uint32_t numOps = 0;
	CigarAlign* align;
	uint32_t* cigar;
	int32_t cigarNum, cigarSize;
	int32_t start1, end1, start2, end2;

	/*allocate space for cigar for s1, assuming that each entry takes 8 bytes*/
	cigarNum = 0;
	cigarSize = 128;
	cigar = new uint32_t[cigarSize];
	if (cigar == NULL)
	{
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	/*trace back to get the CigarAlign*/
	while (!done)
	{
		/*get the transformed coordinate in the CigarAlign matrix*/
		ti = row - 1;
		tj = (column + low + row - 1) - 1;

		/*check the direction of the alignment*/
		dir = tbtable[row * bandBufferSize + column];
		switch (dir)
		{
		case ALIGN_DIR_STOP:
			done = true;
			/*record the last operations*/
			if (numOps > 0)
			{
				/*resize the memory*/
				if (cigarNum >= cigarSize)
				{
					cigarSize += 128;
					uint32_t* buffer = new uint32_t[cigarSize];
					if (buffer == NULL)
					{
						Utils::exit(
								"Memory allocation failed in function %s line %d\n",
								__FUNCTION__, __LINE__);
					}
					memcpy(buffer, cigar, cigarNum * sizeof(cigar[0]));
					delete[] cigar;
					cigar = buffer;
				}
				cigar[cigarNum++] = (numOps << 2) | op;
			}
			break;

		case ALIGN_DIR_DIAGONAL:
			/*save the aligned bases*/
			c1 = s1[ti];
			c2 = s2[tj];
			++alignLength;

			/*increase the number of mismatches*/
			if (c1 != c2)
			{
				++numMismatches;
			}
			/*check the operation type*/
			if (lastOp == ALIGN_DIR_DIAGONAL)
			{
				++numOps;
			}
			else
			{
				/*save the previous operation*/
				if (numOps > 0)
				{
					/*resize the memory*/
					if (cigarNum >= cigarSize)
					{
						cigarSize += 1024;
						uint32_t* buffer = new uint32_t[cigarSize];
						if (buffer == NULL)
						{
							Utils::exit(
									"Memory allocation failed in function %s line %d\n",
									__FUNCTION__, __LINE__);
						}
						memcpy(buffer, cigar, cigarNum * sizeof(cigar[0]));
						delete[] cigar;
						cigar = buffer;
					}
					cigar[cigarNum++] = (numOps << 2) | op;
				}

				/*change to a new operation*/
				numOps = 1;
				op = ALIGN_OP_M;
				lastOp = ALIGN_DIR_DIAGONAL;
			}

			/*adjusting the row and column*/
			prow = row;
			pcolumn = column;
			--row;
			break;

		case ALIGN_DIR_UP: /*an insertion in s1*/
			++alignLength;

			/*increase the number gaps*/
			++numGaps;

			/*check the operation type*/
			if (lastOp == ALIGN_DIR_UP)
			{
				++numOps;
			}
			else
			{
				/*save the previous operation*/
				if (numOps > 0)
				{
					/*resize the memory*/
					if (cigarNum >= cigarSize)
					{
						cigarSize += 1024;
						uint32_t* buffer = new uint32_t[cigarSize];
						if (buffer == NULL)
						{
							Utils::exit(
									"Memory allocation failed in function %s line %d\n",
									__FUNCTION__, __LINE__);
						}
						memcpy(buffer, cigar, cigarNum * sizeof(cigar[0]));
						delete[] cigar;
						cigar = buffer;
					}
					cigar[cigarNum++] = (numOps << 2) | op;
				}

				/*change to a new operation*/
				numOps = 1;
				op = ALIGN_OP_I;
				lastOp = ALIGN_DIR_UP;
			}

			/*adjust the row and column*/
			prow = row;
			pcolumn = column;
			--row;
			++column;

			break;
		case ALIGN_DIR_LEFT: /*a deletion in s1*/
			++alignLength;

			/*increase the number of gaps*/
			++numGaps;

			/*check the operation type*/
			if (lastOp == ALIGN_DIR_LEFT)
			{
				++numOps;
			}
			else
			{
				/*save the previous operation*/
				if (numOps > 0)
				{
					/*resize the memory*/
					if (cigarNum >= cigarSize)
					{
						cigarSize += 1024;
						uint32_t* buffer = new uint32_t[cigarSize];
						if (buffer == NULL)
						{
							Utils::exit(
									"Memory allocation failed in function %s line %d\n",
									__FUNCTION__, __LINE__);
						}
						memcpy(buffer, cigar, cigarNum * sizeof(cigar[0]));
						delete[] cigar;
						cigar = buffer;
					}
					cigar[cigarNum++] = (numOps << 2) | op;
				}

				/*change to a new operation*/
				numOps = 1;
				op = ALIGN_OP_D;
				lastOp = ALIGN_DIR_LEFT;
			}

			/*adjust the row and column*/
			prow = row;
			pcolumn = column;
			--column;
			break;

		default:
			Utils::exit(
					"Unexpected value (%d) while tracing back the CigarAlign\n",
					dir);
			break;
		}
	}

	/*NO local CigarAlign*/
	if (alignLength == 0)
	{
		delete[] cigar;
		return NULL;
	}

	/*create an cigar-based CigarAlign object for s1*/
	start1 = prow - 1;
	start2 = (pcolumn + low + prow - 1) - 1;
	end1 = cibest - 1;
	end2 = (cjbest + low + cibest - 1) - 1;

	if (rowwiseQuery == true)
	{
		/*create alignment for the first sequence*/
		align = new CigarAlign(alignLength, start1, end1, start2, end2,
				numMismatches + numGaps, cigar, cigarNum, alignScore,
				rowwiseQuery);
		//Utils::log("rowwiseQuery %d start1 %d end1 %d start2 %d end2 %d\n", rowwiseQuery, start1, end1, start2, end2);
	}
	else
	{
		/*create alignment for the second sequence*/
		align = new CigarAlign(alignLength, start2, end2, start1, end1,
				numMismatches + numGaps, cigar, cigarNum, alignScore,
				rowwiseQuery);
		//Utils::log("rowwiseQuery %d start1 %d end1 %d start2 %d end2 %d\n", rowwiseQuery, start2, end2, start1, end1);
	}

	/*release buffer*/
	delete[] cigar;

	return align;
}




/*private member variables and functions for SSE2*/
#define N16_CHANNELS 	16	/*16 parallel lanes*/
#define N8_CHANNELS		8	/*8 parallel lanes*/
#define CDEPTH 			4

static inline void dprofile_fill7(BYTE * dprofile, BYTE * score_matrix,
			BYTE * dseq)
	{
		__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
		__m128i xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;

		// 4 x 16 db symbols
		// ca 188 x 4 = 752 instructions

		for (int j = 0; j < CDEPTH; j++)
		{
			unsigned d[N16_CHANNELS];
			for (int i = 0; i < N16_CHANNELS; i++)
				d[i] = dseq[j * N16_CHANNELS + i] << 5;

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + d[0]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + d[2]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + d[4]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + d[6]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + d[8]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + d[10]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + d[12]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + d[14]));

			xmm0 = _mm_unpacklo_epi8(xmm0, *(__m128i *) (score_matrix + d[1]));
			xmm2 = _mm_unpacklo_epi8(xmm2, *(__m128i *) (score_matrix + d[3]));
			xmm4 = _mm_unpacklo_epi8(xmm4, *(__m128i *) (score_matrix + d[5]));
			xmm6 = _mm_unpacklo_epi8(xmm6, *(__m128i *) (score_matrix + d[7]));
			xmm8 = _mm_unpacklo_epi8(xmm8, *(__m128i *) (score_matrix + d[9]));
			xmm10 = _mm_unpacklo_epi8(xmm10,
					*(__m128i *) (score_matrix + d[11]));
			xmm12 = _mm_unpacklo_epi8(xmm12,
					*(__m128i *) (score_matrix + d[13]));
			xmm14 = _mm_unpacklo_epi8(xmm14,
					*(__m128i *) (score_matrix + d[15]));

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 320), xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 448), xmm15);

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[0]));
			xmm1 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[1]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[2]));
			xmm3 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[3]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[4]));
			xmm5 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[5]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[6]));
			xmm7 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[7]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[8]));
			xmm9 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[9]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[10]));
			xmm11 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[11]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[12]));
			xmm13 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[13]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[14]));
			xmm15 = _mm_loadl_epi64((__m128i *) (score_matrix + 8 + d[15]));

			xmm0 = _mm_unpacklo_epi8(xmm0, xmm1);
			xmm2 = _mm_unpacklo_epi8(xmm2, xmm3);
			xmm4 = _mm_unpacklo_epi8(xmm4, xmm5);
			xmm6 = _mm_unpacklo_epi8(xmm6, xmm7);
			xmm8 = _mm_unpacklo_epi8(xmm8, xmm9);
			xmm10 = _mm_unpacklo_epi8(xmm10, xmm11);
			xmm12 = _mm_unpacklo_epi8(xmm12, xmm13);
			xmm14 = _mm_unpacklo_epi8(xmm14, xmm15);

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 320), xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 512 + 448), xmm15);

			xmm0 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[0]));
			xmm2 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[2]));
			xmm4 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[4]));
			xmm6 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[6]));
			xmm8 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[8]));
			xmm10 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[10]));
			xmm12 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[12]));
			xmm14 = _mm_loadl_epi64((__m128i *) (score_matrix + 16 + d[14]));

			xmm0 = _mm_unpacklo_epi8(xmm0,
					*(__m128i *) (score_matrix + 16 + d[1]));
			xmm2 = _mm_unpacklo_epi8(xmm2,
					*(__m128i *) (score_matrix + 16 + d[3]));
			xmm4 = _mm_unpacklo_epi8(xmm4,
					*(__m128i *) (score_matrix + 16 + d[5]));
			xmm6 = _mm_unpacklo_epi8(xmm6,
					*(__m128i *) (score_matrix + 16 + d[7]));
			xmm8 = _mm_unpacklo_epi8(xmm8,
					*(__m128i *) (score_matrix + 16 + d[9]));
			xmm10 = _mm_unpacklo_epi8(xmm10,
					*(__m128i *) (score_matrix + 16 + d[11]));
			xmm12 = _mm_unpacklo_epi8(xmm12,
					*(__m128i *) (score_matrix + 16 + d[13]));
			xmm14 = _mm_unpacklo_epi8(xmm14,
					*(__m128i *) (score_matrix + 16 + d[15]));

			xmm1 = xmm0;
			xmm0 = _mm_unpacklo_epi16(xmm0, xmm2);
			xmm1 = _mm_unpackhi_epi16(xmm1, xmm2);
			xmm5 = xmm4;
			xmm4 = _mm_unpacklo_epi16(xmm4, xmm6);
			xmm5 = _mm_unpackhi_epi16(xmm5, xmm6);
			xmm9 = xmm8;
			xmm8 = _mm_unpacklo_epi16(xmm8, xmm10);
			xmm9 = _mm_unpackhi_epi16(xmm9, xmm10);
			xmm13 = xmm12;
			xmm12 = _mm_unpacklo_epi16(xmm12, xmm14);
			xmm13 = _mm_unpackhi_epi16(xmm13, xmm14);

			xmm2 = xmm0;
			xmm0 = _mm_unpacklo_epi32(xmm0, xmm4);
			xmm2 = _mm_unpackhi_epi32(xmm2, xmm4);
			xmm6 = xmm1;
			xmm1 = _mm_unpacklo_epi32(xmm1, xmm5);
			xmm6 = _mm_unpackhi_epi32(xmm6, xmm5);
			xmm10 = xmm8;
			xmm8 = _mm_unpacklo_epi32(xmm8, xmm12);
			xmm10 = _mm_unpackhi_epi32(xmm10, xmm12);
			xmm14 = xmm9;
			xmm9 = _mm_unpacklo_epi32(xmm9, xmm13);
			xmm14 = _mm_unpackhi_epi32(xmm14, xmm13);

			xmm3 = xmm0;
			xmm0 = _mm_unpacklo_epi64(xmm0, xmm8);
			xmm3 = _mm_unpackhi_epi64(xmm3, xmm8);
			xmm7 = xmm2;
			xmm2 = _mm_unpacklo_epi64(xmm2, xmm10);
			xmm7 = _mm_unpackhi_epi64(xmm7, xmm10);
			xmm11 = xmm1;
			xmm1 = _mm_unpacklo_epi64(xmm1, xmm9);
			xmm11 = _mm_unpackhi_epi64(xmm11, xmm9);
			xmm15 = xmm6;
			xmm6 = _mm_unpacklo_epi64(xmm6, xmm14);
			xmm15 = _mm_unpackhi_epi64(xmm15, xmm14);

			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 0), xmm0);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 64), xmm3);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 128), xmm2);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 192), xmm7);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 256), xmm1);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 320),
					xmm11);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 384), xmm6);
			_mm_store_si128((__m128i *) (dprofile + 16 * j + 1024 + 448),
					xmm15);
		}
	}
#ifdef HAVE_SSSE3
static inline void dprofile_shuffle7(BYTE * dprofile, BYTE * score_matrix,
			BYTE * dseq_byte)
	{
		__m128i a, b, c, d, x, y, m0, m1, m2, m3, m4, m5, m6, m7;
		__m128i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;
		__m128i u0, u1, u2, u3, u4, u5, u8, u9, u10, u11, u12, u13;
		__m128i * dseq = (__m128i *) dseq_byte;

		// 16 x 4 = 64 db symbols
		// ca 458 instructions

		// make masks

		x = _mm_set_epi8(0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
				0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10);

		y = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
				0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80);

		a = _mm_load_si128(dseq);
		t0 = _mm_and_si128(a, x);
		t1 = _mm_slli_epi16(t0, 3);
		t2 = _mm_xor_si128(t1, y);
		m0 = _mm_or_si128(a, t1);
		m1 = _mm_or_si128(a, t2);

		b = _mm_load_si128(dseq + 1);
		t3 = _mm_and_si128(b, x);
		t4 = _mm_slli_epi16(t3, 3);
		t5 = _mm_xor_si128(t4, y);
		m2 = _mm_or_si128(b, t4);
		m3 = _mm_or_si128(b, t5);

		c = _mm_load_si128(dseq + 2);
		u0 = _mm_and_si128(c, x);
		u1 = _mm_slli_epi16(u0, 3);
		u2 = _mm_xor_si128(u1, y);
		m4 = _mm_or_si128(c, u1);
		m5 = _mm_or_si128(c, u2);

		d = _mm_load_si128(dseq + 3);
		u3 = _mm_and_si128(d, x);
		u4 = _mm_slli_epi16(u3, 3);
		u5 = _mm_xor_si128(u4, y);
		m6 = _mm_or_si128(d, u4);
		m7 = _mm_or_si128(d, u5);

		/* Note: pshufb only on modern Intel cpus (SSSE3), not AMD */
		/* SSSE3: Supplemental SSE3 */

#define profline(j)                                   \
  t6  = _mm_load_si128((__m128i*)(score_matrix)+2*j);   \
  t7  = _mm_load_si128((__m128i*)(score_matrix)+2*j+1); \
  t8  = _mm_shuffle_epi8(t6, m0); \
  t9  = _mm_shuffle_epi8(t7, m1);  \
  t10 = _mm_shuffle_epi8(t6, m2); \
  t11 = _mm_shuffle_epi8(t7, m3); \
  u8  = _mm_shuffle_epi8(t6, m4); \
  u9  = _mm_shuffle_epi8(t7, m5);  \
  u10 = _mm_shuffle_epi8(t6, m6); \
  u11 = _mm_shuffle_epi8(t7, m7); \
  t12 = _mm_or_si128(t8,  t9); \
  t13 = _mm_or_si128(t10, t11); \
  u12 = _mm_or_si128(u8,  u9); \
  u13 = _mm_or_si128(u10, u11); \
  _mm_store_si128((__m128i*)(dprofile)+4*j,   t12); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+1, t13); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+2, u12); \
  _mm_store_si128((__m128i*)(dprofile)+4*j+3, u13)

		profline(0);
		profline(1);
		profline(2);
		profline(3);
		profline(4);
		profline(5);
		profline(6);
		profline(7);
		profline(8);
		profline(9);
		profline(10);
		profline(11);
		profline(12);
		profline(13);
		profline(14);
		profline(15);
		profline(16);
		profline(17);
		profline(18);
		profline(19);
		profline(20);
		profline(21);
		profline(22);
		profline(23);

		//  dprofile_dump7(dprofile);
	}
#else

#define dprofile_shuffle7(dprofile, score_matrix, dseq_byte) dprofile_fill7(dprofile, score_matrix, dseq_byte)

#endif	//HAVE_SSSE3


// Register usage
// rdi:   hep
// rsi:   qp
// rdx:   Qm
// rcx:   Rm
// r8:    ql
// r9:    Sm/Mm

// rax:   x, temp
// r10:   ql2
// r11:   qi
// xmm0:  H0
// xmm1:  H1
// xmm2:  H2
// xmm3:  H3
// xmm4:  F0
// xmm5:  F1
// xmm6:  F2
// xmm7:  F3
// xmm8:  N0
// xmm9:  N1
// xmm10: N2
// xmm11: N3
// xmm12: E
// xmm13: S
// xmm14: Q 
// xmm15: R

#define INITIALIZE7					    \
                 "        movq      %0, rax             \n" \
		 "        movdqa    (rax), xmm13        \n" \
		 "        movdqa    (%3), xmm14         \n" \
		 "        movdqa    (%4), xmm15         \n" \
		 "        movq      %6, rax             \n" \
		 "        movdqa    (rax), xmm0         \n" \
		 "        movdqa    xmm0, xmm1          \n" \
		 "        movdqa    xmm0, xmm2          \n" \
		 "        movdqa    xmm0, xmm3          \n" \
		 "        movdqa    xmm0, xmm4          \n" \
		 "        movdqa    xmm0, xmm5          \n" \
		 "        movdqa    xmm0, xmm6          \n" \
		 "        movdqa    xmm0, xmm7          \n" \
		 "        movq      %5, r12             \n" \
		 "        shlq      $3, r12             \n" \
		 "        movq      r12, r10            \n" \
		 "        andq      $-16, r10           \n" \
		 "        xorq      r11, r11            \n" 

#define ONESTEP7(H, N, F, V)	         		    \
                 "        paddsb    "V"(rax), "H"       \n" \
                 "        pmaxub    "F", "H"            \n" \
                 "        pmaxub    xmm12, "H"          \n" \
                 "        pmaxub    "H", xmm13          \n" \
		 "        psubsb    xmm15, "F"          \n" \
		 "        psubsb    xmm15, xmm12        \n" \
		 "        movdqa    "H", "N"            \n" \
		 "        psubsb    xmm14, "H"          \n" \
		 "        pmaxub    "H", xmm12          \n" \
		 "        pmaxub    "H", "F"            \n"

static inline void donormal7(__m128i * Sm, __m128i * hep, __m128i ** qp,
			__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm)
	{
#ifdef DEBUG
		printf("donormal\n");
		printf("Sm=%p\n", Sm);
		printf("hep=%p\n", hep);
		printf("qp=%p\n", qp);
		printf("Qm=%p\n", Qm);
		printf("Rm=%p\n", Rm);
		printf("qlen=%ld\n", ql);
		printf("Zm=%p\n", Zm);
#endif

		__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE7
				"        jmp       2f                  \n"

				"1:      movq      0(%2,r11,1), rax    \n" // load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8   \n"// load N0
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4) \n"// save E
				"        movq      8(%2,r11,1), rax    \n"// load x from qp[qi+1]
				"        movdqa    32(%1,r11,4), xmm0  \n"// load H0
				"        movdqa    48(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP7("xmm9", "xmm2", "xmm5", "16")
				ONESTEP7("xmm10", "xmm3", "xmm6", "32")
				ONESTEP7("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4) \n"// save E
				"        addq      $16, r11            \n"// qi++
				"2:      cmpq      r11, r10            \n"// qi = ql4 ?
				"        jne       1b                  \n"// loop

				"4:      cmpq      r11, r12            \n"
				"        je        3f                  \n"
				"        movq      0(%2,r11,1), rax    \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:
				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
				"xmm4", "xmm5", "xmm6", "xmm7",
				"xmm8", "xmm9", "xmm10", "xmm11",
				"xmm12", "xmm13", "xmm14", "xmm15",
				"rax", "r10", "r11", "r12",
				"cc"
		);
	}

static inline void domasked7(__m128i * Sm, __m128i * hep, __m128i ** qp,
			__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm, __m128i * Mm)
	{

#ifdef DEBUG
		printf("domasked\n");
		printf("Sm=%p\n", Sm);
		printf("hep=%p\n", hep);
		printf("qp=%p\n", qp);
		printf("Qm=%p\n", Qm);
		printf("Rm=%p\n", Rm);
		printf("qlen=%ld\n", ql);
		printf("Zm=%p\n", Zm);
		printf("Mm=%p\n", Mm);
#endif

#if 1
		__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE7
				"        paddsb    (%7), xmm13          \n" // mask
				"        jmp       2f                   \n"

				"1:      movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8    \n"// load N0
				"        paddsb    (%7), xmm8           \n"// mask
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"        movq      8(%2,r11,1), rax     \n"// load x from qp[qi+1]
				"        movdqa    32(%1,r11,4), xmm0   \n"// load H0
				"        paddsb    (%7), xmm0           \n"// mask
				"        movdqa    48(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP7("xmm9", "xmm2", "xmm5", "16")
				ONESTEP7("xmm10", "xmm3", "xmm6", "32")
				ONESTEP7("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4)  \n"// save E
				"        addq      $16, r11             \n"// qi++
				"2:      cmpq      r11, r10             \n"// qi = ql4 ?
				"        jne       1b                   \n"// loop

				"        cmpq      r11, r12             \n"
				"        je        3f                   \n"
				"        movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsb    (%7), xmm12          \n"// mask

				ONESTEP7("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP7("xmm1", "xmm10", "xmm5", "16")
				ONESTEP7("xmm2", "xmm11", "xmm6", "32")
				ONESTEP7("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:

				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm),
				"r"(Mm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
				"xmm4", "xmm5", "xmm6", "xmm7",
				"xmm8", "xmm9", "xmm10", "xmm11",
				"xmm12", "xmm13", "xmm14", "xmm15",
				"rax", "r10", "r11", "r12",
				"cc"
		);
#endif

	}

#define INITIALIZE16					    \
                 "        movq      %0, rax             \n" \
		 "        movdqa    (rax), xmm13        \n" \
		 "        movdqa    (%3), xmm14         \n" \
		 "        movdqa    (%4), xmm15         \n" \
		 "        movq      %6, rax             \n" \
		 "        movdqa    (rax), xmm0         \n" \
		 "        movdqa    xmm0, xmm1          \n" \
		 "        movdqa    xmm0, xmm2          \n" \
		 "        movdqa    xmm0, xmm3          \n" \
		 "        movdqa    xmm0, xmm4          \n" \
		 "        movdqa    xmm0, xmm5          \n" \
		 "        movdqa    xmm0, xmm6          \n" \
		 "        movdqa    xmm0, xmm7          \n" \
		 "        shlq      $3, %5              \n" \
		 "        movq      %5, r10             \n" \
		 "        andq      $-16, r10           \n" \
		 "        xorq      r11, r11            \n"

#define ONESTEP16(H, N, F, V)	         		    \
                 "        paddsw    "V"(rax), "H"       \n" \
                 "        pmaxsw    "F", "H"            \n" \
                 "        pmaxsw    xmm12, "H"          \n" \
                 "        pmaxsw    "H", xmm13          \n" \
		 "        psubsw    xmm15, "F"          \n" \
		 "        psubsw    xmm15, xmm12        \n" \
		 "        movdqa    "H", "N"            \n" \
		 "        psubsw    xmm14, "H"          \n" \
		 "        pmaxsw    "H", xmm12          \n" \
		 "        pmaxsw    "H", "F"            \n" \

static inline void donormal16(__m128i * Sm, /* r9  */
	__m128i * hep, /* rdi */
	__m128i ** qp, /* rsi */
	__m128i * Qm, /* rdx */
	__m128i * Rm, /* rcx */
	long ql, /* r8  */
	__m128i * Zm)
	{
		__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE16
				"        jmp       2f                  \n"

				"1:      movq      0(%2,r11,1), rax    \n" // load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8   \n"// load N0
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP16("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP16("xmm1", "xmm10", "xmm5", "16")
				ONESTEP16("xmm2", "xmm11", "xmm6", "32")
				ONESTEP16("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movq      8(%2,r11,1), rax    \n"// load x from qp[qi+1]
				"        movdqa    xmm12, 16(%1,r11,4) \n"// save E
				"        movdqa    32(%1,r11,4), xmm0  \n"// load H0
				"        movdqa    48(%1,r11,4), xmm12 \n"// load E

				ONESTEP16("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP16("xmm9", "xmm2", "xmm5", "16")
				ONESTEP16("xmm10", "xmm3", "xmm6", "32")
				ONESTEP16("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4) \n"// save E
				"        addq      $16, r11            \n"// qi++
				"2:      cmpq      r11, r10            \n"// qi = ql4 ?
				"        jne       1b                  \n"// loop

				"        cmpq      r11, %5             \n"
				"        je        3f                  \n"
				"        movq      0(%2,r11,1), rax    \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12 \n"// load E

				ONESTEP16("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP16("xmm1", "xmm10", "xmm5", "16")
				ONESTEP16("xmm2", "xmm11", "xmm6", "32")
				ONESTEP16("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        shrq      $3, %5               \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:
				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
				"xmm4", "xmm5", "xmm6", "xmm7",
				"xmm8", "xmm9", "xmm10", "xmm11",
				"xmm12", "xmm13", "xmm14", "xmm15",
				"rax", "r10", "r11", "cc"
		);
	}

static inline void domasked16(__m128i * Sm, __m128i * hep, __m128i ** qp,
			__m128i * Qm, __m128i * Rm, long ql, __m128i * Zm, __m128i * Mm)
	{
		__asm__
		__volatile__(".att_syntax noprefix    # Change assembler syntax \n"
				INITIALIZE16
				"        paddsw    (%7), xmm13          \n" // add M
				"        jmp       2f                   \n"

				"1:      movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    0(%1,r11,4), xmm8    \n"// load N0
				"        paddsw    (%7), xmm8           \n"// add M
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsw    (%7), xmm12          \n"// add M

				ONESTEP16("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP16("xmm1", "xmm10", "xmm5", "16")
				ONESTEP16("xmm2", "xmm11", "xmm6", "32")
				ONESTEP16("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"        movq      8(%2,r11,1), rax     \n"// load x from qp[qi+1]
				"        movdqa    32(%1,r11,4), xmm0   \n"// load H0
				"        paddsw    (%7), xmm0           \n"// add M
				"        movdqa    48(%1,r11,4), xmm12  \n"// load E
				"        paddsw    (%7), xmm12          \n"// add M

				ONESTEP16("xmm8", "xmm1", "xmm4", "0" )
				ONESTEP16("xmm9", "xmm2", "xmm5", "16")
				ONESTEP16("xmm10", "xmm3", "xmm6", "32")
				ONESTEP16("xmm11", "32(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 48(%1,r11,4)  \n"// save E
				"        addq      $16, r11             \n"// qi++
				"2:      cmpq      r11, r10             \n"// qi = ql4 ?
				"        jne       1b                   \n"// loop

				"        cmpq      r11, %5              \n"
				"        je        3f                   \n"
				"        movq      0(%2,r11,1), rax     \n"// load x from qp[qi]
				"        movdqa    16(%1,r11,4), xmm12  \n"// load E
				"        paddsw    (%7), xmm12          \n"// add M

				ONESTEP16("xmm0", "xmm9", "xmm4", "0" )
				ONESTEP16("xmm1", "xmm10", "xmm5", "16")
				ONESTEP16("xmm2", "xmm11", "xmm6", "32")
				ONESTEP16("xmm3", "0(%1,r11,4)", "xmm7", "48")

				"        movdqa    xmm12, 16(%1,r11,4)  \n"// save E
				"3:      movq      %0, rax              \n"// save S
				"        movdqa    xmm13, (rax)         \n"
				"        shrq      $3, %5               \n"
				"        .att_syntax prefix      # Change back to standard syntax"

				:

				: "m"(Sm), "r"(hep),"r"(qp), "r"(Qm), "r"(Rm), "r"(ql), "m"(Zm),
				"r"(Mm)

				: "xmm0", "xmm1", "xmm2", "xmm3",
				"xmm4", "xmm5", "xmm6", "xmm7",
				"xmm8", "xmm9", "xmm10", "xmm11",
				"xmm12", "xmm13", "xmm14", "xmm15",
				"rax", "r10", "r11", "cc"
		);
	}

static inline void dprofile_fill16(WORD * dprofile_word, WORD * score_matrix_word,
			BYTE * dseq)
	{
		__m128i xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
		__m128i xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
		__m128i xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23;
		__m128i xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30, xmm31;

		for (int j = 0; j < CDEPTH; j++)
		{
			int d[N8_CHANNELS];
			for (int z = 0; z < N8_CHANNELS; z++)
				d[z] = dseq[j * N8_CHANNELS + z] << 5;

			for (int i = 0; i < 24; i += 8)
			{
				xmm0 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[0] + i));
				xmm1 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[1] + i));
				xmm2 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[2] + i));
				xmm3 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[3] + i));
				xmm4 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[4] + i));
				xmm5 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[5] + i));
				xmm6 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[6] + i));
				xmm7 = _mm_load_si128(
						(__m128i *) (score_matrix_word + d[7] + i));

				xmm8 = _mm_unpacklo_epi16(xmm0, xmm1);
				xmm9 = _mm_unpackhi_epi16(xmm0, xmm1);
				xmm10 = _mm_unpacklo_epi16(xmm2, xmm3);
				xmm11 = _mm_unpackhi_epi16(xmm2, xmm3);
				xmm12 = _mm_unpacklo_epi16(xmm4, xmm5);
				xmm13 = _mm_unpackhi_epi16(xmm4, xmm5);
				xmm14 = _mm_unpacklo_epi16(xmm6, xmm7);
				xmm15 = _mm_unpackhi_epi16(xmm6, xmm7);

				xmm16 = _mm_unpacklo_epi32(xmm8, xmm10);
				xmm17 = _mm_unpackhi_epi32(xmm8, xmm10);
				xmm18 = _mm_unpacklo_epi32(xmm12, xmm14);
				xmm19 = _mm_unpackhi_epi32(xmm12, xmm14);
				xmm20 = _mm_unpacklo_epi32(xmm9, xmm11);
				xmm21 = _mm_unpackhi_epi32(xmm9, xmm11);
				xmm22 = _mm_unpacklo_epi32(xmm13, xmm15);
				xmm23 = _mm_unpackhi_epi32(xmm13, xmm15);

				xmm24 = _mm_unpacklo_epi64(xmm16, xmm18);
				xmm25 = _mm_unpackhi_epi64(xmm16, xmm18);
				xmm26 = _mm_unpacklo_epi64(xmm17, xmm19);
				xmm27 = _mm_unpackhi_epi64(xmm17, xmm19);
				xmm28 = _mm_unpacklo_epi64(xmm20, xmm22);
				xmm29 = _mm_unpackhi_epi64(xmm20, xmm22);
				xmm30 = _mm_unpacklo_epi64(xmm21, xmm23);
				xmm31 = _mm_unpackhi_epi64(xmm21, xmm23);

				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 0)
								+ N8_CHANNELS * j), xmm24);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 1)
								+ N8_CHANNELS * j), xmm25);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 2)
								+ N8_CHANNELS * j), xmm26);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 3)
								+ N8_CHANNELS * j), xmm27);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 4)
								+ N8_CHANNELS * j), xmm28);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 5)
								+ N8_CHANNELS * j), xmm29);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 6)
								+ N8_CHANNELS * j), xmm30);
				_mm_store_si128(
						(__m128i *) (dprofile_word
								+ CDEPTH * N8_CHANNELS * (i + 7)
								+ N8_CHANNELS * j), xmm31);
			}
		}
	}


#ifndef USE_FULL_SW_64
int32_t Aligner::search7(BYTE** qtable, BYTE gap_open_penalty,
		BYTE gap_extend_penalty, BYTE * score_matrix, BYTE * dprofile,
		BYTE * hearray, int32_t qlen, int32_t numSeqs, int32_t *seqOffsets,
		uint8_t *sequences, AlignScore* scores)
{

	int32_t maxScore = 0;
	__m128i S, Q, R, T, M, Z, T0;
	__m128i *hep, **qp;
	BYTE * d_begin[N16_CHANNELS];

	__m128i dseqalloc[CDEPTH];

	BYTE * dseq = (BYTE*) &dseqalloc;
	BYTE zero;

	long seq_id[N16_CHANNELS];
	long next_id = 0;
	int32_t done;

	memset(hearray, 0x80, qlen * 32);

	Z = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
			0x80, 0x80, 0x80, 0x80, 0x80, 0x80);
	T0 = _mm_set_epi8(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80);
	Q = _mm_set_epi8(gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty);
	R = _mm_set_epi8(gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty);
	zero = 0;
	done = 0;

	S = Z;

	hep = (__m128i *) hearray;
	qp = (__m128i **) qtable;

	for (int c = 0; c < N16_CHANNELS; c++)
	{
		d_begin[c] = &zero;
		seq_id[c] = -1;
	}

	int easy = 0;

	while (1)
	{
		if (easy)
		{
			// fill all channels

			for (int c = 0; c < N16_CHANNELS; c++)
			{
				for (int j = 0; j < CDEPTH; j++)
				{
					BYTE v = *(d_begin[c]);
					dseq[N16_CHANNELS * j + c] = v;
					if (v)
						d_begin[c]++;
				}
				if (!*(d_begin[c]))
					easy = 0;
			}

			if (_haveSSSE3)
			{
				dprofile_shuffle7(dprofile, score_matrix, dseq);
			}
			else
			{
				dprofile_fill7(dprofile, score_matrix, dseq);
			}

			donormal7(&S, hep, qp, &Q, &R, qlen, &Z);
		}
		else
		{
			// One or more sequences ended in the previous block
			// We have to switch over to a new sequence

			easy = 1;

			M = _mm_setzero_si128();
			T = T0;
			for (int c = 0; c < N16_CHANNELS; c++)
			{
				if (*(d_begin[c]))
				{
					// this channel has more sequence

					for (int j = 0; j < CDEPTH; j++)
					{
						BYTE v = *(d_begin[c]);
						dseq[N16_CHANNELS * j + c] = v;
						if (v)
							d_begin[c]++;
					}
					if (!*(d_begin[c]))
						easy = 0;
				}
				else
				{
					// sequence in channel c ended
					// change of sequence

					M = _mm_xor_si128(M, T);

					long cand_id = seq_id[c];

					if (cand_id >= 0)
					{
						// save score
						long score = ((BYTE*) &S)[c] - 0x80;
						scores[cand_id]._score = score;
						if (score > maxScore)
						{
							maxScore = score;
						}
						done++;
					}
					if (next_id < numSeqs)
					{
						// get next sequence
						seq_id[c] = next_id;
						d_begin[c] = sequences + seqOffsets[next_id];
						next_id++;

						// fill channel
						for (int j = 0; j < CDEPTH; j++)
						{
							BYTE v = *(d_begin[c]);
							dseq[N16_CHANNELS * j + c] = v;
							if (v)
								d_begin[c]++;
						}
						if (!*(d_begin[c]))
							easy = 0;
					}
					else
					{
						// no more sequences, empty channel
						seq_id[c] = -1;
						d_begin[c] = &zero;
						for (int j = 0; j < CDEPTH; j++)
							dseq[N16_CHANNELS * j + c] = 0;
					}

				}

				T = _mm_slli_si128(T, 1);
			}

			if (done == numSeqs)
				break;

			if (_haveSSSE3)
			{
				dprofile_shuffle7(dprofile, score_matrix, dseq);
			}
			else
			{
				dprofile_fill7(dprofile, score_matrix, dseq);
			}

			domasked7(&S, hep, qp, &Q, &R, qlen, &Z, &M);
		}
	}
	return maxScore;
}
void Aligner::search16(WORD** qtable, WORD gap_open_penalty,
		WORD gap_extend_penalty, WORD * score_matrix, WORD * dprofile,
		WORD * hearray, int32_t qlen, int32_t numSeqs, int32_t *seqOffsets,
		uint8_t *sequences, AlignScore* scores)
{

	__m128i S, Q, R, T, M, Z, T0;
	__m128i *hep, **qp;
	BYTE * d_begin[N8_CHANNELS];

	__m128i dseqalloc[CDEPTH];

	BYTE * dseq = (BYTE *) &dseqalloc;
	BYTE zero;

	int32_t seq_id[N8_CHANNELS];
	int32_t next_id = 0;
	int32_t done;

	Z = _mm_set_epi16(0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
			0x8000);
	T0 = _mm_set_epi16(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
			0x8000);
	Q = _mm_set_epi16(gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty, gap_open_penalty,
			gap_open_penalty, gap_open_penalty);
	R = _mm_set_epi16(gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty,
			gap_extend_penalty, gap_extend_penalty, gap_extend_penalty);

	zero = 0;
	done = 0;

	S = Z;

	hep = (__m128i *) hearray;
	qp = (__m128i **) qtable;

	for (int a = 0; a < qlen; a++)
	{
		hep[2 * a] = Z;
		hep[2 * a + 1] = Z;
	}

	for (int c = 0; c < N8_CHANNELS; c++)
	{
		d_begin[c] = &zero;
		seq_id[c] = -1;
	}

	int easy = 0;

	while (1)
	{
		if (easy)
		{
			for (int c = 0; c < N8_CHANNELS; c++)
			{
				for (int j = 0; j < CDEPTH; j++)
				{
					BYTE v = *(d_begin[c]);
					dseq[N8_CHANNELS * j + c] = v;
					if (v)
						d_begin[c]++;
				}
				if (!*(d_begin[c]))
					easy = 0;
			}

			dprofile_fill16(dprofile, score_matrix, dseq);

			donormal16(&S, hep, qp, &Q, &R, qlen, &Z);

		}
		else
		{

			easy = 1;

			M = _mm_setzero_si128();
			T = T0;

			for (int c = 0; c < N8_CHANNELS; c++)
			{
				if (*(d_begin[c]))
				{
					for (int j = 0; j < CDEPTH; j++)
					{
						BYTE v = *(d_begin[c]);
						dseq[N8_CHANNELS * j + c] = v;
						if (v)
							d_begin[c]++;
					}

					if (!*(d_begin[c]))
						easy = 0;

				}
				else
				{
					M = _mm_xor_si128(M, T);

					long cand_id = seq_id[c];

					if (cand_id >= 0)
					{
						int score = ((WORD*) &S)[c] - 0x8000;
						/*save the alignment score*/
						scores[cand_id]._score = score;
						done++;
					}
#ifndef ONLY_8CHANNEL
					/*find the next non-processed sequence*/
					for (;
							next_id < numSeqs
									&& scores[next_id]._score < _scorelimit7;
							++next_id)
					{
						done++;
					}
#endif
					if (next_id < numSeqs)
					{
						seq_id[c] = next_id;
						d_begin[c] = sequences + seqOffsets[next_id];
						next_id++;

						for (int j = 0; j < CDEPTH; j++)
						{
							BYTE v = *(d_begin[c]);
							dseq[N8_CHANNELS * j + c] = v;
							if (v)
								d_begin[c]++;
						}
						if (!*(d_begin[c]))
							easy = 0;
					}
					else
					{
						seq_id[c] = -1;
						d_begin[c] = &zero;
						for (int j = 0; j < CDEPTH; j++)
							dseq[N8_CHANNELS * j + c] = 0;
					}
				}
				T = _mm_slli_si128(T, 2);
			}

			if (done == numSeqs)
				break;

			dprofile_fill16(dprofile, score_matrix, dseq);

			domasked16(&S, hep, qp, &Q, &R, qlen, &Z, &M);
		}
	}
}
#else
long Aligner::fullsw(long gap_open_penalty, long gap_extend_penalty, long* score_matrix, long* hearray,
      uint8_t* dseq, uint8_t * dend, uint8_t* qseq, uint8_t* qend)
{
  long h, n, e, f, s;
  long *hep;
 	uint8_t *qp, *dp;
  long * sp;

  s = 0;
  dp = dseq;
  memset(hearray, 0, 2 * sizeof(long) * (qend-qseq));

  while (dp < dend)
    {
      f = 0;
      h = 0;
      hep = hearray;
      qp = qseq;
      sp = score_matrix + (*dp << 5);

      while (qp < qend)
        {
          n = *hep;
          e = *(hep+1);
          h += sp[*qp];

          if (e > h)
            h = e;
          if (f > h)
            h = f;
          if (h < 0)
            h = 0;
          if (h > s)
            s = h;

          *hep = h;
          e -= gap_extend_penalty;
          f -= gap_extend_penalty;
          h -= gap_open_penalty;

          if (h > e)
            e = h;
          if (h > f)
            f = h;

          *(hep+1) = e;
          h = n;
          hep += 2;
          qp++;
        }

      dp++;
    }
  return s;
}
#endif




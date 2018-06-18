

/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#include "hardwarearchitecturecpusse2.h"

#include <QtCore/QTime>
#include "blosum.h"
#include <string.h>

#define maxx(a, b) ( (a) > (b) ) ? (a) : (b)

#define SHORT_BIAS 32768

HardwareArchitectureCPUSSE2::HardwareArchitectureCPUSSE2(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : HardwareArchitectureAPI(lib, tbu, num, ofs, sz, sc, endpos), lastSubstMatrix(""), matrix(NULL)
{
	//conversione libreria per uso SSE2
	lib_sse2 = new unsigned char[tbu];

	for (unsigned j=0; j<tbu; ++j) {
		char c = lib[j];

		if (!c || c == '@')
			lib_sse2[j] = 255;
		else {
			int value = AMINO_ACID_VALUE[c];
			if (value < 0)
				throw string("Unknown amino acid: " + c);
			lib_sse2[j] = (unsigned char) value;
		}
	}
}

HardwareArchitectureCPUSSE2::~HardwareArchitectureCPUSSE2() {
	if ( matrix )
		delete [] matrix;

	delete []lib_sse2;
}

void HardwareArchitectureCPUSSE2::freeDevice( )
{
}

unsigned HardwareArchitectureCPUSSE2::calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug ) {
	if (lastSubstMatrix != subMat) {
		if (subMat == "BL50") 
			matrix = readMatrix(cpu_Farrar_bl50);
		else if (subMat == "BL62")
			matrix = readMatrix(cpu_Farrar_bl62);
		else if (subMat == "BL90")
			matrix = readMatrix(cpu_Farrar_bl90);
		else
			matrix = readMatrix(cpu_Farrar_dna1);
	}

	lastSubstMatrix = subMat;

	unsigned mc = 0;

	unsigned char seqSearch[MAX_SEARCHED_SEQUENCE_LENGTH];

	for (unsigned j=1; j<sizeNotPad; ++j) {
		char c = strToAlign[j];

		if (!c || c == '@')
			//lib_sse2[j] = 255;			///???????????????????????????????????
			seqSearch[j-1] = 255;
		else {
			int value = AMINO_ACID_VALUE[c];
			if (value < 0)
				throw string("Unknown amino acid");

			seqSearch[j-1] = (unsigned char) value;
		}
	}

	QTime tott;
	int time=0;

	tott.start();

	void *swData = initSWsse2(seqSearch, sizeNotPad-1, matrix);

	SwStripedData *stripedData = (SwStripedData *) swData;

	for(unsigned j=startPos; j<=stopPos; ++j) {
		scores[j] = 0;

		scores[j] = swSse2Byte(sizeNotPad-1, lib_sse2+(offsets[j])+1, sizes[j]-1, alpha+beta, beta, stripedData->pvbQueryProf, stripedData->pvH1, stripedData->pvH2, stripedData->pvE, stripedData->bias);

		if ( scores[j] >= 255 ) {
			scores[j] = swSse2Word(sizeNotPad-1, lib_sse2+(offsets[j])+1, sizes[j]-1, alpha+beta, beta, stripedData->pvsQueryProf, stripedData->pvH1, stripedData->pvH2, stripedData->pvE);
		}
	}

	time = tott.elapsed();

	printf("CPU elapsed: %7.3f (s)\n", time/1000.0);

	//calcolo megacups
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	time = (time > 1) ? time : 1;

	mc = static_cast<unsigned>((static_cast<double>(seqLibSize) / time) * (sizeNotPad/1048.576));

	return mc;
}

unsigned HardwareArchitectureCPUSSE2::getAvailableDevicesNumber()
{
	return 1;
}

HardwareArchitectureAPI *HardwareArchitectureCPUSSE2::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos)
{
// 	if ( i >= getAvailableDevicesNumber() ) 
// 		throw string("cannot allocate more CPU devices than available");

	return new HardwareArchitectureCPUSSE2(lib, tbu, num, ofs, sz, sc, endpos);
}

char *skipSpaces (char *line)
{
	while (*line && isspace(*line)) {
		++line;
	}

	return line;
}

char * HardwareArchitectureCPUSSE2::readMatrix (const char *matString) {

	const unsigned BUF_SIZE = 70;
	char line[BUF_SIZE+1];

	char *matrix;

	int mark[ALPHA_SIZE];
	int order[ALPHA_SIZE];

	int done;
	unsigned i;

	unsigned count = 0;

	matrix = new char[ALPHA_SIZE * ALPHA_SIZE];
	memset(matrix, 0, ALPHA_SIZE * ALPHA_SIZE);

	if (!matrix)
		throw string("Unable to allocate memory for scoring matrix");

	/* initialize the order and mark arrays */
	for (i = 0; i < ALPHA_SIZE; ++i) {
		order[i] = -1;
		mark[i] = -1;
	}

    /* read the first line of the matrix giving the amino acid order */
	done = 0;
	while (!done) {
		strncpy (line, matString, BUF_SIZE);
		line[BUF_SIZE] = 0;
		char *ptr = skipSpaces (line);
		if (*ptr && *ptr != '#') {

			while (*ptr && *ptr != '#') {
				int inx = AMINO_ACID_VALUE[*ptr];

				if (inx == -1) {
					throw string("Unknown amino acid");
				} else if (mark[inx] != -1) {
					throw string("Amino acid defined twice");
				} else if (count >= ALPHA_SIZE) {
					throw string("Too many amino acids");
				} else {
					order[count++] = inx;
					mark[inx] = inx;
				}
				ptr = skipSpaces (ptr + 1);
			}

			done = 1;
		}
	}

	/* make sure all amino acids are defined */
	for (i = 0; i < ALPHA_SIZE; ++i) {
		if (order[i] < 0) {
			throw string("Missing column for amino acid");
		}
		mark[i] = -1;
	}

    /* read the scores for the amino acids */
	for(unsigned cnt=1; cnt<ALPHA_SIZE+1; ++cnt) {
    	strncpy (line, matString+(cnt*BUF_SIZE), BUF_SIZE);
		line[BUF_SIZE] = 0;
		char *row;
		char *ptr = skipSpaces (line);
		if (*ptr && *ptr != '#') {
			int inx = AMINO_ACID_VALUE[*ptr];
			if (inx == -1) {
				throw string("Unknown amino acid in matrix");
			} else if (mark[inx] != -1) {
				throw string("Row defined twice");
			}

			row = &matrix[inx * ALPHA_SIZE];

			for (i = 0; i < ALPHA_SIZE; ++i) {
				int sign = 1;
				int num = 0;

				ptr = skipSpaces (ptr + 1);

				/* check the sign */
				if (*ptr == '-') {
					sign = -1;
					++ptr;
				}

				do {
					if (*ptr >= '0' && *ptr <= '9') {
						num = num * 10 + (*ptr - '0');
						ptr++;
					} else {
						char name[16];
						char *pName;
						if (isspace (*ptr)) {
							pName = "space";
						} else if (*ptr == 0) {
							pName = "end of line";
						} else {
							name[0] = *ptr;
							name[1] = 0;
							pName = name;
						}
						throw string("Row expecting digit found");
					}
				} while (*ptr && !isspace (*ptr));

				num = num * sign;

				if (num < -128 || num > 127) {
					throw string("Weight out of range row");
					num = 0;
				}

				row[order[i]] = (char) num;
			}

			if (i < ALPHA_SIZE) {
				throw string("Amino acid row incomplete");
			}

			mark[inx] = 1;
		}

	}

	/* make sure all amino acids are defined */
	for (i = 0; i < ALPHA_SIZE; ++i) {
		if (mark[i] < 0)
			throw string("Missing row for amino acid");
	}

	return matrix;
}

void * HardwareArchitectureCPUSSE2::initSWsse2(unsigned char *querySeq, unsigned queryLength, char *matrix) {

	unsigned i, j, k;

	unsigned segSize;
	unsigned nCount;

	int bias;

	int lenQryByte;
	int lenQryShort;

	int weight;

	short *ps;
	char *pc;

	char *matrixRow;

	size_t aligned;

	SwStripedData *pSwData;

	lenQryByte = (queryLength + 15) / 16;
	lenQryShort = (queryLength + 7) / 8;

	pSwData = (SwStripedData *) malloc (sizeof (SwStripedData));
	if (!pSwData)
		throw string("Unable to allocate memory for SW data");

	nCount = 64 +                             /* slack bytes */
			lenQryByte * ALPHA_SIZE +        /* query profile byte */
			lenQryShort * ALPHA_SIZE +       /* query profile short */
			(lenQryShort * 3);               /* vH1, vH2 and vE */

	pSwData->pData = (unsigned char *) calloc (nCount, sizeof (__m128i));
	if (!pSwData->pData)
		throw string("Unable to allocate memory for SW data buffers");

	/* since we might port this to another platform, lets align the data */
	/* to 16 byte boundries ourselves */
	aligned = ((size_t) pSwData->pData + 15) & ~(0x0f);

	pSwData->pvbQueryProf = (__m128i *) aligned;
	pSwData->pvsQueryProf = pSwData->pvbQueryProf + lenQryByte * ALPHA_SIZE;

	pSwData->pvH1 = pSwData->pvsQueryProf + lenQryShort * ALPHA_SIZE;
	pSwData->pvH2 = pSwData->pvH1 + lenQryShort;
	pSwData->pvE  = pSwData->pvH2 + lenQryShort;

	/* Use a scoring profile for the SSE2 implementation, but the layout
	* is a bit strange.  The scoring profile is parallel to the query, but is
	* accessed in a stripped pattern.  The query is divided into equal length
	* segments.  The number of segments is equal to the number of elements
	* processed in the SSE2 register.  For 8-bit calculations, the query will
	* be divided into 16 equal length parts.  If the query is not long enough
	* to fill the last segment, it will be filled with neutral weights.  The
	* first element in the SSE register will hold a value from the first segment,
	* the second element of the SSE register will hold a value from the
	* second segment and so on.  So if the query length is 288, then each
	* segment will have a length of 18.  So the first 16 bytes will  have
	* the following weights: Q1, Q19, Q37, ... Q271; the next 16 bytes will
	* have the following weights: Q2, Q20, Q38, ... Q272; and so on until
	* all parts of all segments have been written.  The last seqment will
	* have the following weights: Q18, Q36, Q54, ... Q288.  This will be
	* done for the entire alphabet.
	*/

	/* Find the bias to use in the substitution matrix */
	bias = 127;
	for (i = 0; i < ALPHA_SIZE * ALPHA_SIZE; i++) {
		if (matrix[i] < bias) {
			bias = matrix[i];
		}
	}
	if (bias > 0) {
		bias = 0;
	}

	/* Fill in the byte query profile */
	pc = (char *) pSwData->pvbQueryProf;
	segSize = (queryLength + 15) / 16;
	nCount = segSize * 16;
	for (i = 0; i < ALPHA_SIZE; ++i) {
		matrixRow = matrix + i * ALPHA_SIZE;
		for (j = 0; j < segSize; ++j) {
			for (k = j; k < nCount; k += segSize) {
				if (k >= queryLength) {
					weight = 0;
				} else {
					weight = matrixRow[*(querySeq + k)];
				}
				*pc++ = (char) (weight - bias);
			}
		}
	}

	/* Fill in the short query profile */
	ps = (short *) pSwData->pvsQueryProf;
	segSize = (queryLength + 7) / 8;
	nCount = segSize * 8;
	for (i = 0; i < ALPHA_SIZE; ++i) {
		matrixRow = matrix + i * ALPHA_SIZE;
		for (j = 0; j < segSize; ++j) {
			for (k = j; k < nCount; k += segSize) {
				if (k >= queryLength) {
					weight = 0;
				} else {
					weight = matrixRow[*(querySeq + k)];
				}
				*ps++ = (unsigned short) weight;
			}
		}
	}

	pSwData->bias = (unsigned short) -bias;

	return pSwData;
}

int HardwareArchitectureCPUSSE2::swSse2Byte( unsigned queryLength, unsigned char *dbSeq, unsigned dbLength, unsigned short gapOpenOrig, unsigned short gapExtend, __m128i *pvQueryProf, __m128i *pvHLoad, __m128i *pvHStore, __m128i *pvE, unsigned short bias)
{
	unsigned i, j;
	int score;

	int dup;
	int cmp;
	unsigned iter = (queryLength + 15) / 16;
	
	unsigned short gapOpenFarrar = gapOpenOrig - gapExtend;
	
	__m128i *pv;

	__m128i vE, vF, vH;

	__m128i vMaxScore;
	__m128i vBias;
	__m128i vGapOpen;
	__m128i vGapExtend;

	__m128i vTemp;
	__m128i vZero;

	__m128i *pvScore;


	/* Load the bias to all elements of a constant */
	dup    = (bias << 8) | (bias & 0x00ff);
	vBias = _mm_insert_epi16 (vBias, dup, 0);
	vBias = _mm_shufflelo_epi16 (vBias, 0);
	vBias = _mm_shuffle_epi32 (vBias, 0);

	/* Load gap opening penalty to all elements of a constant */
	dup    = (gapOpenFarrar << 8) | (gapOpenFarrar & 0x00ff);
	vGapOpen = _mm_insert_epi16 (vGapOpen, dup, 0);
	vGapOpen = _mm_shufflelo_epi16 (vGapOpen, 0);
	vGapOpen = _mm_shuffle_epi32 (vGapOpen, 0);

	/* Load gap extension penalty to all elements of a constant */
	dup    = (gapExtend << 8) | (gapExtend & 0x00ff);
	vGapExtend = _mm_insert_epi16 (vGapExtend, dup, 0);
	vGapExtend = _mm_shufflelo_epi16 (vGapExtend, 0);
	vGapExtend = _mm_shuffle_epi32 (vGapExtend, 0);

	vMaxScore = _mm_xor_si128 (vMaxScore, vMaxScore);

	vZero = _mm_xor_si128 (vZero, vZero);

	/* Zero out the storage vector */
	for (i = 0; i < iter; i++)
	{
		_mm_store_si128 (pvE + i, vMaxScore);
		_mm_store_si128 (pvHStore + i, vMaxScore);
	}

	for (i = 0; i < dbLength; ++i)
	{
		/* fetch first data asap. */
		pvScore = pvQueryProf + dbSeq[i] * iter;

		/* zero out F. */
		vF = _mm_xor_si128 (vF, vF);

		/* load the next h value */
		vH = _mm_load_si128 (pvHStore + iter - 1);
		vH = _mm_slli_si128 (vH, 1);

		pv = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pv;

		for (j = 0; j < iter; j++)
		{
			/* load values of vF and vH from previous row (one unit up) */
			vE = _mm_load_si128 (pvE + j);

			/* add score to vH */
			vH = _mm_adds_epu8 (vH, *(pvScore + j));
			vH = _mm_subs_epu8 (vH, vBias);

			/* Update highest score encountered this far */
			vMaxScore = _mm_max_epu8 (vMaxScore, vH);

			/* get max from vH, vE and vF */
			vH = _mm_max_epu8 (vH, vE);
			vH = _mm_max_epu8 (vH, vF);

			/* save vH values */
			_mm_store_si128 (pvHStore + j, vH);

			/* update vE value */
			vH = _mm_subs_epu8 (vH, vGapOpen);
			vE = _mm_subs_epu8 (vE, vGapExtend);
			vE = _mm_max_epu8 (vE, vH);

			/* update vF value */
			vF = _mm_subs_epu8 (vF, vGapExtend);
			vF = _mm_max_epu8 (vF, vH);

			/* save vE values */
			_mm_store_si128 (pvE + j, vE);

			/* load the next h value */
			vH = _mm_load_si128 (pvHLoad + j);
		}

		/* reset pointers to the start of the saved data */
		j = 0;
		vH = _mm_load_si128 (pvHStore + j);

		/*  the computed vF value is for the given column.  since */
		/*  we are at the end, we need to shift the vF value over */
		/*  to the next column. */
		vF = _mm_slli_si128 (vF, 1);
		vTemp = _mm_subs_epu8 (vH, vGapOpen);
		vTemp = _mm_subs_epu8 (vF, vTemp);
		vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
		cmp  = _mm_movemask_epi8 (vTemp);


		while (cmp != 0xffff)
		//for (unsigned cnt=0; cnt<iter; ++cnt)
		{
			vE = _mm_load_si128 (pvE + j);

			vH = _mm_max_epu8 (vH, vF);

			/* save vH values */
			_mm_store_si128 (pvHStore + j, vH);

			/*  update vE incase the new vH value would change it */
			vH = _mm_subs_epu8 (vH, vGapOpen);
			vE = _mm_max_epu8 (vE, vH);
			_mm_store_si128 (pvE + j, vE);

			/* update vF value */
			vF = _mm_subs_epu8 (vF, vGapExtend);

			j++;
			if (j >= iter)
			{
				j = 0;
				vF = _mm_slli_si128 (vF, 1);
			}

			vH = _mm_load_si128 (pvHStore + j);

			vTemp = _mm_subs_epu8 (vH, vGapOpen);
			vTemp = _mm_subs_epu8 (vF, vTemp);
			vTemp = _mm_cmpeq_epi8 (vTemp, vZero);
			cmp  = _mm_movemask_epi8 (vTemp);
		}
	}

	/* find largest score in the vMaxScore vector */
	vTemp = _mm_srli_si128 (vMaxScore, 8);
	vMaxScore = _mm_max_epu8 (vMaxScore, vTemp);
	vTemp = _mm_srli_si128 (vMaxScore, 4);
	vMaxScore = _mm_max_epu8 (vMaxScore, vTemp);
	vTemp = _mm_srli_si128 (vMaxScore, 2);
	vMaxScore = _mm_max_epu8 (vMaxScore, vTemp);
	vTemp = _mm_srli_si128 (vMaxScore, 1);
	vMaxScore = _mm_max_epu8 (vMaxScore, vTemp);

	/* store in temporary variable */
	score = _mm_extract_epi16 (vMaxScore, 0);
	score = score & 0x00ff;

	/*  check if we might have overflowed */
	if (score + bias >= 255)
	{
		score = 255;
	}

	/* return largest score */
	return score;
}


int HardwareArchitectureCPUSSE2::swSse2Word(unsigned queryLength, unsigned char *dbSeq, unsigned dbLength, unsigned short gapOpenOrig, unsigned short gapExtend, __m128i *pvQueryProf, __m128i *pvHLoad, __m128i *pvHStore, __m128i *pvE) {

	unsigned i, j;
	int     score;

	int cmp;
	unsigned iter = (queryLength + 7) / 8;

	unsigned short gapOpenFarrar = gapOpenOrig - gapExtend;
	
	__m128i *pv;

	__m128i vE, vF, vH;

	__m128i vMaxScore;
	__m128i vGapOpen;
	__m128i vGapExtend;

	__m128i vMin;
	__m128i vMinimums;
	__m128i vTemp;

	__m128i *pvScore;


	/* Load gap opening penalty to all elements of a constant */
	vGapOpen = _mm_insert_epi16 (vGapOpen, gapOpenFarrar, 0);
	vGapOpen = _mm_shufflelo_epi16 (vGapOpen, 0);
	vGapOpen = _mm_shuffle_epi32 (vGapOpen, 0);

	//printf("queryLength=%u, dbLength=%u, gapOpen=%u, gapExtend=%u, iter=%u\n", queryLength, dbLength, gapOpen, gapExtend, iter);

	/* Load gap extension penalty to all elements of a constant */
	vGapExtend = _mm_insert_epi16 (vGapExtend, gapExtend, 0);
	vGapExtend = _mm_shufflelo_epi16 (vGapExtend, 0);
	vGapExtend = _mm_shuffle_epi32 (vGapExtend, 0);

	/*  load vMaxScore with the zeros.  since we are using signed */
	/*  math, we will bias the maxscore to -32768 so we have the */
	/*  full range of the short. */
	vMaxScore = _mm_cmpeq_epi16 (vMaxScore, vMaxScore);
	vMaxScore = _mm_slli_epi16 (vMaxScore, 15);

	vMinimums = _mm_shuffle_epi32 (vMaxScore, 0);

	vMin = _mm_shuffle_epi32 (vMaxScore, 0);
	vMin = _mm_srli_si128 (vMin, 14);

    /* Zero out the storage vector */
	for (i = 0; i < iter; ++i) {

		_mm_store_si128 (pvE + i, vMaxScore);
		_mm_store_si128 (pvHStore + i, vMaxScore);
	}

	for (i = 0; i < dbLength; ++i) {

		/* fetch first data asap. */
		pvScore = pvQueryProf + dbSeq[i] * iter;

		/* zero out F. */
		vF = _mm_cmpeq_epi16 (vF, vF);
		vF = _mm_slli_epi16 (vF, 15);

		/* load the next h value */
		vH = _mm_load_si128 (pvHStore + iter - 1);
		vH = _mm_slli_si128 (vH, 2);
		vH = _mm_or_si128 (vH, vMin);

		pv = pvHLoad;
		pvHLoad = pvHStore;
		pvHStore = pv;

		for (j = 0; j < iter; ++j) {

			/* load values of vF and vH from previous row (one unit up) */
			vE = _mm_load_si128 (pvE + j);

			/* add score to vH */
			vH = _mm_adds_epi16 (vH, *pvScore++);

			/* Update highest score encountered this far */
			vMaxScore = _mm_max_epi16 (vMaxScore, vH);

			/* get max from vH, vE and vF */
			vH = _mm_max_epi16 (vH, vE);
			vH = _mm_max_epi16 (vH, vF);

			/* save vH values */
			_mm_store_si128 (pvHStore + j, vH);

			/* update vE value */
			vH = _mm_subs_epi16 (vH, vGapOpen);
			vE = _mm_subs_epi16 (vE, vGapExtend);
			vE = _mm_max_epi16 (vE, vH);

			/* update vF value */
			vF = _mm_subs_epi16 (vF, vGapExtend);
			vF = _mm_max_epi16 (vF, vH);

			/* save vE values */
			_mm_store_si128 (pvE + j, vE);

			/* load the next h value */
			vH = _mm_load_si128 (pvHLoad + j);
		}

		/* reset pointers to the start of the saved data */
		j = 0;
		vH = _mm_load_si128 (pvHStore + j);

		/*  the computed vF value is for the given column.  since */
		/*  we are at the end, we need to shift the vF value over */
		/*  to the next column. */
		vF = _mm_slli_si128 (vF, 2);
		vF = _mm_or_si128 (vF, vMin);
		vTemp = _mm_subs_epi16 (vH, vGapOpen);
		vTemp = _mm_cmpgt_epi16 (vF, vTemp);
		cmp  = _mm_movemask_epi8 (vTemp);

		while (cmp != 0x0000) 
		//for (unsigned cnt=0; cnt<iter; ++cnt)
		{

			vE = _mm_load_si128 (pvE + j);

			vH = _mm_max_epi16 (vH, vF);

			/* save vH values */
			_mm_store_si128 (pvHStore + j, vH);

			/*  update vE incase the new vH value would change it */
			vH = _mm_subs_epi16 (vH, vGapOpen);
			vE = _mm_max_epi16 (vE, vH);
			_mm_store_si128 (pvE + j, vE);

			/* update vF value */
			vF = _mm_subs_epi16 (vF, vGapExtend);

			j++;
			if (j >= iter) {

				j = 0;
				vF = _mm_slli_si128 (vF, 2);
				vF = _mm_or_si128 (vF, vMin);
			}

			vH = _mm_load_si128 (pvHStore + j);

			vTemp = _mm_subs_epi16 (vH, vGapOpen);
			vTemp = _mm_cmpgt_epi16 (vF, vTemp);
			cmp  = _mm_movemask_epi8 (vTemp);
		}
	}

	/* find largest score in the vMaxScore vector */
	vTemp = _mm_srli_si128 (vMaxScore, 8);
	vMaxScore = _mm_max_epi16 (vMaxScore, vTemp);
	vTemp = _mm_srli_si128 (vMaxScore, 4);
	vMaxScore = _mm_max_epi16 (vMaxScore, vTemp);
	vTemp = _mm_srli_si128 (vMaxScore, 2);
	vMaxScore = _mm_max_epi16 (vMaxScore, vTemp);

	/* store in temporary variable */
	score = (short) _mm_extract_epi16 (vMaxScore, 0);

	/* return largest score */
	return score + SHORT_BIAS;
}

void HardwareArchitectureCPUSSE2::swComplete(void *pSwData)
{
	SwStripedData *pStripedData = (SwStripedData *) pSwData;

	free (pStripedData->pData);
	free (pStripedData);
}


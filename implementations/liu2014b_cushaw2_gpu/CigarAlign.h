/*
 * CigarAlign.h
 *
 *  Created on: Dec 30, 2011
 *      Author: yongchao
 */

#ifndef CIGARALIGN_H_
#define CIGARALIGN_H_
#include "Macros.h"
#include "Utils.h"
#include "Sequence.h"

/*alignment operations*/
class CigarAlign
{
public:
	CigarAlign(Sequence& seq);
	CigarAlign(int32_t alignLength, int32_t start1, int32_t end1,
			int32_t start2, int32_t end2, int32_t numErrors, uint32_t* cigar,
			int32_t ncigar, int32_t alignScore, bool rev);
	~CigarAlign();

	inline uint32_t* getCigar() {
		return _cigar;
	}
	inline int32_t getCigarLength() {
		return _ncigar;
	}
	inline int32_t getNumErrors() {
		return _numErrors;
	}
	inline float getIdentity() {
		/*return value in the range [0, 100)*/
		return ((float) (_alignLength - _numErrors) * 100) / _alignLength;
	}
	inline int32_t getAlignLength() {
		return _alignLength;
	}
	inline int32_t getAlignScore() {
		return _alignScore;
	}
	/*get the alignment information*/
	inline int32_t getNumBases1() {
		return _end1 - _start1 + 1;
	}
	inline int32_t getNumBases2() {
		return _end2 - _start2 + 1;
	}
	inline int32_t getStart() {
		return _start1;
	}
	inline int32_t getEnd() {
		return _end1;
	}
	inline int32_t getMateStart() {
		return _start2;
	}
	inline int32_t getMateEnd() {
		return _end2;
	}
	inline void getRegion(int32_t& start1, int32_t& start2, int32_t& end1,
			int32_t& end2) {
		start1 = _start1;
		start2 = _start2;
		end1 = _end1;
		end2 = _end2;
	}
	inline void cigarOut(FILE* file) {
		uint32_t cigar;
		for (int32_t i = 0; i < _ncigar; ++i) {
			cigar = _cigar[i];
			/*print the cigar to the file*/
			fprintf(file, "%d%c", cigar >> 2, _alignOpName[cigar & 3]);
		}
	}
	inline int32_t cigarOut(char* buffer) {
		int32_t offset = 0;
		uint32_t cigar;
		for (int32_t i = 0; i < _ncigar; ++i) {
			cigar = _cigar[i];
			/*print the cigar to the file*/
			offset += sprintf(buffer + offset, "%d%c", cigar >> 2, _alignOpName[cigar & 3]);
		}

		return offset;
	}
	/*extend the cigar to the full length of the sequence*/
	int extendCigar(uint8_t* s, int32_t length);

private:
	int32_t _start1, _end1; /*its self*/
	int32_t _start2, _end2; /*its mate*/
	int32_t _alignLength; /*alignment length*/
	int32_t _alignScore;
	int32_t _numErrors;

	uint32_t* _cigar;
	int32_t _ncigar;

	/*reverse operations for cigar*/
	static const uint8_t _alignRcOp[4];
	static const uint8_t _alignOpName[4];

	friend class SAM;
};

#endif /* CigarAlign_H_ */

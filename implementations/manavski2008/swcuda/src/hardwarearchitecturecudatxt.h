
#ifndef HARDWAREARCHITECTURECUDATXT_H
#define HARDWAREARCHITECTURECUDATXT_H

/**
	@author Svetlin Manavski <svetlin@manavski.com>
 */

#include "smithwaterman.h"

#include <math.h>
//per blocco si intende un quadrato con CHAR_PER_SEQ_PER_BLOCK caratteri di SEQ_PER_BLOCK sequenze (24x24 char)
const unsigned SEQ_PER_BLOCK 				= 64;
const unsigned CHAR_PER_SEQ_PER_BLOCK 		= 9;
const unsigned SQ_R_SEQ_PER_BLOCK 			= static_cast<unsigned> (sqrt(SEQ_PER_BLOCK));
const unsigned SQ_R_CHAR_PER_SEQ_PER_BLOCK 	= static_cast<unsigned> (sqrt(CHAR_PER_SEQ_PER_BLOCK));
const unsigned BLOCK_SIDE 					= 24;

struct seqBlock {
	char blockArr[BLOCK_SIDE][BLOCK_SIDE];
};


// ConcreteImplementor 2/3
class HardwareArchitectureCUDATxt : public HardwareArchitectureAPI {
public:
	void virtual freeDevice();

	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const unsigned debug);

	static unsigned getAvailableDevicesNumber();
	static HardwareArchitectureAPI *getDevice(unsigned i, seqBlock *linLib, const unsigned linLibSize, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc);

protected:
	HardwareArchitectureCUDATxt(const unsigned deviceNum, seqBlock *linLib, const unsigned linLibSize, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc);

private:
	unsigned device;
	seqBlock* linSeqLib;
	unsigned linSeqLibSize;
};

#endif


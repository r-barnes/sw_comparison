#ifndef HARDWAREARCHITECTURECPUSSE2_H
#define HARDWAREARCHITECTURECPUSSE2_H

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


/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/
/**
Questa classe utilizza il codice di Farrar per il calcolo di S-W su CPU.
Tale codice dava segmentation fault se lanciato su più query consecutive. Il problema non è stato mai risolto
*/


#include "smithwaterman.h"
#include "swsse2_def.h"

class HardwareArchitectureCPUSSE2 : public HardwareArchitectureAPI {
public:
	
	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug );

	virtual ~HardwareArchitectureCPUSSE2();
	void virtual freeDevice();

	static unsigned getAvailableDevicesNumber();

	static HardwareArchitectureAPI *getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

protected:
	HardwareArchitectureCPUSSE2(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

	char *readMatrix (const char *matString);

	void * initSWsse2(unsigned char *querySeq, unsigned queryLength, char *matrix);

	int swSse2Word(unsigned queryLength, unsigned char *dbSeq, unsigned dbLength, unsigned short gapOpen, unsigned short gapExtend, __m128i *pvQueryProf, __m128i *pvHLoad, __m128i *pvHStore, __m128i *pvE);

	int swSse2Byte( unsigned queryLength, unsigned char *dbSeq, unsigned dbLength, unsigned short gapOpen, unsigned short gapExtend, __m128i *pvQueryProf, __m128i *pvHLoad, __m128i *pvHStore, __m128i *pvE, unsigned short bias);

	void swComplete(void *pSwData);

	std::string lastSubstMatrix;
	char *matrix;

	unsigned char *lib_sse2;
private:
};

#endif

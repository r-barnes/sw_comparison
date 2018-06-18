#ifndef HARDWAREARCHITECTURECPU_H
#define HARDWAREARCHITECTURECPU_H


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
Classe che implementa il calcolo di S-W su CPU.
Il metodo che svolge il calcolo è calcSmithWaterman che usa la funzione sw_single.
*/

#include "smithwaterman.h"

// ConcreteImplementor 1/3 
class HardwareArchitectureCPU : public HardwareArchitectureAPI {
public:
	
	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug );

	virtual ~HardwareArchitectureCPU();
	void virtual freeDevice();

	static unsigned getAvailableDevicesNumber();

	static HardwareArchitectureAPI *getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

protected:
	HardwareArchitectureCPU(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

	int sbt(char A, char B);
	int sw_single( const char* seqA, const unsigned sizeA, const char* seqB, const unsigned sizeB, int alpha, int beta );
	int sw_single_with_endpos( const char* seqA, const unsigned sizeA, const char* seqB, const unsigned sizeB, int alpha, int beta, unsigned &endpos );

	int H[50*1024], F[50*1024];

	int substMatrix[32][32];
	std::string lastSubstMatrix;

private:

};

#endif

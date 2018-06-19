#ifndef HARDWAREARCHITECTURECUDA_H
#define HARDWAREARCHITECTURECUDA_H

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
Classe che svolge il calcolo di S-W su GPU.
Il metodo che realizza il calcolo è calcSmithWaterman che chiama l'handler smithWatermanCuda2.
*/

#include "smithwaterman.h"

// ConcreteImplementor 2/3
class HardwareArchitectureCUDA : public HardwareArchitectureAPI {
public:
	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug);

	static unsigned getAvailableDevicesNumber();
	static HardwareArchitectureAPI *getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

 	virtual ~HardwareArchitectureCUDA();
	void virtual freeDevice();

protected:
	HardwareArchitectureCUDA(const unsigned deviceNum, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

private:
	unsigned device;

 	char *dev_seqlib;
 	unsigned *dev_offsets;
 	unsigned *dev_sizes;
 	int *dev_scores;
	unsigned *dev_endpos;
	std::string lastSubstMatrix;
};


#endif
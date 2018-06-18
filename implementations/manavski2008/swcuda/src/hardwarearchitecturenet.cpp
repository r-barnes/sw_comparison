
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


#include "hardwarearchitecturenet.h"

#include <QtCore/QTime>

HardwareArchitectureNet::HardwareArchitectureNet(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : HardwareArchitectureAPI(lib, tbu, num, ofs, sz, sc, endpos)
{
}

void HardwareArchitectureNet::freeDevice( )
{
}

unsigned HardwareArchitectureNet::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug)
{
	unsigned mc = 0;

	QTime tott; tott.start();

	//trovare i megacups

	printf("NET elapsed: %d (ms)\n", tott.elapsed());

	return mc;
}

unsigned HardwareArchitectureNet::getAvailableDevicesNumber()
{
	return 0;
}

HardwareArchitectureAPI *HardwareArchitectureNet::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos)
{
	if ( i >= getAvailableDevicesNumber() ) 
		throw string("cannot allocate more NET devices than available");

	return new HardwareArchitectureNet(lib, tbu, num, ofs, sz, sc, endpos);
}


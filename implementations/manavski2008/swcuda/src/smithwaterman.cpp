
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


#include "smithwaterman.h"


#include <QtCore/QMutexLocker>
#include <iostream>



//------HardwareArchitectureAPI------------------------------------------------------------------------------------------------------------------
HardwareArchitectureAPI::HardwareArchitectureAPI(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : seqlib(lib), totBytesUsed(tbu), numSeqs(num), offsets(ofs), sizes(sz), scores(sc), end_positions(endpos)
{
}


HardwareArchitectureAPI::~HardwareArchitectureAPI()
{
}

unsigned HardwareArchitectureAPI::getAvailableDevicesNumber()
{
	return 0;
}

HardwareArchitectureAPI *HardwareArchitectureAPI::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc)
{
	throw string("unimplemented factory method HardwareArchitectureAPI::getDevice()");
}

//--end----HardwareArchitectureAPI-----------------------------------------------------------------------------------------------------------



//------AlignmentAlgorithm------------------------------------------------------------------------------------------------------------------

AlignmentAlgorithm::AlignmentAlgorithm(HardwareArchitectureAPI *p) : _hwapi(p), megaCUPS(0), isJobCompleted(false), isJobAvailable(false), mustQuit(false)
{
}

AlignmentAlgorithm::~AlignmentAlgorithm() 
{
}


void AlignmentAlgorithm::wait() 
{
	QMutexLocker mlock(&m_jobcomp);
	while (!isJobCompleted) {
		w_jobcomp.wait(&m_jobcomp);
	}
}

void AlignmentAlgorithm::quit() 
{
	m_jobav.lock();
	isJobAvailable = true;
	mustQuit = true;
	m_jobav.unlock();
	w_jobav.wakeAll();
}

unsigned AlignmentAlgorithm::getMegaCUPS() const {
	return megaCUPS;
}

//-end-----AlignmentAlgorithm------------------------------------------------------------------------------------------------------------------


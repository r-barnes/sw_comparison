

#ifndef SMITHWATERMAN_H
#define SMITHWATERMAN_H

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
La classe HardwareArchitectureAPI rappresenta la base di tutte le altre.
Qui viene impostato il design del multithreading dell'applicazione.
*/

#include <string>
using namespace std;

#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>

//############################Limitazioni############################//
//stabilisce la massima lunghezza delle sequenze da cercare
#define MAX_SEARCHED_SEQUENCE_LENGTH 2050
//stabilisce il numero massimo di caratteri usati per il contatore dei file di output (output_0, output_1 etc..). In questo caso il massimo numero di file in uscita e quindi di sequenze cercate è limitato a 1000000
#define MAX_NUM_SEQUENCE_ALIGNED_COUNTER 1000000
//stabilisce il massimo numero di caratteri usati per il nome delle sequenze nel databse nei file di output
#define MAX_NUM_CHAR 40
//############################Limitazioni############################//

// here we apply the "BRIDGE" Design pattern

//Implementor
class HardwareArchitectureAPI {

public:
	virtual ~HardwareArchitectureAPI();
	void virtual freeDevice() = 0;

	// returns the MegaCUPS based on the last transaction
	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug ) = 0;

	static unsigned getAvailableDevicesNumber();
	static HardwareArchitectureAPI *getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc);

protected:
	HardwareArchitectureAPI(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos);

	const char *seqlib;
	const unsigned totBytesUsed;
	const unsigned numSeqs;
	unsigned *offsets;
	unsigned *sizes;

	int *scores; // output
	// end-position of the local alignment for both the query and the subject
	// q_endpos = end_position & 0xFFFF, s_endpos = end_position >> 16;
	unsigned *end_positions; 

};



//Abstraction
class AlignmentAlgorithm : public QThread {
public:
	AlignmentAlgorithm(HardwareArchitectureAPI *p);
	virtual ~AlignmentAlgorithm();

	void wait();
	void quit();

	unsigned getMegaCUPS() const;

protected:

	HardwareArchitectureAPI * _hwapi;

	unsigned megaCUPS;

	bool isJobCompleted, isJobAvailable, mustQuit;
	QMutex m_jobcomp, m_jobav;
	QWaitCondition w_jobcomp, w_jobav;
};

//Refined Abstraction: define a SmithWaterman
class SmithWaterman : public AlignmentAlgorithm {
public:
	SmithWaterman(HardwareArchitectureAPI *p) : AlignmentAlgorithm (p), strToAlign(NULL), strToAlignSizeNotPad(0), alpha(0), beta(0), startPos(0), stopPos(0), sbstMat("") {
	}
	virtual ~SmithWaterman() {
	}

	void setJob(const char *sta, const unsigned size, const int a, const int b, const std::string mat, const unsigned start, const unsigned stop,  const bool calcEndpos, unsigned debug)  {
		m_jobcomp.lock();
		isJobCompleted = false, 
		m_jobcomp.unlock();
		strToAlign = sta;
		strToAlignSizeNotPad = size;
		alpha = a; 
		beta = b;
		sbstMat = mat;
		startPos = start;
		stopPos = stop;
		cEndp = calcEndpos;
		deb = debug;
		m_jobav.lock();
		isJobAvailable = true;
		m_jobav.unlock();
		w_jobav.wakeAll();
	}

	virtual void run() {
		while (true) {
			{
				QMutexLocker mlock(&m_jobav);
				while (!isJobAvailable) {
					w_jobav.wait(&m_jobav);
				}
				isJobAvailable = false;
			}
			if (mustQuit) {
				_hwapi->freeDevice();
				break;
			}
			megaCUPS = _hwapi->calcSmithWaterman( strToAlign, strToAlignSizeNotPad, alpha, beta, sbstMat, startPos, stopPos, cEndp, deb );
			m_jobcomp.lock();
			isJobCompleted = true;
			m_jobcomp.unlock();
			w_jobcomp.wakeAll();
		}
	}

private:
	const char *strToAlign;
	unsigned strToAlignSizeNotPad;
	int alpha, beta;
	unsigned startPos, stopPos;
	std::string sbstMat;
	bool cEndp;
	unsigned deb;

};

#endif

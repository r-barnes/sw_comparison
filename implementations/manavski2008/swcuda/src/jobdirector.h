
#ifndef JOBDIRECTOR_H
#define JOBDIRECTOR_H

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
E' la classe di base dell'applicazione che gestisce il load-balancing tra le risorse in gioco.
*/

#include <string>
#include <vector>
#include "smithwaterman.h"
#include "hardwarearchitecturecudatxt.h"
#include "bioconfig.h"
#include "inout.h"

/// struttura usata per raccogliere le info sui vari device disponibili
struct HWDevice {
	HWDevice() : megaCUPS(0), startPos(0), stopPos(0), device(NULL), sw(NULL), type("") {
	};
	~HWDevice() {
		if (device) delete device;
		if (sw) delete sw;
	}

	unsigned int megaCUPS;
	unsigned int startPos, stopPos;
	unsigned int bytesToDo;
	HardwareArchitectureAPI *device;
	SmithWaterman *sw;
	string type;
};


struct SWResults {
	
	SWResults(double sc, unsigned endpos) {
		score = sc;
		q_endpos = endpos & 0xFFFF; 
		s_endpos = (endpos >> 16) & 0xFFFF; 
	};
	
	double score;
	unsigned q_endpos;
	unsigned s_endpos;
};

class JobDirector {
public:
	JobDirector();
	~JobDirector();

	/// funzione di inizializzazione. Legge il DB (posto nel file dbpath) e il config (salvandolo in cf) e crea gli oggetti per gestire i vari dispositivi
	void init(const std::string dbpath, const BioConfig &cf);

	unsigned getSequenceCount() const;

	/** 
	smithWatermanMultiSeq calcola più query in sequenza chiamando per ognuna smithWatermanDyn.
	*/
	/**
	const char * strToAlign - query
	const int alpha - penalty per apertura gap
	const int beta - penalty per estensione gap
	const std::string subMat - matrice di sostituzione
	const BioConfig &cf - oggetto che conserva il config
	const unsigned startPos - numero di sequenza del DB da cui bisogna partire
	const unsigned stopPos - numero di sequenza del DB a cui bisogna fermarsi
	string &searchedStringName - nome della query
	string &outFiles - nome del file di output
	unsigned seqPos - numero della sequenza del DB con cui si sta facendo l'allineamento
	unsigned alignOffsets - parametro che permette di partire con l'allineamento da una sequenza del DB che non sia la prima
	*/
	void smithWatermanDyn(const char * strToAlign, const int alpha, const int beta, const std::string subMat, const BioConfig &cf, const unsigned startPos, const unsigned stopPos, string &searchedStringName, string &outFiles, unsigned seqPos, unsigned alignOffsets);

	/**
	const std::string &seqFileName - file della query
	const string &libFileName - file della libreria
	const int alpha - penalty per apertura gap
	const int beta - penalty per estensione gap
	const std::string subMat - matrice di sostituzione
	const BioConfig &cf - oggetto che conserva il config
	const unsigned startPos - numero di sequenza del DB da cui bisogna partire
	const unsigned stopPos - numero di sequenza del DB a cui bisogna fermarsi
	unsigned alignOffsets - parametro che permette di partire con l'allineamento da una sequenza del DB che non sia la prima
	*/
	void smithWatermanMultiSeq(const std::string &seqFileName, const string &libFileName, const int alpha, const int beta, const std::string subMat, const BioConfig &cf, const unsigned startPos, const unsigned stopPos, unsigned alignOffsets);

protected:

	void clear();
	void scoresToFile(const std::string &outFile, const string &searchedStringName, const BioConfig &cf, const unsigned seqPos);

	/** 
	come input utilizza i valori di HWDevice::megaCUPS per ricalcolare
	le posizioni di start e stop per tutti i device disponibili
	*/
	unsigned repartition(const unsigned startPos, const unsigned stopPos, const BioConfig &cf);

	char *seqlib;
	
	unsigned *offsets;
	unsigned *sizes;
	unsigned *sizesPad;
	int* scores;
	unsigned *end_positions;

	unsigned numSeqs;
	unsigned totBytesUsed;
	std::vector<std::string> seqNamesOrdered;
	std::vector<std::string> seqNamesNotOrdered;

private:
	std::vector<HWDevice> devs;

	unsigned numCalls;
	std::string dbName;
	double normFact;
};

#endif

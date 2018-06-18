


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


// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
using namespace std;

#include "jobdirector.h"

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

/**
Main. Dopo la lettura del config, delle opzioni da linea di comando e il caricamento del DB, il jobDirector chiama smithWatermanMultiSeq che per ogni query usa smithWatermanDyn.
*/
int main( int argc, char** argv) {
	try {
		//verifyMAC();

		///file di configurazione e run
		BioConfig cf("config.ini");

		string seqFile, libFile;
		
		///gestione input
		unsigned alignOffsets = commandLineManager(argc, argv, seqFile, libFile);

		///controllo il formato dei file di input
		cout << endl;
		cout << "Checking the input files......." << endl << endl;
		fastaVerifier(seqFile);
		fastaVerifier(libFile);
		cout << "Input files in fasta format." << endl << endl;

		cout << endl;
		cout << "-----------------------------------------------------------------------" << endl;
		cout << "Smith-Waterman GPU engine, v.1.92 for Cuda 1.1" << endl;
		cout << "Copyright (c) 2006 - 2008 by Svetlin Manavski, CRIBI, University of Padova" << endl;
		cout << "-----------------------------------------------------------------------" << endl << endl;
	
		///avvio del jobDirector
		JobDirector jd;
		jd.init(libFile, cf);
		cout << "-------------------------------------------------------------------------------------" << endl << endl;
		jd.smithWatermanMultiSeq( seqFile, libFile, cf.getGapFirst(), cf.getGapNext(), cf.getMatValue(), cf, 0, jd.getSequenceCount()-1, alignOffsets);

	} catch ( string &ex) {
		cout << "exception occurred: " <<  ex << endl;
	} catch ( QString &ex) {
		cout << "exception occurred: " <<  ex.toStdString() << endl;
	}

	return EXIT_SUCCESS;
}


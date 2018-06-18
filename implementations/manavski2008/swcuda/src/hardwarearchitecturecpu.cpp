
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

#include "hardwarearchitecturecpu.h"

#include <QtCore/QTime>
#include "blosum.h"
#include <string.h>

#define maxx(a, b) ( (a) > (b) ) ? (a) : (b)

//------HardwareArchitectureCPU------------------------------------------------------------------------------------------------------------------
HardwareArchitectureCPU::HardwareArchitectureCPU(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : HardwareArchitectureAPI(lib, tbu, num, ofs, sz, sc, endpos), lastSubstMatrix("")
{
}

HardwareArchitectureCPU::~HardwareArchitectureCPU()
{
}

void HardwareArchitectureCPU::freeDevice( )
{
}

unsigned HardwareArchitectureCPU::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug) {

	if (lastSubstMatrix != subMat) {
		if (subMat == "BL50") {
			for (unsigned i=0; i<32; ++i) {
				for (unsigned j=0; j<32;++j) {
					substMatrix[i][j] = cpu_blosum50[i][j];
				}
			}
		} else if (subMat == "BL62") {
			for (unsigned i=0; i<32; ++i) {
				for (unsigned j=0; j<32;++j) {
					substMatrix[i][j] = cpu_blosum62[i][j];
				}
			}
		} else if (subMat == "BL90") {
			for (unsigned i=0; i<32; ++i) {
				for (unsigned j=0; j<32;++j) {
					substMatrix[i][j] = cpu_blosum90[i][j];
				}
			}
		} else {
			for (unsigned i=0; i<32; ++i) {
				for (unsigned j=0; j<32;++j) {
					substMatrix[i][j] = cpu_dna1[i][j];
				}
			}
		}
	}

	lastSubstMatrix = subMat;

	unsigned mc = 0;

	QTime tott; tott.start();
	
	if (calcEndpos) {
		for(unsigned j=startPos; j<=stopPos; ++j) {
			scores[j] = sw_single_with_endpos( seqlib + (offsets[j]), sizes[j], strToAlign, strlen(strToAlign), alpha, beta, end_positions[j]);
		}
	} else {
		for(unsigned j=startPos; j<=stopPos; ++j) {
			scores[j] = sw_single( seqlib + (offsets[j]), sizes[j], strToAlign, strlen(strToAlign), alpha, beta);
		}
	}

	int time = tott.elapsed();
	printf("CPU elapsed: %7.3f (s)\n", time/1000.0);

	//calcolo megacups
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	if (debug)
		printf("CPU amynoacids: %u\n", seqLibSize);

	time = (time > 1) ? time : 1;

	mc = static_cast<unsigned>((static_cast<double>(seqLibSize) / time) * (sizeNotPad/1048.576));

	return mc;
}

unsigned HardwareArchitectureCPU::getAvailableDevicesNumber()
{
	return 1;
}

HardwareArchitectureAPI *HardwareArchitectureCPU::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos)
{
// 	if ( i >= getAvailableDevicesNumber() ) 
// 		throw string("cannot allocate more CPU devices than available");

	return new HardwareArchitectureCPU(lib, tbu, num, ofs, sz, sc, endpos);
}

inline int HardwareArchitectureCPU::sbt(char A, char B)
{
		return substMatrix[A-60][B-60];
}

int HardwareArchitectureCPU::sw_single( const char* seqA, const unsigned sizeA, const char* seqB, const unsigned sizeB, int alpha, int beta )
{
	int score = 0;

	unsigned sizeMax = (sizeA > sizeB) ? sizeA : sizeB;

	memset(H, 0, (sizeMax+1)*sizeof(int));
	memset(F, 0, (sizeMax+1)*sizeof(int));

	for(unsigned i=1; i<sizeB; ++i) {
		register int h_jprev=0, e_jprev=0, f_jprev=0;
		unsigned j=1;
		for(; j<sizeA; ++j) {
			// calcolo di f
			int tmp1 = H[j] - alpha;
			int tmp2 = F[j] - beta;
			int f = maxx(tmp1, tmp2); //f = maxx( f, 0 );

			//calcolo di e
			tmp1 = h_jprev - alpha;
			tmp2 = e_jprev - beta;
			int e = maxx(tmp1, tmp2); //e = maxx( e, 0 );

			//calcolo di h
			int h = H[j-1];
			//char temp = seqA[j];
			//temp = seqB[i];
			h = h + sbt(seqA[j], seqB[i]);
			h = maxx(0, h);
			h = maxx(h, e);
			h = maxx(h, f);
			
			//printf ("%d\t", sbt(seqA[j], seqB[i]));
			//cout << h << '\t';
			score = maxx(score, h);

			H[j-1] = h_jprev;
			F[j-1] = f_jprev;
			e_jprev = e;
			h_jprev = h;
			f_jprev = f;

		}
		H[j-1] = h_jprev;
		F[j-1] = f_jprev;
		//cout << endl;
	}
	return score;
}


int HardwareArchitectureCPU::sw_single_with_endpos(const char * seqA, const unsigned sizeA, const char * seqB, const unsigned sizeB, int alpha, int beta, unsigned & endpos)
{
	int score = 0;
	unsigned short q_endpos = 0;
	unsigned short s_endpos = 0;
	endpos = 0;
	
	unsigned sizeMax = (sizeA > sizeB) ? sizeA : sizeB;

	memset(H, 0, (sizeMax+1)*sizeof(int));
	memset(F, 0, (sizeMax+1)*sizeof(int));

	for(unsigned i=1; i<sizeB; ++i) {
		register int h_jprev=0, e_jprev=0, f_jprev=0;
		unsigned j=1;
		for(; j<sizeA; ++j) {
			// calcolo di f
			int tmp1 = H[j]-alpha;
			int tmp2 = F[j];
			tmp2 = tmp2 - beta;
			int f = maxx(tmp1, tmp2); //f = maxx( f, 0 );

			//calcolo di e
			tmp1 = h_jprev - alpha;
			tmp2 = e_jprev - beta;
			int e = maxx(tmp1, tmp2); //e = maxx( e, 0 );

			//calcolo di h
			int h = H[j-1];
			//char temp = seqA[j];
			//temp = seqB[i];
			h = h + sbt(seqA[j], seqB[i]);
			h = maxx(0, h);
			h = maxx(h, e);
			h = maxx(h, f);
			
			//printf ("%d\t", sbt(seqA[j], seqB[i]));
			//cout << h << '\t';
			if (h>score){
				score = h;
				q_endpos = i;
				s_endpos = j;
			}

			H[j-1] = h_jprev;
			F[j-1] = f_jprev;
			e_jprev = e;
			h_jprev = h;
			f_jprev = f;

		}
		H[j-1] = h_jprev;
		F[j-1] = f_jprev;
		//cout << endl;
	}
	
	endpos = s_endpos;
	endpos = (endpos<<16) + q_endpos;

	return score;

}

//--end----HardwareArchitectureCPU-----------------------------------------------------------------------------------------------------------


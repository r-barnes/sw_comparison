
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


#include "hardwarearchitecturecuda.h"

#include <iostream>

#include <QtCore/QTime>

#include <cutil.h>
#include <cuda_runtime_api.h>


// l'handler di cui sotto è stato il primo sviluppato nel progetto. E' il famoso handler dei triangoli. Non è più usato ma rimane perchè potrebbe tornare utile negli sviluppi futuri.
extern "C" double smithWatermanCuda( const char* strToAlign, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, int* h_scores);

extern "C" double smithWatermanCuda2(const char* strToAlign, const unsigned sizeNotPad, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* h_scores, unsigned* h_endpos, char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int * &d_scores,  unsigned * &d_endpos, bool calc_endpos, unsigned debug);

extern "C" void swInitMem( const char *seqlib, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, char **d_seqlib, unsigned **d_offsets, unsigned **d_sizes, int **d_scores, unsigned **d_endpos);

extern "C" void swCleanMem( char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int *d_scores, unsigned *d_endpos);

 
//------HardwareArchitectureCUDA------------------------------------------------------------------------------------------------------------------
HardwareArchitectureCUDA::HardwareArchitectureCUDA(const unsigned deviceNum, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : HardwareArchitectureAPI(lib, tbu, num, ofs, sz, sc, endpos), dev_seqlib(NULL), dev_offsets(NULL), dev_sizes(NULL), dev_scores(NULL), lastSubstMatrix("")
{
	device = deviceNum;
}

HardwareArchitectureCUDA::~HardwareArchitectureCUDA()
{
}

void HardwareArchitectureCUDA::freeDevice( )
{
	swCleanMem( dev_seqlib, dev_offsets, dev_sizes, dev_scores, dev_endpos);
}

unsigned HardwareArchitectureCUDA::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug)
{
	unsigned mc = 0;


	///dovrebbe bastare questo per l'attivazione del device opportuno
	if (!dev_seqlib) {
		//cout << "GPU device " << device << ", initialization..." << endl;
		cudaSetDevice(device);
		cudaGetLastError();
		swInitMem( seqlib, numSeqs, totBytesUsed, offsets, sizes, &dev_seqlib, &dev_offsets, &dev_sizes, &dev_scores, &dev_endpos);
	}

	QTime tott; tott.start();

	if (debug)
		cout << "using kernel without profile...." << endl;

	smithWatermanCuda2( strToAlign, sizeNotPad, seqlib, startPos, stopPos, numSeqs, totBytesUsed, offsets, sizes, alpha, beta, subMat.c_str(), lastSubstMatrix.c_str(), scores, end_positions, dev_seqlib, dev_offsets, dev_sizes, dev_scores, dev_endpos, calcEndpos, debug);

	int time = tott.elapsed();
	
	lastSubstMatrix = subMat;

	printf("GPU device %d, elapsed: %7.3f (s)\n", device, time/1000.0);

	//calcolo megacups
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	time = (time > 1) ? time : 1;

	if (debug)
		cout << "GPU device " << device << ", amynoacids: " << seqLibSize << endl;

	mc = static_cast<unsigned>( (seqLibSize / time) * ((sizeNotPad-1)/1048.576) );


	return mc;
}

unsigned HardwareArchitectureCUDA::getAvailableDevicesNumber()
{
	int cnt = 0;

	cudaError_t err = cudaSetDevice ( 0 );

	struct cudaDeviceProp prop;
	err = cudaGetDeviceProperties ( &prop, 0 );
	string sterr = cudaGetErrorString(err);
	if (sterr != "no error") {
		throw sterr;
	}

	cudaGetDeviceCount(&cnt);

	return cnt;
}

HardwareArchitectureAPI *HardwareArchitectureCUDA::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos)
{
	if ( i >= getAvailableDevicesNumber() ) 
		throw string("cannot allocate more CUDA devices than available");

 	struct cudaDeviceProp prop;
 	cudaGetDeviceProperties(&prop, i);
 	cudaGetLastError();
 
 	cout << "Properties of CUDA device " << i << ":" << endl;
 	cout << "\t-" << prop.name << endl;
	//cout << "\t-" << prop->bytes << endl;
	//cout << "\t-" << prop->major << endl;
	//cout << "\t-" << prop->minor << endl;

	return new HardwareArchitectureCUDA(i, lib, tbu, num, ofs, sz, sc, endpos);
}

//-end-----HardwareArchitectureCUDA----------------------------------------------------------------------------------------------------------

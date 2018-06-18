
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


#include "hardwarearchitecturecudaprof.h"

#include <iostream>

#include <QtCore/QTime>

#include <cuda_runtime_api.h>

extern "C" double smithWatermanCudaProf( const char* strToAlign, const unsigned sizeNotPad, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* h_scores, char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int * &d_scores, unsigned debug);

extern "C" void swInitMemProf( const char *seqlib, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, int* h_scores, char **d_seqlib, unsigned **d_offsets, unsigned **d_sizes, int **d_scores);

extern "C" void swCleanMemProf( char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int* d_scores );

HardwareArchitectureCUDAProf::HardwareArchitectureCUDAProf(const unsigned deviceNum, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos) : HardwareArchitectureAPI(lib, tbu, num, ofs, sz, sc, endpos), dev_seqlib(NULL), dev_offsets(NULL), dev_sizes(NULL), dev_scores(NULL), lastSubstMatrix("")
{
	device = deviceNum;
}


HardwareArchitectureCUDAProf::~HardwareArchitectureCUDAProf()
{
}

void HardwareArchitectureCUDAProf::freeDevice( )
{
	swCleanMemProf( dev_seqlib, dev_offsets, dev_sizes, dev_scores);
}

unsigned HardwareArchitectureCUDAProf::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const bool calcEndpos, const unsigned debug)
{
	unsigned mc = 0;

	if (!dev_seqlib) {
		cudaSetDevice(device);
		cudaGetLastError();
		swInitMemProf( seqlib, numSeqs, totBytesUsed, offsets, sizes, scores, &dev_seqlib, &dev_offsets, &dev_sizes, &dev_scores);
	}

	QTime tott; tott.start();
	if (debug)
		cout << "using kernel with profile...." << endl;

	smithWatermanCudaProf( strToAlign, sizeNotPad, seqlib, startPos, stopPos, numSeqs, totBytesUsed, offsets, sizes, alpha, beta, subMat.c_str(), lastSubstMatrix.c_str(), scores, dev_seqlib, dev_offsets, dev_sizes, dev_scores, debug);

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

unsigned HardwareArchitectureCUDAProf::getAvailableDevicesNumber()
{
	int cnt = 0;

	cudaGetDeviceCount(&cnt);

	return cnt;
}

HardwareArchitectureAPI *HardwareArchitectureCUDAProf::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc, unsigned *endpos)
{
	if ( i >= getAvailableDevicesNumber() ) 
		throw string("cannot allocate more CUDA devices than available");

	struct cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, i);
	cudaGetLastError();

	cout << "Properties of CUDA device " << i << ":" << endl;
	cout << "\t-" << prop.name << endl;

	return new HardwareArchitectureCUDAProf(i, lib, tbu, num, ofs, sz, sc, endpos);
}

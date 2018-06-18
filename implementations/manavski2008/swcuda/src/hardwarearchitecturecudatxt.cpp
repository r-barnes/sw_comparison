
#include "hardwarearchitecturecudatxt.h"

#include <iostream>

#include <QtCore/QTime>

#include <cuda_runtime_api.h>

extern "C" double smithWatermanCudaTxt( const char* strToAlign, const unsigned sizeNotPad, seqBlock *seqlib, const unsigned linLibSize, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, int* h_scores);

//------HardwareArchitectureCUDA---------------------------------------------------------------------------------------------
HardwareArchitectureCUDATxt::HardwareArchitectureCUDATxt(const unsigned deviceNum, seqBlock *linLib, const unsigned linLibSize, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc) : HardwareArchitectureAPI(NULL, tbu, num, ofs, sz, sc)
{
	device = deviceNum;
	linSeqLibSize = linLibSize;
	linSeqLib = linLib;
}

void HardwareArchitectureCUDATxt::freeDevice( )
{
}

unsigned HardwareArchitectureCUDATxt::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const unsigned debug)
{
	unsigned mc = 0;

	QTime tott; tott.start();
	
	double timer = 0;//smithWatermanCudaTxt( strToAlign, sizeNotPad, linSeqLib, linSeqLibSize, startPos, stopPos, numSeqs, totBytesUsed, offsets, sizes, alpha, beta, scores);
		
	int tot_time = tott.elapsed();

	int time = static_cast<int>(timer);

	printf("GPU tot elapsed: %d (ms), kernel elapsed: %d (ms)\n", tot_time, time);

	//calcolo megacups
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	time = (time > 1) ? time : 1;
	tot_time = (tot_time > 1) ? tot_time : 1;

	mc = static_cast<unsigned>((static_cast<double>(seqLibSize) / tot_time) * (sizeNotPad/1048.576));

	return mc;
}


unsigned HardwareArchitectureCUDATxt::getAvailableDevicesNumber()
{
	int cnt = 0;

	cudaGetDeviceCount(&cnt);

	return cnt;
}

HardwareArchitectureAPI *HardwareArchitectureCUDATxt::getDevice(unsigned i, seqBlock *linLib, const unsigned linLibSize, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc) {

	if ( i >= getAvailableDevicesNumber() ) 
		throw string("cannot allocate more CUDA devices than available");

	return new HardwareArchitectureCUDATxt(i, linLib, linLibSize, tbu, num, ofs, sz, sc);
}





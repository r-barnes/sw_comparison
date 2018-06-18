#include "hardwarearchitecturecpuprof.h"

#include <QtCore/QTime>
#include "blosum.h"

#define maxx(a, b) ( (a) > (b) ) ? (a) : (b)
/*
const unsigned SEGNUM = 16;

void rMax(const SIMDRegister &a, const SIMDRegister &b, SIMDRegister &c) {
	if (a.reg.size()!= b.reg.size() || a.reg.size()!= c.reg.size())
		throw string("registers must have the same dimension");

	for (unsigned cnt=0; cnt < a.reg.size(); ++cnt)
		c.reg[cnt] = (a.reg[cnt] > b.reg[cnt]) ? a.reg[cnt] : b.reg[cnt];
}

void copySegNum(const SIMDRegister &a, const unsigned posa, SIMDRegister &b, const unsigned posb) {
	copy(a.reg.begin()+posa, a.reg.begin()+posa+SEGNUM, b.reg.begin()+posb);
}

void rSwap(SIMDRegister &a, SIMDRegister &b) {
	swap(a.reg, b.reg);
}

bool cmp(const SIMDRegister &a, const unsigned posa, const SIMDRegister &b, const unsigned posb) {
	return lexicographical_compare(a.reg.begin()+posa, a.reg.begin()+posa+SEGNUM, b.reg.begin()+posb, b.reg.begin()+posb+SEGNUM );

// 	bool res = TRUE;
// 	unsigned cnt=0;
// 
// 	while (res && cnt!=SEGNUM) {
// 		if (a.reg[posa+cnt] > b.reg[posb+cnt])
// 			res = FALSE;
// 		cnt++;
// 	}
// 
// 	return res;
}

//#######################################SIMD_Register#################################
SIMDRegister::SIMDRegister(const unsigned elem) {
	reg.resize(elem);

	for (unsigned cnt=0; cnt<reg.size(); ++cnt)
		reg[cnt] = 0;
}

void SIMDRegister::zeros() {
	for (unsigned cnt=0; cnt<reg.size(); ++cnt)
		reg[cnt] = 0;
}

void SIMDRegister::rightShift() {
	if (reg.size()<1)
		throw string("register must have at least one element");

	rotate(reg.begin(), reg.begin()+SEGNUM-1, reg.end());
	reg[0] = 0;
// 	rotate(reg.begin(), reg.begin()+1, reg.end());
// 	reg[SEGNUM-1] = 0;
}

SIMDRegister SIMDRegister::operator-(const unsigned op) {
	SIMDRegister temp(reg.size());

	for (unsigned cnt=0; cnt<reg.size(); ++cnt)
		temp.reg[cnt] = reg[cnt] - op;

	return temp;
}

void SIMDRegister::operator=(const SIMDRegister& b) {
	if (reg.size()!= b.reg.size())
		throw string("registers must have the same dimension");

	for (unsigned cnt=0; cnt<reg.size(); ++cnt)
		reg[cnt] = b.reg[cnt];
}

int SIMDRegister::maxElem() {
	vector<int>::const_iterator it = max_element(reg.begin(), reg.end());
	int score = *it;
	return score;
}

unsigned SIMDRegister::rSize() {
	return reg.size();
}

void SIMDRegister::addSegNum(const int* prof) {
	if (reg.size()!=SEGNUM)
		throw string("register must have a size equal to SEGNUM");

	for (unsigned cnt=0; cnt<SEGNUM; ++cnt)
		reg[cnt] = reg[cnt] + prof[cnt];
}


//#######################################HardwareArchitectureCPUProf#################################

HardwareArchitectureCPUProf::HardwareArchitectureCPUProf(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc) : HardwareArchitectureCPU(lib, tbu, num, ofs, sz, sc), queryProf(NULL)
{
}

HardwareArchitectureCPUProf::~HardwareArchitectureCPUProf() {
	if ( queryProf )
		delete [] queryProf;
}

HardwareArchitectureAPI *HardwareArchitectureCPUProf::getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc) {

	return new HardwareArchitectureCPUProf(lib, tbu, num, ofs, sz, sc);
}

unsigned HardwareArchitectureCPUProf::calcSmithWaterman(const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const unsigned debug) {
	
	
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
		} else {
			for (unsigned i=0; i<32; ++i) {
				for (unsigned j=0; j<32;++j) {
					substMatrix[i][j] = cpu_blosum90[i][j];
				}
			}
		}
	}

	lastSubstMatrix = subMat;

	unsigned mc = 0;

	QTime tott;
	int time=0;

	tott.start();

//####################################################################################
	//profiling
	profileCreator(strToAlign+1, sizeNotPad-1);

	//computation
	for(unsigned j=startPos; j<=stopPos; ++j) {
		scores[j] = 0;
		scores[j] = swSingle(sizeNotPad-1, seqlib+(offsets[j])+1, sizes[j]-1, alpha+beta, beta );
	}
//####################################################################################

	time = tott.elapsed();

	printf("CPU elapsed: %7.3f (s)\n", time/1000.0);

	//calcolo megacups
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	time = (time > 1) ? time : 1;

	mc = static_cast<unsigned>((static_cast<double>(seqLibSize) / time) * (sizeNotPad/1048.576));

	return mc;
}

void HardwareArchitectureCPUProf::profileCreator(const char *querySeq, unsigned queryLength) {

	unsigned segLen	= (queryLength + SEGNUM-1) / SEGNUM;
	unsigned row	= segLen * SEGNUM;

	queryProf = new int[32*segLen*SEGNUM];
	memset(queryProf, 0, 32*segLen*SEGNUM*sizeof(int));

	unsigned h, i, j, k;

	char a, b;
	unsigned pos;

	//profiling
	for (unsigned cnt=0; cnt<ALPHA_SIZE; ++cnt) {
		h = 0;

		a = AMINO_ACIDS[cnt];
		pos = (unsigned)a - 60;

		for (i=0; i<segLen; ++i) {
			j = i;

			for (k=0; k < SEGNUM; ++k) {
				if (j >= queryLength) {
					queryProf[pos*row + h] = 0;
				} else {
					b = querySeq[j];
					queryProf[pos*row + h] = substMatrix[a-60][b-60];
				}
			
				h++;
				j += segLen;
			}
		}
	}

// 	for (unsigned cnt=0; cnt< 32*segLen*SEGNUM; ++cnt) {
// 		if (cnt!=0 && (cnt%(segLen*SEGNUM))==0)
// 			printf("\n");
// 		printf("%d ", queryProf[cnt]);
// 	}
}


int HardwareArchitectureCPUProf::swSingle( const unsigned querySize, const char* seqDb, const unsigned dbSize, int alpha, int beta ) {

	unsigned segLen	= (querySize + (SEGNUM-1)) / SEGNUM;
	unsigned row	= segLen * SEGNUM;

	SIMDRegister vHStore(segLen*SEGNUM);
	SIMDRegister vHLoad (segLen*SEGNUM);
	SIMDRegister vEStore(segLen*SEGNUM);
	SIMDRegister buf(segLen*SEGNUM);

	SIMDRegister vH(SEGNUM);
	SIMDRegister vF(SEGNUM);
	SIMDRegister vE(SEGNUM);
	SIMDRegister vMax(SEGNUM);

	//outer loop to process the database sequence
	for (unsigned i = 0; i<dbSize; ++i) {

		//initialize F to 0
		vF.zeros();

		//adjust the last H value to be used in the next segment over
		copySegNum(vHStore, vHStore.rSize() - SEGNUM, vH, 0);
		vH.rightShift();

		//swap the two H buffers
		rSwap(vHStore, vHLoad);

		char a = seqDb[i];

		//inner loop to process query sequence
		unsigned offset;
		unsigned j=0;
		for ( ; j<segLen; ++j) {
			
			offset = j * SEGNUM;

			//add the scoring profile to vH
			int * ptr = queryProf + ( (a-60)*row + offset );
			//printf("%d %d\n", ptr[0], ptr[1]);
			vH.addSegNum(ptr);
			
			//save any vH values greater than the maz
			rMax(vMax, vH, vMax);
			
			//adjust vH with any greater vE or vF value
			copySegNum(vEStore, offset, vE, 0);
			rMax(vH, vE, vH);
			rMax(vH, vF, vH);

			//save the vH values off
			copySegNum(vH, 0, vHStore, offset);

			//calculate the new vE and vF based on the gap penalties for this search
			vH = vH - alpha;
			vE = vE - beta;
			rMax(vH, vE, vE);
			copySegNum(vE, 0, vEStore, offset);

			vF = vF - beta;
			rMax(vF, vH, vF);

			//load the next vH value to process
			copySegNum(vHLoad, offset, vH, 0);
		}

		//#########################################LAZY_F###########################
		//shift the vF left
		vF.rightShift();

		//correct the vH values until there are no elements in vF that could influence the vH values
		j = 0;
		buf = vHStore - alpha;

		//offset = j * SEGNUM se j=0 offset=0;
		offset = 0;

		while( cmp(buf, offset, vF, 0) ) {

			copySegNum(vEStore, offset, vE, 0);		///

			copySegNum(vHStore, offset, vH, 0);
			rMax(vH, vF, vH);
			copySegNum(vH, 0, vHStore, offset);

			vH = vH - alpha;
			rMax(vE, vH, vE);						///
			copySegNum(vE, 0, vEStore, offset);		///
			
			vF = vF - beta;

			if (++j > segLen) {
				vF.rightShift();
				j = 0;
			}
			offset = j * SEGNUM;
		}
	}

	//calculate the max
	int score = vMax.maxElem();

	return score;
}
*/

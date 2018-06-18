#ifndef HARDWAREARCHITECTURECPUPROF_H
#define HARDWAREARCHITECTURECPUPROF_H

#include <hardwarearchitecturecpu.h>
#include <vector>

/**
	@author Svetlin Manavski <svetlin@manavski.com>
 */

///perchè questo commento
///questa classe si basava sulla vecchia versione non più esistente di hardwarearchitecturecpu.h. Quella in cui era contenuto anche farrar.
///ora è diverso ma visto che questa classe non funziona e non viene usata non ho perso tempo a sistemarla

/*
class SIMDRegister {
public:
	SIMDRegister(const unsigned elem);

	void zeros();
	void rightShift();

	SIMDRegister operator-(const unsigned op);
	void operator=(const SIMDRegister& b);
	int maxElem();
	unsigned rSize();
	void addSegNum(const int* prof);

	friend void rMax(const SIMDRegister &a, const SIMDRegister &b, SIMDRegister &c);
	friend void copySegNum(const SIMDRegister &a, const unsigned posa, SIMDRegister &b, const unsigned posb);
	friend void rSwap(SIMDRegister &a, SIMDRegister &b);
	friend bool cmp(const SIMDRegister &a, const unsigned posa, const SIMDRegister &b, const unsigned posb);

private:
	vector<int> reg;
};

class HardwareArchitectureCPUProf : public HardwareArchitectureCPU
{
public:
	virtual unsigned calcSmithWaterman ( const char *strToAlign, const unsigned sizeNotPad, const int alpha, const int beta, const std::string subMat, const unsigned startPos, const unsigned stopPos, const unsigned debug );

	virtual ~HardwareArchitectureCPUProf();

	static HardwareArchitectureAPI *getDevice(unsigned i, const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc);

protected:
	HardwareArchitectureCPUProf(const char *lib, const unsigned tbu, const unsigned num, unsigned *ofs, unsigned *sz, int *sc);

	void profileCreator(const char *querySeq, unsigned queryLength);

	int swSingle(const unsigned querySize, const char* seqDb, const unsigned dbSize, int alpha, int beta );

private:

	int *queryProf;
};
*/
#endif

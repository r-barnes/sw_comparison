
#ifndef _TEMPLATE_HANDLERSOLEXA_H_
#define _TEMPLATE_HANDLERSOLEXA_H_

#include "e2gengine.h"

class GPUHandlerSolexa : public E2GEngine {

public:
	GPUHandlerSolexa(unsigned device, const unsigned maxSeqs, const unsigned dbbytes);
	~GPUHandlerSolexa();
	
	void est2genome(vector<EstAlignPair *> &input_pairs, const short unsigned first_gap_penalty, const short unsigned next_gap_penalty, const short unsigned splice_penalty, const short unsigned intron_penalty, bool debug);

private:
	unsigned myDevice;
	unsigned maxSeqsNumber;
	unsigned maxDBBytes;

	char *d_queries, *d_seqlib, *d_splice_sites;
	unsigned *d_offsets;
	unsigned *d_sizes;
	int *d_scores;

	char *h_queries, *h_subjects, *h_splice_sites;
	unsigned *h_offsets, *h_sizes;
	
	void allocMem();
	
	void setMem( const unsigned numPairs, const unsigned totBytesUsed );

	void solexaCleanMem();
	
	void convertData(vector<EstAlignPair *> &input_pairs, unsigned &totBytesUsed);
	
};




#endif


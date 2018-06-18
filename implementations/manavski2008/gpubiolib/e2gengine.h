#ifndef E2GENGINE_H
#define E2GENGINE_H

/**
	@author Svetlin Manavski <svetlin.a@manavski.com>
*/

#include "alignmentpair.h"

#include <QMutex>

struct EstSSavePair {
	EstSSavePair() : col(0), row(0) {
	}
	
	int col;
	int row;
};
#define EstPSavePair EstSSavePair*


/*****************************************************************************
** Coordinates data structure
**
** @attr left: left end
** @attr right: right end 
******************************************************************************/
typedef struct EstSCoord
{
	int left;
	int right;
} EstOCoord;
#define EstPCoord EstOCoord*


struct E2GBacktrackBuffer {
	
	E2GBacktrackBuffer();
	~E2GBacktrackBuffer();
	
	void pairFree();
	void memReset();
	void pairInit(unsigned int max_bytes);
	void doNotForget( int col, int row );
	int pairRemember( int col, int row );


	float megabytes;
	
	// memory allocated one time only
	EstPSavePair static_rpair;
	int static_rpairs;
	int static_rpair_size;
	bool static_rpairs_sorted;
};


/// basic Est2Genome engine prototipe
class E2GEngine{
public:
	E2GEngine();
	E2GEngine(int match, int mismatch, int gap, int neutral, char pad_char);

    ~E2GEngine();

	void alignNonRecursiveBackTrack(EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, E2GBacktrackBuffer *);
	
	void constructAlignmentResult(EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, E2GBacktrackBuffer *);
	
	int getMatrixValue(unsigned char a, unsigned char b) const;
protected:
	int ali_lsimmat[256][256];

		/* find the GT/AG splice sites */
	void estFindSpliceSites(EstAlignPair &pair);
	
	void matInit(int match, int mismatch, int gap, int neutral, char pad_char);
};


class Est2GenomePrinter {
	public:
		enum TITLE {FEFGRG, FEFGFG, REFGRG, REFGFG, FEFG };
		
		Est2GenomePrinter(const string &stroutfile, bool gffstyle);
		~Est2GenomePrinter();
		
		bool est2genomeOutputGFF(const AlignPair &, const EstResultSummary &, unsigned genStartPos, unsigned genStopPos);

		bool est2genomeMakeOutput(Est2GenomePrinter::TITLE title, EstAlignPair *pair, unsigned genStartPos, unsigned genStopPos, E2GEngine *e2gEngine, int gap_penalty, int intron_penalty, int splice_penalty, int minscore, unsigned minscore_nointron, bool align, int width);

	private:
		FILE* outfile;
		bool isOpen;
		QMutex m_makeOutput;
		bool GFFOutputStyle;
		unsigned counter;
		
		bool isThereIntron(const EstResultDetailed *ge);
		void printAlign(E2GEngine *e2gEngine,  EstAlignPair *pair, int width );
	
		void myestWriteMsp(int *matches, int *len, int *tsub, EstAlignPair *pair, int gsub, int gpos, int esub, int epos, int reverse, int gapped);
		void outBlastStyle(E2GEngine *e2gEngine,  EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, int gapped, int reverse);
		bool outGFFStyle(E2GEngine *e2gEngine,  EstAlignPair *pair, unsigned genStartPos, unsigned genStopPos, int gap_penalty, int intron_penalty, int splice_penalty);


};



#endif

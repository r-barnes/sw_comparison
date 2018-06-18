#ifndef ALIGNMENTPAIR_H
#define ALIGNMENTPAIR_H

/**
	@author Svetlin Manavski <svetlin.a@manavski.com>
*/

#include "biosequence.h"

#include <vector>
using namespace std;

typedef enum { INTRON=0, DIAGONAL=1, DELETE_EST=2, DELETE_GENOME=3, FORWARD_SPLICED_INTRON=-1, REVERSE_SPLICED_INTRON=-2 } directions;


struct PairResult {
	PairResult(bool fw) : subjectStart(0), queryStart(0), subjectStop(0), queryStop(0), score(0), len(0), forward(fw) {};
			
	int subjectStart;
	int queryStart;
	int subjectStop;
	int queryStop;
	int score;
	int len;

	bool forward;
};

struct EstResultDetailed : public PairResult {
	EstResultDetailed(bool fw) : PairResult(fw) {};

	vector<int> align_path;
};

struct EstResultSummary : public PairResult {
	EstResultSummary() : PairResult(true), numGaps(0), numMismatches(0) {};

	unsigned numGaps;
	unsigned numMismatches;
	vector< pair<unsigned, unsigned> > exons;
};


class AlignPair {
public:
	AlignPair(BioSequence *q, const BioSequence *s, bool fw);
	~AlignPair();

	BioSequence *getQuery();
	const BioSequence *getQuery() const;
	const BioSequence *getSubject() const;
	bool isForward() const;
protected:
	// we are not owners of these pointers
	BioSequence *query;
	const BioSequence *subject;
	bool forward;
};


struct EstAlignPair : AlignPair {
	EstAlignPair(BioSequence *q, const BioSequence *s, bool fw);
	~EstAlignPair();
	
	void freePaths();
	void allocPaths();
	EstResultSummary makeSummary() const;

	void operator=(const EstAlignPair &v) {
		query = v.query;
		subject = v.subject;
		forward = v.forward;
		
		splice_sites = v.splice_sites;
		freePaths();
		ppath = NULL;
		best_intron_coord = NULL;
	
		gmax = v.gmax;
		emax = v.emax;
		max_score = v.max_score;
		alignmentResult = v.alignmentResult;
	};
	
	string splice_sites;
	
	// memory allocated before every alignment (depends upon the current genome and the est)
	unsigned char **ppath;
	int *best_intron_coord;
	int gmax;
	int emax;
	int max_score;
	
	EstResultDetailed alignmentResult;

};


#endif

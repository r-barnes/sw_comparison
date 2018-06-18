
#include "alignmentpair.h"

#include <iostream>
using namespace std;

AlignPair::AlignPair(BioSequence *q, const BioSequence *s, bool fw) : query(q), subject(s), forward(fw)
{
}

AlignPair::~AlignPair()
{
}

BioSequence * AlignPair::getQuery()
{
	return query;
}

const BioSequence * AlignPair::getQuery() const
{
	return query;
}

const BioSequence * AlignPair::getSubject() const 
{
	return subject;
}

bool AlignPair::isForward() const
{
	return forward;
}

///------------EstAlignPair--------------------------------------------------------------------------------------------
EstAlignPair::EstAlignPair(BioSequence *q, const BioSequence *s, bool fw) : AlignPair(q, s, fw), splice_sites(""), ppath(NULL), best_intron_coord(NULL), gmax(-1), emax(-1), max_score(0), alignmentResult(fw) {};

EstAlignPair::~EstAlignPair() {
}

void EstAlignPair::freePaths()
{
	unsigned gpos = 0;
	if (best_intron_coord) {
		delete []best_intron_coord;	
		best_intron_coord = NULL;
	}
	if (ppath) {
		unsigned glen = subject->getSize();
		for(gpos=0;gpos<glen;gpos++)
			delete [] (ppath[gpos]);
		delete []ppath;
		ppath = NULL; 
	}
}

void EstAlignPair::allocPaths() {
	unsigned glen = subject->getSize();
	unsigned elen = query->getSize();
	
	int e_len_pack;
	e_len_pack = elen/4+1;
	ppath = new unsigned char* [glen];
	for (unsigned gpos=0; gpos<glen; gpos++) {
		ppath[gpos] = new unsigned char [e_len_pack];
		fill(ppath[gpos], ppath[gpos]+e_len_pack, 0);
	}
	best_intron_coord = new int[elen+1];
	fill(best_intron_coord, best_intron_coord+elen+1, 0);
	emax = -1;
	gmax = -1;
	max_score = 0;
}

EstResultSummary EstAlignPair::makeSummary() const
{
	EstResultSummary newResult;

	PairResult *p_newResult = &newResult;

	*p_newResult = alignmentResult;

	int gsub;
	int gpos;
	int esub;
	int epos;
	int p;
	int total_len     = 0;
	int goff = 0; //genome->getOffset();
	
	const EstResultDetailed *ge = &alignmentResult;

	gsub = gpos = ge->subjectStart;
	esub = epos = ge->queryStart;

	string qstr = query->getSequence();
	string sstr = subject->getSequence();

	newResult.numMismatches = 0;
	newResult.numMismatches += newResult.queryStart;
	newResult.numMismatches += (qstr.size() - newResult.queryStop  - 1);

	newResult.numGaps = 0;

	for(p=0;p<ge->len;p++) {
			
		if (ge->align_path[p] <= INTRON) {
				
			pair<unsigned, unsigned> np(goff+gsub+1, goff+gpos);
			newResult.exons.push_back(np);
				
			gpos += ge->align_path[++p];
			esub = epos;
			gsub = gpos;
			
		} else if(ge->align_path[p] == DIAGONAL) {
			total_len++;
			if (qstr[epos] != sstr[gpos]) ++newResult.numMismatches;
			gpos++;
			epos++;
			
		} else if(ge->align_path[p] == DELETE_EST) {
			epos++;
			total_len++;
			++newResult.numGaps;
				
		} else if (ge->align_path[(int)p] == DELETE_GENOME) {
			gpos++;
			total_len++;
			++newResult.numGaps;
		}
	}
	pair<unsigned, unsigned> np(goff+gsub+1, goff+gpos);
	newResult.exons.push_back(np);


	return newResult;
}
///------end------EstAlignPair--------------------------------------------------------------------------------------------








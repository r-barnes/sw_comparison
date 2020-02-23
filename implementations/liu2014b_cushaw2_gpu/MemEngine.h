/*
 * MemEngine.h
 *
 *  Created on: Dec 24, 2011
 *      Author: yongchao
 */

#ifndef MEMENGINE_H_
#define MEMENGINE_H_
#include "Macros.h"
#include "Utils.h"
#include "Genome.h"
#include "Sequence.h"
#include "Aligner.h"
#include "SAM.h"
#include "Options.h"
#include "Mapping.h"
#include "Structs.h"

/*engine for maximal exact match (MEM) seed generation*/
class MemEngine
{
public:
	MemEngine(Options* options, Genome* rgenome, SAM* sam);
	virtual ~MemEngine();

	/*for single-end alignment*/
	bool align(Sequence& seq, Mapping*& bestMapping);

	/*for multiple single-end alignment*/
	bool align(Sequence& seq, vector<Mapping*>& mappings);

	/*for paired-end alignment*/
	bool align(Sequence& seq1, Sequence& seq2, Mapping*& bestMapping1,
			Mapping*& bestMapping2);

	/*multiple paired-end alignment */
	bool align(Sequence& seq1, Sequence& seq2, vector<Mapping*>& mappings1,
			vector<Mapping*>& mappings2);

	/*update distance information*/
	inline void updateDistance() {
		_maxDistance = _options->getInsertSize()
				+ 4 * _options->getStdInsertSize();
	}

protected:
	/*protected member variables*/
	Options* _options;
	Genome* _rgenome;
	BWT* _rbwt; /*pointer to the reverse BWT data of the target sequence*/
	SuffixArray* _rsa; /*pointer to the reverse Suffix Array data of the target sequence*/
	int64_t _bwtSeqLength;
	uint8_t* _pacGenome;
	SAM* _sam;
	Aligner* _aligner; /*aligner with the user-specified scoring scheme*/
#ifdef HAVE_TWICE_RESCUE
	Aligner* _swaligner; /*aligner for the second-time rescuing using SW*/
#endif

	/*parameters*/
	float _minRatio; /*the minimal portion of the query in the optimal local alignment*/
	float _minIdentity; /*the minimal identity in the optimal local alignment*/
	uint32_t _maxSeedOcc;
	uint32_t _mapRegionSizeFactor; /*the factor for maximal mapping region*/
	int _minAlignScore;
	int _mapQualReliable;
	int _maxGapSize;
	int64_t _maxDistance;
	int _maxMultiAligns; /*maximal number of best alignments*/

	size_t _targetSize;
	uint8_t* _target;

	/*processing*/
	vector<vector<Seed*> > _seedPairs;
	vector<int> _genomeIndices;
	vector<Seed> _seeds, _seeds1, _seeds2;
	vector<int32_t> _bestHits1, _bestHits2;
	set<int64_t> _mapPositions;

	/*for SSE2*/
	vector<uint8_t> _sequences;
	vector<int32_t> _seqOffsets;
	vector<AlignScore> _alignScores;

	/*compute the alignment for a sequence*/
	inline Mapping* _getAlignment(Aligner* aligner, Sequence& seq,
			size_t numSeeds, Seed* seeds, float minBasePortion) {
		int32_t bestHit;
		int mapQual;

		/*get the single-end alignment*/
		if (numSeeds > 0) {
			bestHit = _getBestHit(_aligner, seq, numSeeds, seeds, _minRatio,
					mapQual);

			/*get the alignment*/
			if (bestHit >= 0) {
				return _getAlignment(_aligner, seq, seeds[bestHit], _minRatio,
						mapQual);
			}
		}
		return NULL;
	}
	/*calculate the mate read mapping region*/
	inline void _getMateRegion(int64_t& genomeStart, int64_t& genomeEnd,
			int64_t selfMapPosition, int selfStrand, int selfLength,
			int mateLength) {
		if (selfStrand == 0) {
			genomeStart = selfMapPosition + 1;
			genomeEnd = selfMapPosition + _maxDistance + mateLength;
		} else {
			genomeStart = selfMapPosition + selfLength - _maxDistance;
			genomeEnd = selfMapPosition + selfLength - 1;
		}
	}

	/*normal single-end alignment*/
	Mapping* _getAlignment(Aligner* aligner, Sequence& seq, Seed& seeds,
			float minBasePortion, int mapQual);

	Mapping* _getAlignment(Aligner* aligner, Sequence& seq, int strand,
			int window, int genomeIndex, uint32_t genomeStart,
			size_t genomeLength, float minBasePortion, float minIdentity,
			int mapQual);

	/*rescue reads using paired-end information*/
	Mapping* _getAlignment(Aligner* aligner, Sequence& seq, int strand,
			int window, int genomeIndex, uint32_t genomeStart,
			size_t genomeLength, float minMatchLengthRatio, int mapQual);

private:
	/*get the best seed hits*/
	int32_t _getBestHits(Aligner* aligner, Sequence& seq, size_t numSeeds,
			Seed* seeds, float minBasePortion, vector<int32_t>& bestHits,
			int& mapQual);

	/*get only one best hit*/
	int32_t _getBestHit(Aligner* aligner, Sequence& seq, size_t numSeeds,
			Seed* seeds, float minBasePortion, int& mapQual);

	/*compute the alignment scores*/
	size_t _getAlignmentScores(Aligner* aligner, Sequence& seq, size_t numSeeds,
			Seed* seeds, float minBasePortion);

	/*generate seeds for a single strand. Returns true if it has an exact match in the full length, and
	 * false otherwise*/
	void _genMEMSeeds(Sequence& seq, int strand, uint4* ranges,
			size_t& numRanges, uint32_t minSeedSize);

	/*locate the mapping positions of each seeds and perform voting on conditions*/
	void _locateSeeds(Sequence& seq, vector<Seed>& seeds);

	static int cmpAlignScores(const void *arg1, const void* arg2) {
		const AlignScore* p = static_cast<const AlignScore*>(arg1);
		const AlignScore* q = static_cast<const AlignScore*>(arg2);

		if (p->_score > q->_score) {
			return -1;
		} else if (p->_score < q->_score) {
			return 1;
		}
		return 0;
	}
};

#endif /* MEMENGINE_H_ */

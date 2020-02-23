/*
 * MemEngine.cpp
 *
 *  Created on: Dec 24, 2011
 *      Author: yongchao
 */

#include "MemEngine.h"

MemEngine::MemEngine(Options* options, Genome* rgenome, SAM* sam) {
	_options = options;
	_rgenome = rgenome;
	_sam = sam;
	_rbwt = _rgenome->getBWT();
	_rsa = _rgenome->getSuffixArray();

	/*get the reference sequence length*/
	_bwtSeqLength = _rbwt->getBwtSeqLength();

	/*get the packed genome*/
	_pacGenome = _rgenome->getPacGenome();

	/*get the minimal seed size*/
	_minIdentity = _options->getMinIdentity() * 100;
	_minRatio = _options->getMinRatio(); /*the minimal portion of the query in the optimal local alignment*/
	_maxSeedOcc = _options->getMaxSeedOcc();
	_mapRegionSizeFactor = 2;
	_minAlignScore = _options->getMinAlignScore();
	_mapQualReliable = 20;
	_maxGapSize = 10;
	_maxMultiAligns = _options->getMaxMultiAligns();

	/*for processing*/
	_maxDistance = _options->getInsertSize() + 4 * _options->getStdInsertSize();

	_seedPairs.resize(1024);
	for (size_t i = 0; i < _seedPairs.size(); ++i) {
		_seedPairs[i].reserve(256);
	}
	_genomeIndices.resize(1024);

	/*create default local aligner*/
	_aligner = new Aligner(options);

#ifdef HAVE_TWICE_RESCUE
	/*create twice-rescuing local aligner*/
	_swaligner = new Aligner(options, 11, 19, 26, 9);
	//_swaligner = new Aligner(options, 2, 6, 5, 3);
#endif

	_targetSize = 1024;
	_target = new uint8_t[_targetSize];
	if (_target == NULL) {
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	/*reserve space*/
	_bestHits1.reserve(128);
	_bestHits2.reserve(128);
}
MemEngine::~MemEngine() {
	for (size_t i = 0; i < _seedPairs.size(); ++i) {
		_seedPairs[i].clear();
	}
	_seedPairs.clear();
	_genomeIndices.clear();
	_seeds.clear();
	_seeds1.clear();
	_seeds2.clear();
	_bestHits1.clear();
	_bestHits2.clear();
	_sequences.clear();
	_seqOffsets.clear();
	_alignScores.clear();
	_mapPositions.clear();

	/*release the aligner*/
	delete _aligner;
#ifdef HAVE_TWICE_RESCUE
	delete _swaligner;
#endif

	if (_target) {
		delete[] _target;
	}
}
/*single-end alignment*/
bool MemEngine::align(Sequence& seq, Mapping*& bestMapping) {
	/*initialize the mapping*/
	bestMapping = NULL;

	/*get all seeds*/
	_locateSeeds(seq, _seeds);

	/*get the single-end alignment*/
	if (_seeds.size() > 0) {
		bestMapping = _getAlignment(_aligner, seq, _seeds.size(), &_seeds[0],
				_minRatio);
	}

	return bestMapping != NULL;
}

/*for multiple single-end alignment*/
bool MemEngine::align(Sequence& seq, vector<Mapping*>& mappings) {
	Mapping* bestMapping;
	Seed* seeds;
	int mapQual;

	/*clear the set*/
	_mapPositions.clear();

	/*get all seeds*/
	_locateSeeds(seq, _seeds1);
	seeds = &_seeds1[0];

	/*get the best hits for the sequence*/
	_getBestHits(_aligner, seq, _seeds1.size(), seeds, _minRatio, _bestHits1,
			mapQual);

	/*get the alignment for the sequence*/
	for (size_t i = 0; i < _bestHits1.size(); ++i) {

		/*get the alignment*/
		bestMapping = _getAlignment(_aligner, seq, seeds[_bestHits1[i]],
				_minRatio, mapQual);

		if (!bestMapping) {
			break;
		}
		/*insert the mapping position to the set to check the uniquness*/
		pair<set<int64_t>::iterator, bool> ret = _mapPositions.insert(
				bestMapping->_gposition);
		if (ret.second == true) { /*a new position*/
			mappings.push_back(bestMapping);
		}

		if (mappings.size() >= _maxMultiAligns) {
			break;
		}
	}
	return mappings.size() > 0;
}
/* paired-end alignment, return only the "best" alignment*/
bool MemEngine::align(Sequence& seq1, Sequence& seq2, Mapping*& bestMapping1,
		Mapping*& bestMapping2) {
	int mapQual1, mapQual2;
	Seed* seeds1, *seeds2;
	int genomeIndex;
	int window, numErrors;
	int64_t genomeStart1 = 0, genomeEnd1 = 0, genomeStart2 = 0, genomeEnd2 = 0;
	int64_t qleft, qright, insertSize;
	bool good1, good2;

	/*initialize the mapping*/
	bestMapping1 = bestMapping2 = NULL;

	/*vote the seeds based on neighboring relationships*/
	_locateSeeds(seq1, _seeds1);
	seeds1 = &_seeds1[0];

	_locateSeeds(seq2, _seeds2);
	seeds2 = &_seeds2[0];

	/*get the best hits for the left sequence*/
	_getBestHits(_aligner, seq1, _seeds1.size(), seeds1, _minRatio, _bestHits1,
			mapQual1);

	/*get the best hits for the second sequence*/
	_getBestHits(_aligner, seq2, _seeds2.size(), seeds2, _minRatio, _bestHits2,
			mapQual2);

	/*check the distance between top seed pairs*/
	/*****************************************************************
	 * STAGE 1
	 *****************************************************************/
	/*resize the buffer and clear each entry*/
	if (_seedPairs.size() < _bestHits1.size()) {
		_seedPairs.resize(_bestHits1.size() << 1);
	}
	for (size_t i = 0; i < _seedPairs.size(); ++i) {
		_seedPairs[i].clear();
	}

	/*calculate the genome indices*/
	if (_bestHits2.size() > _genomeIndices.size()) {
		_genomeIndices.resize(_bestHits2.size() << 1);
	}
	for (size_t i = 0; i < _bestHits2.size(); ++i) {
		_rgenome->getGenomeIndex(seeds2[_bestHits2[i]]._targetPosition,
				_genomeIndices[i]);
	}

	/*evaluate all top seeds within the allowable insert size*/
	numErrors = _options->getNumErrors(seq1._length);
	for (size_t i = 0; i < _bestHits1.size(); ++i) {
		Seed& left = seeds1[_bestHits1[i]];

		/*get the genome index*/
		_rgenome->getGenomeIndex(left._targetPosition, genomeIndex);

		/*estimate the position of the 5' end*/
		qleft = (left._strand == 0) ?
				left._queryPosition : seq1._length - left._queryPosition - 1;
		for (size_t j = 0; j < _bestHits2.size(); ++j) {
			int32_t seedIndex2 = _bestHits2[j];
			Seed& right = seeds2[seedIndex2];

			/*check the genome index and the sequence strand*/
			if (genomeIndex != _genomeIndices[j]
					|| left._strand == right._strand) {
				continue;
			}
			/*estimate the position of the 5'end*/
			qright =
					(right._strand == 0) ?
							right._queryPosition :
							seq2._length - right._queryPosition - 1;

			/*estimate the distance between the two reads*/
			insertSize = labs(
					(int64_t) left._targetPosition - qleft
							- right._targetPosition + qright);
			if (insertSize <= _maxDistance + 2 * numErrors) {
				/*store the seed pair*/
				_seedPairs[i].push_back(&seeds2[seedIndex2]);
			}
		}
	}
	/*try to get short reads paired through paired seeds*/
	for (size_t i = 0; i < _bestHits1.size(); ++i) {
		vector<Seed*>& seeds = _seedPairs[i];
		/*skip the seed*/
		if (seeds.size() == 0) {
			continue;
		}
		/*get the alignment for the current seed*/
		Seed& left = seeds1[_bestHits1[i]];
		if ((bestMapping1 = _getAlignment(_aligner, seq1, left, _minRatio,
				mapQual1)) == NULL) {
			continue;
		}
		for (size_t j = 0; j < seeds.size(); ++j) {
			Seed& right = *(seeds[j]);
			/*perform alignment for the second sequence*/
			if ((bestMapping2 = _getAlignment(_aligner, seq2, right, _minRatio,
					mapQual2)) == NULL) {
				continue;
			}

			//check the final mapping positions of the alignment*/
			insertSize = labs(
					bestMapping1->_position - bestMapping2->_position);
			if (insertSize <= _maxDistance) {
				/*the two reads are paired and return*/
				return true;
			}
			delete bestMapping2;
		}
		delete bestMapping1;
	}

	/*if not paired*/
	bestMapping1 = bestMapping2 = NULL;
	/*get the alignment for the left sequence*/
	if (_bestHits1.size() > 0) {
		bestMapping1 = _getAlignment(_aligner, seq1, seeds1[_bestHits1[0]],
				_minRatio, mapQual1);
	}

	/*get the alignment for the right sequence*/
	if (_bestHits2.size() > 0) {
		bestMapping2 = _getAlignment(_aligner, seq2, seeds2[_bestHits2[0]],
				_minRatio, mapQual2);
	}

	/*check the distance between the sequences*/
	good1 = bestMapping1 && bestMapping1->_mapQual >= _mapQualReliable;
	good2 = bestMapping2 && bestMapping2->_mapQual >= _mapQualReliable;
	if (good1 && good2) {
		insertSize = labs(bestMapping1->_position - bestMapping2->_position);
		if (insertSize <= _maxDistance) {
			/*the two reads are paired and return*/
			return true;
		}
	}

	/*try to find the best alignment for the right sequence*/
	float swMinRatio = _minRatio;
	Mapping *mateMapping1 = NULL, *mateMapping2 = NULL;
	if (_options->rescueMate()) {
		if (good1) { /*have a reliable alignment*/
			//Utils::log("map1: %u %u\n", bestMapping1->_position, bestMapping1->_strand);
			window = _options->getNumErrors(seq2._length) * 2;

			/*calcualte the mapping region of the mate read*/
			_getMateRegion(genomeStart1, genomeEnd1, bestMapping1->_gposition,
					bestMapping1->_strand, seq1._length, seq2._length);

			/*refine the region*/
			_rgenome->refineRegionRange(bestMapping1->_genomeIndex,
					genomeStart1, genomeEnd1);

			/*perform the alignment*/
			mateMapping1 = _getAlignment(_aligner, seq2,
					1 - bestMapping1->_strand, window,
					bestMapping1->_genomeIndex, genomeStart1,
					genomeEnd1 - genomeStart1 + 1, swMinRatio, _minIdentity,
					SW_MAP_QUALITY_SCORE * mapQual1);

			/*succeeded in finding an alignment*/
			if (mateMapping1) {
				/*output the paired-end alignments*/
				if (bestMapping2 != NULL) {
					delete bestMapping2;
				}
				bestMapping2 = mateMapping1;

				return true;
			}
		}

		/*rescue the alignment from the right sequence*/
		if (good2) { /*have a reliable alignment*/
			window = _options->getNumErrors(seq1._length) * 2;
			//Utils::log("map2: %u %u\n", bestMapping2->_position, bestMapping2->_strand);

			/*calcualte the mapping region of the mate read*/
			_getMateRegion(genomeStart2, genomeEnd2, bestMapping2->_gposition,
					bestMapping2->_strand, seq2._length, seq1._length);

			/*refine the region*/
			_rgenome->refineRegionRange(bestMapping2->_genomeIndex,
					genomeStart2, genomeEnd2);

			/*perform the alignment*/
			mateMapping2 = _getAlignment(_aligner, seq1,
					1 - bestMapping2->_strand, window,
					bestMapping2->_genomeIndex, genomeStart2,
					genomeEnd2 - genomeStart2 + 1, swMinRatio, _minIdentity,
					SW_MAP_QUALITY_SCORE * mapQual2);

			/*failed to find an alignment*/
			if (mateMapping2) {
				/*output the paired-end alignments*/
				if (bestMapping1 != NULL) {
					delete bestMapping1;
				}
				bestMapping1 = mateMapping2;

				return true;
			}
		}
#ifdef HAVE_TWICE_RESCUE
		/*twice rescuing*/
		if (_options->rescueTwice()) {
			if (good1) {
				window = _options->getNumErrors(seq2._length) * 2;
				/*perform the alignment*/
				mateMapping1 = _getAlignment(_swaligner, seq2,
						1 - bestMapping1->_strand, window,
						bestMapping1->_genomeIndex, genomeStart1,
						genomeEnd1 - genomeStart1 + 1, swMinRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * mapQual1);

				/*succeeded in finding an alignment*/
				if (mateMapping1) {
					/*output the paired-end alignments*/
					if(bestMapping2 != NULL) {
						delete bestMapping2;
					}
					bestMapping2 = mateMapping1;

					return true;
				}
			}
			if (good2) {
				window = _options->getNumErrors(seq1._length) * 2;
				/*perform the alignment*/
				mateMapping2 = _getAlignment(_swaligner, seq1,
						1 - bestMapping2->_strand, window,
						bestMapping2->_genomeIndex, genomeStart2,
						genomeEnd2 - genomeStart2 + 1, swMinRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * mapQual2);

				/*failed to find an alignment*/
				if (mateMapping2) {
					/*output the paired-end alignments*/
					if(bestMapping1 != NULL) {
						delete bestMapping1;
					}
					bestMapping1 = mateMapping2;

					return true;
				}
			}
		}
#endif
	}

	return false;
}

/*paired-end alignment, returning multiple equalivalent "best" alignments if applicable*/
bool MemEngine::align(Sequence& seq1, Sequence& seq2,
		vector<Mapping *>& mappings1, vector<Mapping *>& mappings2) {
	int mapQual1, mapQual2;
	Seed* seeds1, *seeds2;
	int genomeIndex;
	int window, numErrors;
	int64_t genomeStart1 = 0, genomeEnd1 = 0, genomeStart2 = 0, genomeEnd2 = 0;
	int64_t qleft, qright;
	Mapping *bestMapping1, *bestMapping2;
	bool good1, good2;

	/*clear the set*/
	_mapPositions.clear();

	/*initialize the mapping*/
	bestMapping1 = bestMapping2 = NULL;

	/*vote the seeds based on neighboring relationships*/
	_locateSeeds(seq1, _seeds1);
	seeds1 = &_seeds1[0];

	_locateSeeds(seq2, _seeds2);
	seeds2 = &_seeds2[0];

	/*get the best hits for the left sequence*/
	_getBestHits(_aligner, seq1, _seeds1.size(), seeds1, _minRatio, _bestHits1,
			mapQual1);

	/*get the best hits for the second sequence*/
	_getBestHits(_aligner, seq2, _seeds2.size(), seeds2, _minRatio, _bestHits2,
			mapQual2);

	/*check the distance between top seed pairs*/
	/*****************************************************************
	 * STAGE 1
	 *****************************************************************/
	/*resize the buffer and clear each entry*/
	if (_seedPairs.size() < _bestHits1.size()) {
		_seedPairs.resize(_bestHits1.size() << 1);
	}
	for (size_t i = 0; i < _seedPairs.size(); ++i) {
		_seedPairs[i].clear();
	}

	/*calculate the genome indices*/
	if (_bestHits2.size() > _genomeIndices.size()) {
		_genomeIndices.resize(_bestHits2.size() << 1);
	}
	for (size_t i = 0; i < _bestHits2.size(); ++i) {
		_rgenome->getGenomeIndex(seeds2[_bestHits2[i]]._targetPosition,
				_genomeIndices[i]);
	}

	/*evaluate all top seeds within the allowable insert size*/
	numErrors = _options->getNumErrors(seq1._length);
	for (size_t i = 0; i < _bestHits1.size(); ++i) {
		Seed& left = seeds1[_bestHits1[i]];

		/*get the genome index*/
		_rgenome->getGenomeIndex(left._targetPosition, genomeIndex);

		/*estimate the position of the 5' end*/
		qleft = (left._strand == 0) ?
				left._queryPosition : seq1._length - left._queryPosition - 1;
		for (size_t j = 0; j < _bestHits2.size(); ++j) {
			int32_t seedIndex2 = _bestHits2[j];
			Seed& right = seeds2[seedIndex2];

			/*check the genome index and the sequence strand*/
			if (genomeIndex != _genomeIndices[j]
					|| left._strand == right._strand) {
				continue;
			}
			/*estimate the position of the 5'end*/
			qright =
					(right._strand == 0) ?
							right._queryPosition :
							seq2._length - right._queryPosition - 1;

			/*estimate the distance between the two reads*/
			if (labs(
					(int64_t) left._targetPosition - qleft
							- right._targetPosition + qright)
					<= _maxDistance + 2 * numErrors) {
				/*store the seed pair*/
				_seedPairs[i].push_back(&seeds2[seedIndex2]);
			}
		}
	}

	/*try to get short reads paired through paired seeds*/
	for (size_t i = 0; i < _bestHits1.size(); ++i) {
		vector<Seed*>& seeds = _seedPairs[i];
		/*skip the seed*/
		if (seeds.size() == 0) {
			continue;
		}
		/*get the alignment for the current seed*/
		Seed& left = seeds1[_bestHits1[i]];
		if ((bestMapping1 = _getAlignment(_aligner, seq1, left, _minRatio,
				mapQual1)) == NULL) {
			continue;
		}

		bool paired = false;
		for (size_t j = 0; j < seeds.size(); ++j) {
			Seed& right = *(seeds[j]);
			/*perform alignment for the second sequence*/
			if ((bestMapping2 = _getAlignment(_aligner, seq2, right, _minRatio,
					mapQual2)) == NULL) {
				continue;
			}

			//check the final mapping positions of the alignment*/
			if (labs(bestMapping1->_position - bestMapping2->_position)
					<= _maxDistance) {
				/*the two reads are paired*/
				if (mappings1.size() == 0
						|| mappings1[mappings1.size() - 1]->_position
								!= bestMapping1->_position) {

					pair<set<int64_t>::iterator, bool> ret =
							_mapPositions.insert(bestMapping1->_gposition);
					if (ret.second == true) { /*a new position*/
						mappings1.push_back(bestMapping1);
						mappings2.push_back(bestMapping2);

						/*a seed in the outer loop can only be used once*/
						paired = true;
						break;
					}
				}
			} else {
				delete bestMapping2;
			}

			/*check the number of aligns*/
			if (mappings1.size() >= _maxMultiAligns) {
				break;
			}
		}
		/*if not paired, release the mapping object*/
		if (!paired) {
			delete bestMapping1;
		}
	}
	/*if having found any paired-end alignment*/
	if (mappings1.size() > 0) {
		return true;
	}

	/*if not paired*/
	bestMapping1 = bestMapping2 = NULL;
	/*get the alignment for the left sequence*/
	if (_bestHits1.size() > 0) {
		bestMapping1 = _getAlignment(_aligner, seq1, seeds1[_bestHits1[0]],
				_minRatio, mapQual1);
	}

	/*get the alignment for the right sequence*/
	if (_bestHits2.size() > 0) {
		bestMapping2 = _getAlignment(_aligner, seq2, seeds2[_bestHits2[0]],
				_minRatio, mapQual2);
	}

	/*check the distance between the sequences*/
	good1 = bestMapping1 && bestMapping1->_mapQual >= _mapQualReliable;
	good2 = bestMapping2 && bestMapping2->_mapQual >= _mapQualReliable;
	if (good1 && good2) {
		if (labs(bestMapping1->_position - bestMapping2->_position)
				<= _maxDistance) {
			/*the two reads are paired and return*/
			mappings1.push_back(bestMapping1);
			mappings2.push_back(bestMapping2);
			return true;
		}
	}

	/*try to find the best alignment for the right sequence*/
	float swMinRatio = _minRatio;
	Mapping *mateMapping1 = NULL, *mateMapping2 = NULL;
	if (_options->rescueMate()) {
		if (good1) {
			//Utils::log("map1: %u %u\n", bestMapping1->_position, bestMapping1->_strand);
			window = _options->getNumErrors(seq2._length) * 2;

			/*calcualte the mapping region of the mate read*/
			_getMateRegion(genomeStart1, genomeEnd1, bestMapping1->_gposition,
					bestMapping1->_strand, seq1._length, seq2._length);

			/*refine the region*/
			_rgenome->refineRegionRange(bestMapping1->_genomeIndex,
					genomeStart1, genomeEnd1);

			/*perform the alignment*/
			mateMapping1 = _getAlignment(_aligner, seq2,
					1 - bestMapping1->_strand, window,
					bestMapping1->_genomeIndex, genomeStart1,
					genomeEnd1 - genomeStart1 + 1, swMinRatio, _minIdentity,
					SW_MAP_QUALITY_SCORE * mapQual1);

			/*succeeded in finding an alignment*/
			if (mateMapping1) {
				mappings1.push_back(bestMapping1);
				mappings2.push_back(mateMapping1);
				bestMapping1 = NULL;

				if (mappings1.size() >= _maxMultiAligns) {
					if (bestMapping2) {
						delete bestMapping2;
					}
					return true;
				}
			}
		}

		/*rescue the alignment from the right sequence*/
		if (good2) {
			window = _options->getNumErrors(seq1._length) * 2;
			/*calcualte the mapping region of the mate read*/
			_getMateRegion(genomeStart2, genomeEnd2, bestMapping2->_gposition,
					bestMapping2->_strand, seq2._length, seq1._length);

			/*refine the region*/
			_rgenome->refineRegionRange(bestMapping2->_genomeIndex,
					genomeStart2, genomeEnd2);

			/*perform the alignment*/
			mateMapping2 = _getAlignment(_aligner, seq1,
					1 - bestMapping2->_strand, window,
					bestMapping2->_genomeIndex, genomeStart2,
					genomeEnd2 - genomeStart2 + 1, swMinRatio, _minIdentity,
					SW_MAP_QUALITY_SCORE * mapQual2);

			/*failed to find an alignment*/
			if (mateMapping2) {
				/*output the paired-end alignments*/
				mappings1.push_back(mateMapping2);
				mappings2.push_back(bestMapping2);
				bestMapping2 = NULL;

				if (mappings1.size() >= _maxMultiAligns) {
					if (bestMapping1) {
						delete bestMapping1;
					}
					return true;
				}
			}
		}

		/*check if we have found any paired-end alignment*/
		if (mappings1.size() > 0) {
			if (bestMapping1) {
				delete bestMapping1;
			}
			if (bestMapping2) {
				delete bestMapping2;
			}
			return true;
		}

#ifdef HAVE_TWICE_RESCUE
		/*twice rescuing*/
		if (_options->rescueTwice()) {
			if (good1) {
				window = _options->getNumErrors(seq2._length) * 2;
				/*perform the alignment*/
				mateMapping1 = _getAlignment(_swaligner, seq2,
						1 - bestMapping1->_strand, window,
						bestMapping1->_genomeIndex, genomeStart1,
						genomeEnd1 - genomeStart1 + 1, swMinRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * mapQual1);

				/*succeeded in finding an alignment*/
				if (mateMapping1) {
					/*output the paired-end alignments*/
					mappings1.push_back(bestMapping1);
					mappings2.push_back(mateMapping1);
					bestMapping1=NULL;
					if(mappings1.size() >= _maxMultiAligns) {
						if(bestMapping2) {
							delete bestMapping2;
						}
						return true;
					}
				}
			}
			if (good2) {
				window = _options->getNumErrors(seq1._length) * 2;
				/*perform the alignment*/
				mateMapping2 = _getAlignment(_swaligner, seq1,
						1 - bestMapping2->_strand, window,
						bestMapping2->_genomeIndex, genomeStart2,
						genomeEnd2 - genomeStart2 + 1, swMinRatio, _minIdentity,
						SW_MAP_QUALITY_SCORE * mapQual2);

				/*failed to find an alignment*/
				if (mateMapping2) {
					/*output the paired-end alignments*/
					mappings1.push_back(mateMapping2);
					mappings2.push_back(bestMapping2);
					bestMapping2=NULL;
					if(mappings1.size() >= _maxMultiAligns) {
						if(bestMapping1) {
							delete bestMapping1;
						}
						return true;
					}
				}

			}

			/*if having found any paired-end alignment*/
			if (mappings1.size() > 0) {
				if (bestMapping1) {
					delete bestMapping1;
				}
				if (bestMapping2) {
					delete bestMapping2;
				}
				return true;
			}
		}
#endif
	}

	/*If failed to find any paired-end alignment, will output the best alignments as single-end ones*/
	if (bestMapping1) {
		mappings1.push_back(bestMapping1);
	}
	if (bestMapping2) {
		mappings2.push_back(bestMapping2);
	}
	return false;
}
int32_t MemEngine::_getBestHit(Aligner* aligner, Sequence& seq, size_t numSeeds,
		Seed* seeds, float minBasePortion, int& mapQual) {
	int32_t bestScore, bestScore2, bestSeedIndex;
	size_t totalNseeds, seedIndex;
	bool found = false;
	AlignScore* alignScores;
	int32_t numErrors = _options->getNumErrors(seq._length);
	int64_t targetPosition, targetPosition2;

	/*get the alignment scores*/
	totalNseeds = _getAlignmentScores(aligner, seq, numSeeds, seeds,
			minBasePortion);

	mapQual = 0;
	//Utils::log("totalNSeeds: %d\n", totalNseeds);
	if (totalNseeds > 0) {
		/*select the best alignments*/
		if (_alignScores.size() != totalNseeds) {
			Utils::exit("Inconsistent results\n");
		}
		stable_sort(_alignScores.begin(), _alignScores.end());
		alignScores = &_alignScores[0];

		bestScore = alignScores->_score;
		if (bestScore < _minAlignScore) {
			return -1;
		}
		bestSeedIndex = alignScores->_seedIndex;

		//calculate mapping quality score
		//estimate the starting position of the alignment
		targetPosition = alignScores->_targetPosition;
		targetPosition -= alignScores->_seedPosition;
		++alignScores;
		for (seedIndex = 1; seedIndex < totalNseeds; ++seedIndex) {
			/*get the next best score*/
			bestScore2 = alignScores->_score;
			if (bestScore2 != bestScore) {
				found = true;
				break;
			}

			/*estimate the starting position of the alignment*/
			targetPosition2 = alignScores->_targetPosition;
			targetPosition2 -= alignScores->_seedPosition;
			//Utils::log("pos1 %ld pos2 %ld\n", targetPosition, targetPosition2);
			if (bestScore2 == bestScore
					&& labs(targetPosition - targetPosition2)
							>= _maxGapSize + numErrors) {
				found = true;
				break;
			}
			++alignScores;
		}
		mapQual =
				(found == true) ?
						DEFAULT_MAX_MAP_QUAL * (bestScore - bestScore2)
								/ bestScore :
						DEFAULT_MAX_MAP_QUAL;

#if 0
		/*print out the alignment scores*/
		Utils::log("-------------Seed Score:----------------\n");
		Utils::log("mapping quality: %d\n", mapQual);
		alignScores = &_alignScores[0];
		for (size_t i = 0; i < totalNseeds; ++i)
		{
			Utils::log("query %d strand %d score %d target %u\n",
					seeds[alignScores[i]._seedIndex]._queryPosition,
					seeds[alignScores[i]._seedIndex]._strand,
					alignScores[i]._score, alignScores[i]._targetPosition);
		}
#endif

		return bestSeedIndex;
	}
	return -1;
}

/*for paired-end alignment*/
int32_t MemEngine::_getBestHits(Aligner* aligner, Sequence& seq,
		size_t numSeeds, Seed* seeds, float minBasePortion,
		vector<int32_t>& bestHits, int& mapQual) {
	size_t totalNseeds, seedIndex;
	int32_t bestScore, bestScore2, bestScoreDiff;
	bool found = false;
	int64_t targetPosition, targetPosition2;
	AlignScore* alignScores;
	int32_t numErrors = _options->getNumErrors(seq._length);

	/*get the alignment scores*/
	totalNseeds = _getAlignmentScores(aligner, seq, numSeeds, seeds,
			minBasePortion);

	mapQual = 0;
	bestHits.clear();
	if (totalNseeds > 0) {
		/*select the best alignments*/
		if (_alignScores.size() != totalNseeds) {
			Utils::exit("Inconsistent results\n");
		}
		stable_sort(_alignScores.begin(), _alignScores.end());
		alignScores = &_alignScores[0];

		/*select the best hits*/
		bestScore = alignScores->_score;

		/*check the minimal alignment score*/
		if (bestScore < _minAlignScore) {
			return 0;
		}
		//select the top seeds
		bestScoreDiff = bestScore;

		//estimate the starting position of the alignment
		targetPosition = alignScores->_targetPosition;
		targetPosition -= alignScores->_seedPosition;
		for (seedIndex = 0; seedIndex < totalNseeds; ++seedIndex) {

			/*get the next best score*/
			bestScore2 = alignScores->_score;
			if (bestScore2 != bestScore) {
				if (!found) {
					found = true;
					/*optimal local alignment score diff*/
					bestScoreDiff = bestScore - bestScore2;
				}
				break;
			}
			/*estimate the starting position of the alignment*/
			targetPosition2 = alignScores->_targetPosition;
			targetPosition2 -= alignScores->_seedPosition;
			if (!found
					&& (bestScore2 == bestScore
							&& labs(targetPosition - targetPosition2)
									>= _maxGapSize + numErrors)) {
				found = true;
				/*optimal local aignment score diff*/
				bestScoreDiff = 0;
			}
			bestHits.push_back(alignScores->_seedIndex);
			++alignScores;
		}
		//calculate mapping quality score
		mapQual =
				(found == true) ?
						DEFAULT_MAX_MAP_QUAL * bestScoreDiff / bestScore :
						DEFAULT_MAX_MAP_QUAL;

		/*adjust mapping quality scores*/
		if (mapQual < _mapQualReliable)
			mapQual = 0;
	}

#if 0
	/*print out the alignment scores*/
	Utils::log("-------------Top seeds:----------------\n");
	alignScores = &_alignScores[0];
	for (size_t i = 0; i < bestHits.size(); ++i)
	{
		Seed* p = &seeds[bestHits[i]];
		fprintf(stderr, "%u %u %u %u %u %u\n", p->_queryPosition,
				p->_targetPosition, p->_seedLength, p->_strand, p->_best, alignScores[i]._score);
	}
#endif

	return bestHits.size();
}
size_t MemEngine::_getAlignmentScores(Aligner* aligner, Sequence& seq,
		size_t numSeeds, Seed* seeds, float minBasePortion) {
	int32_t genomeIndex;
	int64_t lowerBound, upperBound, genomeLength;
	int32_t nseeds, offset;
	size_t totalNseeds = 0;
	size_t seedIndex;
	AlignScore* scorePtr;
	uint8_t* sequences;
	int32_t* seqOffsets;
	uint8_t* ptr = NULL;

	/*if there are no seeds*/
	if (numSeeds == 0) {
		_alignScores.clear();
		return 0;
	}

	/*re-allocate buffers*/
	_alignScores.resize(numSeeds);
	_sequences.resize(
			numSeeds
					* (_mapRegionSizeFactor * (((seq._length + 3) >> 2) << 2)
							+ 1));
#ifdef USE_FULL_SW_64
	_seqOffsets.resize(numSeeds + 1);
#else
	_seqOffsets.resize(numSeeds);
#endif

	/*calculate alignment score for all seed extensions*/
	scorePtr = &_alignScores[0];
	sequences = &_sequences[0];
	seqOffsets = &_seqOffsets[0];
	/*evaluate all seeds*/
	for (int strand = 0; strand < 2; ++strand) {
		/*for this strand*/
		nseeds = 0;
		offset = 0;
		for (seedIndex = 0; seedIndex < numSeeds; ++seedIndex) {
			Seed& seed = seeds[seedIndex];
			//Utils::log("strand: %d, tlength: %d length %d\n", seed._strand, seq._tlength, seq._length);
			if (seed._strand != strand) {
				continue;
			}

			/*perform banded local alignment using the highest-scoring seeds*/
			lowerBound = seed._targetPosition;
			lowerBound -= _mapRegionSizeFactor * (seed._queryPosition + 1);
			upperBound = seed._targetPosition;
			upperBound += seed._seedLength
					+ _mapRegionSizeFactor
							* (seq._length - seed._queryPosition
									- seed._seedLength);

			//Utils::log("[before] lowerBound: %ld upperBound: %ld seedLength %ld\n", lowerBound, upperBound, seed._seedLength);
			/*refine the genome region*/
			/*get the genome index*/
			_rgenome->getGenomeIndex(seed._targetPosition, genomeIndex);
			_rgenome->refineRegionRange(genomeIndex, lowerBound, upperBound);

			genomeLength = upperBound - lowerBound + 1;
			if (genomeLength <= 0) {
				continue;
			}
			/*save the offset*/
			seqOffsets[nseeds] = offset;

			/*save the seed index*/
			scorePtr[nseeds]._seedLength = seed._seedLength;
			scorePtr[nseeds]._seedPosition = seed._queryPosition;
			scorePtr[nseeds]._seedIndex = seedIndex;
			scorePtr[nseeds]._targetPosition = seed._targetPosition;
			++nseeds;

			/*load the target sequence slice*/
			/*check the sequence length*/
			ptr = sequences + offset;
			for (int64_t i = 0, j = lowerBound; i < genomeLength; ++i, ++j) {
				ptr[i] = ((_pacGenome[j >> 2] >> ((~j & 3) << 1)) & 3) + 1;
			}
			int64_t pGenomeLength = ((genomeLength + 3) >> 2) << 2;
			for (int64_t i = genomeLength; i < pGenomeLength; ++i) {
				ptr[i] = 6;
			}
			ptr[pGenomeLength] = 0; /*separate the sequences*/

			/*increase the offset*/
			offset += pGenomeLength + 1;
		}
#ifdef USE_FULL_SW_64
		seqOffsets[nseeds] = offset;
#endif
		/*calculate the optimal local alignments*/
		aligner->lalignScore((strand == 0) ? seq._bases : seq._rbases,
				seq._length, sequences, seqOffsets, nseeds, scorePtr);

		/*update the score pointer*/
		scorePtr += nseeds;

		/*calculate the total number of valid seeds*/
		totalNseeds += nseeds;
	}
	_alignScores.resize(totalNseeds);
	return totalNseeds;
}
Mapping* MemEngine::_getAlignment(Aligner* aligner, Sequence& seq, Seed& seed,
		float minBasePortion, int mapQual) {
	int32_t window, genomeIndex;
	vector<CigarAlign*> aligns;
	int64_t lowerBound, upperBound;

	/*get the genome index*/
	_rgenome->getGenomeIndex(seed._targetPosition, genomeIndex);

	/*check if it is an exact-match*/
	if (seed._seedLength == seq._length) {
		/*return an exact-match mapping*/
		return new Mapping(new CigarAlign(seq),
				seed._targetPosition - _rgenome->getGenomeOffset(genomeIndex)
						+ 1, seed._targetPosition, seed._strand, genomeIndex,
				mapQual);
	}

	/*calculate the banded window width for banded local/global alignment*/
	window = _options->getNumErrors(seq._length) * 2;

	/*perform banded local alignment using the highest-scoring seeds*/

	lowerBound = seed._targetPosition;
	lowerBound -= _mapRegionSizeFactor * (seed._queryPosition + 1);
	upperBound = seed._targetPosition;
	upperBound += seed._seedLength
			+ _mapRegionSizeFactor
					* (seq._length - seed._queryPosition - seed._seedLength);

//Utils::log("[before] lowerBound: %ld upperBound: %ld seedLength %ld\n", lowerBound, upperBound, seed._seedLength);
	/*refine the genome region*/
	_rgenome->refineRegionRange(genomeIndex, lowerBound, upperBound);
//Utils::log("[after] lowerBound: %ld upperBound: %ld seedLength %ld\n", lowerBound, upperBound, seed._seedLength);

	int64_t genomeLength = upperBound - lowerBound + 1;
	if (genomeLength <= 0) {
		return NULL;
	}
	/*get the short read alignment*/
	return _getAlignment(aligner, seq, seed._strand, window, genomeIndex,
			lowerBound, genomeLength, minBasePortion, _minIdentity, mapQual);
}

/*normal single-end alignment*/
Mapping* MemEngine::_getAlignment(Aligner* aligner, Sequence& seq, int strand,
		int window, int genomeIndex, uint32_t genomeStart, size_t genomeLength,
		float minBasePortion, float minIdentity, int mapQual) {
	uint8_t* bases;
	int low, up;
	int diff;
	vector<CigarAlign*> aligns;
	Mapping* mapping = NULL;

	/*check the number of leading and trailing bases*/
	bases = (strand == 0) ? seq._bases : seq._rbases;

//Utils::log("genome start: %u genome length %u\n", genomeStart, genomeLength);
	/*get the target sequence*/
	if (genomeLength >= _targetSize) {
		/*resize the target*/
		_targetSize = genomeLength << 1;
		if (_target) {
			delete[] _target;
		}
		_target = new uint8_t[_targetSize];
		if (_target == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}
	/*load the target sequence slice*/
	for (uint32_t i = 0, j = genomeStart; i < genomeLength; ++i, ++j) {
		_target[i] = (_pacGenome[j >> 2] >> ((~j & 3) << 1)) & 3;
	}

	/*calculate the low and up*/
	bool swapped = false;
	if (genomeLength >= seq._length) {
		diff = genomeLength - seq._length;
	} else {
		diff = seq._length - genomeLength;
		swapped = true;
	}
	/*make sure that low <= min(0, -diff) and up >= max(0, -diff)*/
	low = -diff - window / 2;
	up = window + low;
	if (up < window) {
		up = window;
	}

//Utils::log("diff: %d window %d low %d up:%d\n", diff, window, low, up);

	/*perform banded local alignment and output the alignment for the query*/
	if (swapped == false) {
		/*the second  sequence is the query*/
		aligns = aligner->lalignPath(_target, bases, genomeLength, seq._length,
				low, up, 2);
	} else {

		/*the first sequence is the query*/
		aligns = aligner->lalignPath(bases, _target, seq._length, genomeLength,
				low, up, 1);
	}

	/*failed to find an alignment*/
	if (aligns.size() == 0) {
		return NULL;
	}

	/*check the validity of the alignment*/
//Utils::log("%d %d %f\n", aligns[0]->getNumBases1(), (int)(minBasePortion * seq._length), aligns[0]->getIdentity());
	if (aligns[0]->getNumBases1() >= minBasePortion * seq._length
			&& aligns[0]->getIdentity() >= minIdentity) {
		/*extend the alignment*/
		aligns[0]->extendCigar(bases, seq._length);

		/*mapping positions on the target sequence, indexed from 1*/
		//Utils::log("genomeStart: %ld start: %d index %ld\n", genomeStart, aligns[0]->getMateStart(), _rgenome->getGenomeOffset(genomeIndex));
		int64_t gPosition = genomeStart + aligns[0]->getMateStart();
		int64_t mapPosition = gPosition - _rgenome->getGenomeOffset(genomeIndex)
				+ 1;

		/*it is possible that the mapping position is less than 0 after extending the alignment to the begining*/
		if (mapPosition >= 0) {
			mapping = new Mapping(aligns[0], mapPosition, gPosition, strand,
					genomeIndex,
					mapQual * aligns[0]->getNumBases1() / seq._length);
			aligns[0] = NULL;
		} else {
			Utils::log("negative mapping position\n");
		}
	}

	/*release the alignments*/
	for (size_t i = 0; i < aligns.size(); ++i) {
		if (aligns[i]) {
			delete aligns[i];
		}
	}
	aligns.clear();

	return mapping;
}

/*for the read rescue using paired-end information*/
Mapping* MemEngine::_getAlignment(Aligner* aligner, Sequence& seq, int strand,
		int window, int genomeIndex, uint32_t genomeStart, size_t genomeLength,
		float minMatchLengthRatio, int mapQual) {
	uint8_t* bases;
	int low, up;
	int diff;
	vector<CigarAlign*> aligns;
	Mapping* mapping = NULL;
	int minMatchLength = (int) (minMatchLengthRatio * seq._length);

	if (genomeLength < (size_t) (minMatchLength)) {
		return NULL;
	}

	/*check the number of leading and trailing bases*/
	bases = (strand == 0) ? seq._bases : seq._rbases;

//Utils::log("genome start: %u genome length %u\n", genomeStart, genomeLength);
	/*get the target sequence*/
	if (genomeLength >= _targetSize) {
		/*resize the target*/
		_targetSize = genomeLength << 1;
		if (_target) {
			delete[] _target;
		}
		_target = new uint8_t[_targetSize];
		if (_target == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}
	/*load the target sequence slice*/
	for (uint32_t i = 0, j = genomeStart; i < genomeLength; ++i, ++j) {
		_target[i] = (_pacGenome[j >> 2] >> ((~j & 3) << 1)) & 3;
	}

	/*calculate the low and up*/
	bool swapped = false;
	if (genomeLength >= seq._length) {
		diff = genomeLength - seq._length;
	} else {
		diff = seq._length - genomeLength;
		swapped = true;
	}
	/*make sure that low <= min(0, -diff) and up >= max(0, -diff)*/
	low = -diff - window / 2;
	up = window + low;
	if (up < window) {
		up = window;
	}

//Utils::log("diff: %d window %d low %d up:%d\n", diff, window, low, up);

	/*perform banded local alignment and output the alignment for the query*/
	if (swapped == false) {
		/*the second  sequence is the query*/
		aligns = aligner->lalignPath(_target, bases, genomeLength, seq._length,
				low, up, 2);
	} else {

		/*the first sequence is the query*/
		aligns = aligner->lalignPath(bases, _target, seq._length, genomeLength,
				low, up, 1);
	}

	/*failed to find an alignment*/
	if (aligns.size() == 0) {
		return NULL;
	}

	/*validate the alignment*/
	if (aligns[0]->getNumBases1() >= minMatchLength
			&& aligns[0]->getNumBases2() >= minMatchLength) {
		/*extend the alignment*/
		aligns[0]->extendCigar(bases, seq._length);

		/*mapping positions on the target sequence, indexed from 1*/
		//Utils::log("genomeStart: %ld start: %d index %ld\n", genomeStart, aligns[0]->getMateStart(), _rgenome->getGenomeOffset(genomeIndex));
		int64_t gPosition = genomeStart + aligns[0]->getMateStart();
		int64_t mapPosition = gPosition - _rgenome->getGenomeOffset(genomeIndex)
				+ 1;

		/*it is possible the mapping position is less than 0 after extending the alignment to the begining*/
		if (mapPosition >= 0) {
			mapping = new Mapping(aligns[0], mapPosition, gPosition, strand,
					genomeIndex,
					mapQual * aligns[0]->getNumBases1() / seq._length);
			aligns[0] = NULL;
		} else {
			Utils::log("negative mapping position\n");
		}
	}

	/*release the alignments*/
	for (size_t i = 0; i < aligns.size(); ++i) {
		if (aligns[i]) {
			delete aligns[i];
		}
	}
	aligns.clear();

	return mapping;
}

void MemEngine::_genMEMSeeds(Sequence& seq, int strand, uint4* ranges,
		size_t& numRanges, uint32_t minSeedSize) {
	uint8_t ch;
	bool lastStopAtN;
	uint2 range, range2;
	/*For each position of the sequence, we search the maximal exact match*/
	uint32_t startPos, endPos, lastPos;
	uint8_t* bases = (strand == 0) ? seq._bases : seq._rbases;
	uint32_t numSuffixes = seq._length - minSeedSize + 1;

	startPos = 0;
	lastPos = 0;
	numRanges = 0;
	while (startPos < numSuffixes) {
		//for each starting position in the query
		lastStopAtN = false;
		range = make_uint2(0, _bwtSeqLength);
		range2 = make_uint2(1, 0);
		for (endPos = startPos; endPos < seq._length; endPos++) {
			/*get the base*/
			ch = bases[endPos];

			/*unknown bases*/
			if (ch == BWT_NUM_NUCLEOTIDE) {
				lastStopAtN = true;
				range2 = make_uint2(1, 0);
				break;
			}

			//calculate the range
			range2.x = _rbwt->_bwtCCounts[ch] + _rbwt->bwtOcc(ch, range.x - 1)
					+ 1;
			range2.y = _rbwt->_bwtCCounts[ch] + _rbwt->bwtOcc(ch, range.y);
			if (range2.x > range2.y) {
				break;
			}
			range = range2;
		}
		/*If an exact match is found to the end of the query, no need to check any more*/
		if (range2.x <= range2.y) {
			//Utils::log("seed length %d min seed length %d #repeats: %d\n", endPos - startPos, minSeedSize, range2.y - range2.x + 1);
			if (range2.y - range2.x >= _maxSeedOcc) {
				range2.y = range2.x + _maxSeedOcc - 1;
			}
			ranges[numRanges++] = make_uint4(range2.x, range2.y, startPos,
					endPos - startPos);
			break;
		} else if (range.x <= range.y && endPos - startPos >= minSeedSize)/*a mismatch found and record the exact matches to the previous base*/
		{
			if (range.y - range.x >= _maxSeedOcc) {
				range.y = range.x + _maxSeedOcc - 1;
			}
			//Utils::log("seed length %d min seed length %d #repeats: %d\n", endPos - startPos, minSeedSize, range.y - range.x + 1);
			if (lastPos != endPos) {
				ranges[numRanges++] = make_uint4(range.x, range.y, startPos,
						endPos - startPos);
				/*update the last stop position*/
				lastPos = endPos;
			}

		}
		/*filter out consecutive Ns and re-locate the starting position*/
		if (lastStopAtN) {
			for (startPos = endPos + 1; startPos < numSuffixes; startPos++) {
				if (bases[startPos] != BWT_NUM_NUCLEOTIDE) {
					break;
				}
			}
		} else {
			startPos++;
		}
	}
	/*check if there are long enough seeds available*/
}
void MemEngine::_locateSeeds(Sequence& seq, vector<Seed>& seeds) {
	Seed* seedsPtr;
	size_t numSeeds;
	int64_t target;
	uint32_t minSeedSize;
	uint4 *ranges, *rranges;
	size_t numFwSeeds, numRcSeeds;
	size_t numSuffixes, numRanges, numRranges;

	numFwSeeds = numRcSeeds = 0;
	minSeedSize = _options->getMinSeedSize(seq._length);
	if (seq._tlength < minSeedSize) {
		seeds.clear();
		return;
	}

	/*allocate space for the suffix array interval*/
	numSuffixes = seq._length - minSeedSize + 1;
	ranges = new uint4[numSuffixes];
	rranges = new uint4[numSuffixes];

	/*for the forward strand*/
	_genMEMSeeds(seq, 0, ranges, numRanges, minSeedSize);

	/*for the reverse strand*/
	_genMEMSeeds(seq, 1, rranges, numRranges, minSeedSize);

	/*check the availability of seeds*/
//Utils::log("line %d numRanges: %d numRranges: %d\n", __LINE__, numRanges, numRranges);
	if (numRanges + numRranges == 0) {
		uint32_t newSeedSize = (minSeedSize + GLOBAL_MIN_SEED_SIZE) / 2;
		/*for the forward strand*/
		_genMEMSeeds(seq, 0, ranges, numRanges, newSeedSize);

		/*for the reverse strand*/
		_genMEMSeeds(seq, 1, rranges, numRranges, newSeedSize);
	}
//Utils::log("line %d numRanges: %d numRranges: %d\n", __LINE__, numRanges, numRranges);
	/*failed to find any satisfactory seeds*/
	if (numRanges + numRranges == 0) {
		/*release resources*/
		delete[] rranges;
		delete[] ranges;
		seeds.clear();
		return;
	}

	/*calculate the total number of seeds*/
	/*for the forward strand*/
	for (uint4* p = ranges, *q = ranges + numRanges; p < q; ++p) {
		numFwSeeds += p->y - p->x + 1;
	}
	/*for the reverse strand*/
	for (uint4* p = rranges, *q = rranges + numRranges; p < q; ++p) {
		numRcSeeds += p->y - p->x + 1;
	}
	numSeeds = numFwSeeds + numRcSeeds;

	/*allocate memory*/
	seeds.resize(numSeeds);
	seedsPtr = &seeds[0];

	/*calculate the mapping seeds for each occurrence of the seed*/
	Seed* t = seedsPtr;
	/*for the forward strand*/
	for (uint4 *p = ranges, *q = ranges + numRanges; p < q; ++p) {
		//for each suffix array interval
		for (uint32_t pos = p->x; pos <= p->y; ++pos) {
//get the mapping position in the reverse target sequence
			target = _rsa->getPosition(_rbwt, pos);
//get the mapping position in the forward target sequence
			target = _bwtSeqLength - target - p->w;

			/*save the mapping seeds*/
			t->_targetPosition = target;
			t->_queryPosition = p->z;
			t->_seedLength = p->w;
			t->_strand = 0;
			t->_best = 0; /*by default not the best*/
			/*increase the pointer address*/
			++t;
		}
	}
	/*for the reverse complement*/
	for (uint4 *p = rranges, *q = rranges + numRranges; p < q; ++p) {
		//for each suffix array interval
		for (uint32_t pos = p->x; pos <= p->y; ++pos) {
//get the mapping position in the reverse target sequence
			target = _rsa->getPosition(_rbwt, pos);
//get the mapping position in the forward target sequence
			target = _bwtSeqLength - target - p->w;

			/*save the mapping seeds*/
			t->_targetPosition = target;
			t->_queryPosition = p->z;
			t->_seedLength = p->w;
			t->_strand = 1;
			t->_best = 0; //by default, not the best*/
			/*increase the pointer address*/
			++t;
		}
	}

	/*release the buffers*/
	delete[] ranges;
	delete[] rranges;
#if 0
	fprintf(stderr, "query, target, slen, strand, best\n");
	for (Seed* p = seedsPtr, *q = seedsPtr + numSeeds; p < q; ++p)
	{
		fprintf(stderr, "%u %u %u %u %u\n", p->_queryPosition,
				p->_targetPosition, p->_seedLength, p->_strand, p->_best);
	}
	fprintf(stderr, "---------------#Seeds %ld -----------------------\n",
			numSeeds);

#endif
}

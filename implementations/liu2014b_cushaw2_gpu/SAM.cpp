/*
 * SAM.cpp
 *
 *  Created on: Jan 5, 2012
 *      Author: yongchao
 */
#include "SAM.h"
#include "Utils.h"

SAM::SAM(Options* options, Genome* genome) {
	_fileName = options->getSamFileName();
	_numThreads = options->getNumThreads();
	_file = NULL;
	_genome = genome;

	/*open the file*/
	open();

	/*print out the header*/
	fprintf(_file, "@HD\tVN:1.0\tSO:unsorted\n");

	/*print out the read group header information*/
	if (options->_rgID.size() > 0) {
		/*tag ID*/
		fprintf(_file, "@RG\tID:%s\tSM:%s", options->_rgID.c_str(),
				options->_rgSM.c_str());
		/*tag LB*/
		if (options->_rgLB.size() > 0) {
			fprintf(_file, "\tLB:%s", options->_rgLB.c_str());
		}
		/*tag PL*/
		if (options->_rgPL.size() > 0) {
			fprintf(_file, "\tPL:%s", options->_rgPL.c_str());
		}
		/*tag PU*/
		if (options->_rgPU.size() > 0) {
			fprintf(_file, "\tPU:%s", options->_rgPU.c_str());
		}
		/*tag CN*/
		if (options->_rgCN.size() > 0) {
			fprintf(_file, "\tCN:%s", options->_rgCN.c_str());
		}
		/*tag DS*/
		if (options->_rgDS.size() > 0) {
			fprintf(_file, "\tDS:%s", options->_rgDS.c_str());
		}
		/*tag DT*/
		if (options->_rgDT.size() > 0) {
			fprintf(_file, "\tDT:%s", options->_rgDT.c_str());
		}
		/*tag PI*/
		if (options->_rgPI.size() > 0) {
			fprintf(_file, "\tPI:%s", options->_rgPI.c_str());
		}

		/*end of the line*/
		fputc('\n', _file);
	}

	/*print out the genome sequence information in SAM format*/
	_genome->genomeNamesOut(_file);

	/*print out the aligner information*/
	fprintf(_file, "@PG\tID:cushaw2-gpu\tVN:%s\n", CUSHAW2_VERSION);
}
SAM::~SAM() {
	/*close the file*/
	close();
}
void SAM::open() {
	/*if the file is open, close it*/

	if (_file) {
		close();
	}
	/*re-open it*/
	if (_fileName.length() > 0) {
		/*open the output file*/
		_file = fopen(_fileName.c_str(), "wb");
		fseek(_file, 0, SEEK_SET);

	} else {
		/*open the standard output*/
		_file = stdout;
	}

	if (_file == NULL) {
		Utils::exit("Failed to open file %s in function %s line %d\n",
				_fileName.c_str(), __FUNCTION__, __LINE__);
	}
}
void SAM::close() {

	if (_file) {
		fclose(_file);
	}
	_file = NULL;

}
/*print the unaligned read information*/
void SAM::print(Sequence& seq, int _flags) {
	int flags = SAM_FSU | _flags;

	/*print out query name, bitwise-flag, and reference sequence name*/
	fprintf(_file, "%s\t%d\t*\t", seq._name, flags);

	//print 1-based leftmost mapping position, mapping quality (phred-scale)*/
	fprintf(_file, "0\t0\t");

	//print extended CIGAR
	fputs("*\t", _file);

	//print paired-end information, INAVAILABLE
	fprintf(_file, "*\t0\t0\t");

	//print the query sequence
	uint8_t* bases = seq._bases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals = seq._quals;
		for (uint32_t i = 0; i < seq._length; ++i) {
			fputc(*quals, _file);
			++quals;
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);

}
/*print the alignment information through the dynamic programming*/
void SAM::print(Sequence& seq, Mapping& mapping, int _flags) {
	int32_t flags = _flags;
	CigarAlign* align = mapping._align;

	/*check the sequence strand*/
	if (mapping._strand) {
		flags |= SAM_FSR;
	}

	/*print out query name, bitwise-flag, and reference sequence name*/
	fprintf(_file, "%s\t%d\t%s\t", seq._name, flags,
			_genome->getGenomeName(mapping._genomeIndex));

	//print 1-based leftmost mapping position, mapping quality (phred-scale)*/
	fprintf(_file, "%u\t%d\t", (uint32_t) mapping._position, mapping._mapQual);

	//print extended CIGAR
	if (align->getCigar()) {
		align->cigarOut(_file);
	} else {
		fprintf(_file, "%dM", (int) seq._length);
	}
	fputc('\t', _file);

	//print paired-end information, INAVAILABLE
	fprintf(_file, "*\t0\t0\t");

	//print the query sequence
	uint8_t* bases = (mapping._strand == 0) ? seq._bases : seq._rbases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals;
		if (mapping._strand == 0) {
			quals = seq._quals;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals, _file);
				++quals;
			}
		} else {
			quals = seq._quals + seq._length - 1;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals, _file);
				--quals;
			}
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);

}
void SAM::print(Sequence& seq, Mapping& self, Mapping& mate, int _flags) {
	int flags = _flags;
	CigarAlign* align = self._align;

	flags |= SAM_FPD | SAM_FPP; //this function only print reads that have been properly paired

	//check the strand of itself
	if (self._strand) {
		flags |= SAM_FSR;
	}

	//check the strand of the mate
	if (mate._strand) {
		flags |= SAM_FMR;
	}

	//print query-name, bitwise-flag, and reference-sequence-name
	fprintf(_file, "%s\t%d\t%s\t", seq._name, flags,
			_genome->getGenomeName(self._genomeIndex));

	//print 1-based leftmost mapping position, mapping quality (phred-scaled)
	fprintf(_file, "%d\t%d\t", (int) self._position, self._mapQual);

	//print extended CIGAR if applicable
	if (align->getCigar()) {
		align->cigarOut(_file);
	} else {
		fprintf(_file, "%dM", (int) seq._length);
	}
	fputc('\t', _file);

	/****************************
	 print the mate information
	 1-base mate mapping position and estimated insert size)
	 *****************************/
	//print the reference sequence mapped by the mate
	fprintf(_file, "%s\t",
			(self._genomeIndex == mate._genomeIndex) ?
					"=" : _genome->getGenomeName(mate._genomeIndex));

	//print the mapping position of the mate and the distance
	int64_t distance;
	if (self._strand == 0) {
		distance = (int64_t) self._position - mate._position - seq._length;
	} else {
		distance = (int64_t) self._position + seq._length - mate._position;
	}

	fprintf(_file, "%d\t%ld\t", (int) mate._position, distance);

	//print the query sequence on the same strand as the reference sequence
	uint8_t* bases = (self._strand == 0) ? seq._bases : seq._rbases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals;
		if (self._strand == 0) {
			quals = seq._quals;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals, _file);
				++quals;
			}
		} else {
			quals = seq._quals + seq._length - 1;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals, _file);
				--quals;
			}
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);
}

void SAM::print(Sequence& seq, Mapping& self, int32_t mateGenomeIndex,
		int32_t matePosition, int32_t insertSize, int _flags) {
	int32_t offset = 0;
	int flags = _flags;
	CigarAlign* align = self._align;

	flags |= SAM_FPD | SAM_FPP; //this function only print reads that have been properly paired

	//check the strand of itself
	if (self._strand) {
		flags |= SAM_FSR;
	} else {
		flags |= SAM_FMR;
	}

	//print query-name, bitwise-flag, and reference-sequence-name
	fprintf(_file, "%s\t%d\t%s\t", seq._name, flags,
			_genome->getGenomeName(self._genomeIndex));

	//print 1-based leftmost mapping position, mapping quality (phred-scaled)
	fprintf(_file, "%d\t%d\t", (int) self._position, self._mapQual);

	//print extended CIGAR if applicable
	if (align->getCigar()) {
		align->cigarOut(_file);
	} else {
		fprintf(_file, "%dM", (int) seq._length);
	}
	fputc('\t', _file);

	/****************************
	 print the mate information
	 1-base mate mapping position and estimated insert size)
	 *****************************/
	//print the reference sequence mapped by the mate
	fprintf(_file, "%s\t",
			(self._genomeIndex == mateGenomeIndex) ?
					"=" : _genome->getGenomeName(mateGenomeIndex));

	//print the mapping position of the mate and the distance
	fprintf(_file, "%d\t%ld\t", (int) matePosition, (long) insertSize);

	//print the query sequence on the same strand as the reference sequence
	uint8_t* bases = (self._strand == 0) ? seq._bases : seq._rbases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals;
		if (self._strand == 0) {
			quals = seq._quals;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals++, _file);
			}
		} else {
			quals = seq._quals + seq._length - 1;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals--, _file);
			}
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);

}

/*for GPU computing*/
void SAM::print(Sequence& seq, uint16_t* cigars, uint32_t numCigars,
		int32_t genomeIndex, int64_t mapPosition, uint32_t strand,
		int32_t mapQual, int _flags) {

	int32_t offset = 0;
	int32_t flags = _flags;

	/*check the sequence strand*/
	if (strand) {
		flags |= SAM_FSR;
	}

	/*print out query name, bitwise-flag, and reference sequence name*/
	fprintf(_file, "%s\t%d\t%s\t", seq._name, flags,
			_genome->getGenomeName(genomeIndex));

//print 1-based leftmost mapping position, mapping quality (phred-scale)*/
	fprintf(_file, "%u\t%d\t", (uint32_t) mapPosition, mapQual);

	/*print extended CIGAR*/
	if (numCigars > 0) {
		uint16_t cigar;
		for (uint32_t i = 0; i < numCigars; ++i) {
			uint16_t cigar = cigars[i];
			/*print the cigar to the file*/
			fprintf(_file, "%d%c", cigar >> 2,
					CigarAlign::_alignOpName[cigar & 3]);
		}
	} else {
		fprintf(_file, "%dM", (int) seq._length);
	}
	fputc('\t', _file);
	;

	//print paired-end information, INAVAILABLE
	fprintf(_file, "*\t0\t0\t");

//print the query sequence
	uint8_t* bases = (strand == 0) ? seq._bases : seq._rbases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals;
		if (strand == 0) {
			quals = seq._quals;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals++, _file);
			}
		} else {
			quals = seq._quals + seq._length - 1;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals--, _file);
			}
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);
}

void SAM::print(Sequence& seq, uint16_t* cigars, uint32_t numCigars,
		int32_t genomeIndex, int64_t mapPosition, uint32_t strand,
		int32_t mapQual, int32_t mateGenomeIndex, int32_t matePosition,
		int32_t insertSize, int _flags) {

	int32_t offset = 0;
	int32_t flags = _flags;

	/*check the sequence strand*/
	if (strand) {
		flags |= SAM_FSR;
	} else {
		flags |= SAM_FMR;
	}

	/*print out query name, bitwise-flag, and reference sequence name*/
	fprintf(_file, "%s\t%d\t%s\t", seq._name, flags,
			_genome->getGenomeName(genomeIndex));

//print 1-based leftmost mapping position, mapping quality (phred-scale)*/
	fprintf(_file, "%u\t%d\t", (uint32_t) mapPosition, mapQual);

	/*print extended CIGAR*/
	if (numCigars > 0) {
		uint16_t cigar;
		for (uint32_t i = 0; i < numCigars; ++i) {
			uint16_t cigar = cigars[i];
			/*print the cigar to the file*/
			fprintf(_file, "%d%c", cigar >> 2,
					CigarAlign::_alignOpName[cigar & 3]);
		}
	} else {
		fprintf(_file, "%dM", (int) seq._length);
	}
	fputc('\t', _file);

	/****************************
	 print the mate information
	 1-base mate mapping position and estimated insert size)
	 *****************************/
	//print the reference sequence mapped by the mate
	fprintf(_file, "%s\t",
			(genomeIndex == mateGenomeIndex) ?
					"=" : _genome->getGenomeName(mateGenomeIndex));

	//print the mapping position of the mate and the distance
	fprintf(_file, "%d\t%ld\t", (int) matePosition, (long) insertSize);

//print the query sequence
	uint8_t* bases = (strand == 0) ? seq._bases : seq._rbases;
	for (uint32_t i = 0; i < seq._length; ++i) {
		fputc(decode(bases[i]), _file);
	}
	fputc('\t', _file);

	//print the quality scores if available
	if (seq._quals) {
		uint8_t* quals;
		if (strand == 0) {
			quals = seq._quals;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals++, _file);
			}
		} else {
			quals = seq._quals + seq._length - 1;
			for (uint32_t i = 0; i < seq._length; ++i) {
				fputc(*quals--, _file);
			}
		}
	} else {
		fputc('*', _file);
	}
	fputc('\n', _file);

}

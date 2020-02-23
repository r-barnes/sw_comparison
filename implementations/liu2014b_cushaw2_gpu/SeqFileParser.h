/*
 * SeqFileParser.h
 *
 *  Created on: Dec 23, 2011
 *      Author: yongchao
 */

#ifndef SEQFILEPARSER_H_
#define SEQFILEPARSER_H_

#include "Macros.h"
#include "Sequence.h"
#include "Utils.h"
#include "MyFile.h"
extern "C" {
#include "sam.h"
}

class SeqFileParser
{
public:
	/*public member functions*/
	SeqFileParser(const char* path, bool withLock, int type,
			size_t BUFFER_SIZE = 4095);
	~SeqFileParser();

	static inline void finalize()
	{
		/*close the FIFO file*/
		if (_fifoFile != -1) {
			close(_fifoFile);
		}
	}

	//get the next sequence from the file
	inline size_t getSeq(Sequence& seq) {
		size_t ret;
		/*read the sequence from the file*/
		_lock();
		if (_format == FILE_FORMAT_FASTA) {
			ret = getFastaSeq(seq);
		} else if (_format == FILE_FORMAT_FASTQ) {
			ret = getFastqSeq(seq);
		} else if (_format == FILE_FORMAT_BSAM) { /*BAM/SAM format*/
			ret = getBSamSeq(seq);
		} else {
			ret = getFifoSeq(seq);
		}
		_unlock();

		/*check the sequence length*/
		if(seq._length > MAX_SEQ_LENGTH){
			Utils::log("trimm the reads (of %d bases) longer than %d\n", seq._length, MAX_SEQ_LENGTH);
			seq._length = MAX_SEQ_LENGTH;
		}

		/*compute the reverse complement*/
		if (ret > 0) {
			uint32_t numNs = 0;
			srand48(11);
			for (uint32_t i = 0; i < seq._length; i++) {
				if (seq._bases[i] == UNKNOWN_BASE) {
					++numNs;
				}
			}
			seq._tlength = seq._length - numNs;

			//compute the reverse complement of the sequence
			reverseComp(seq._rbases, seq._bases, seq._length);
		}
		return ret;
	}
	inline size_t getSeq(Sequence& seq1, Sequence& seq2) {
		if (!getSeq(seq1)) {
			return 0;
		}
		if (!getSeq(seq2)) {
			Utils::exit(
					"Paired-end reads are inconsistent at line %d in file %s\n",
					__LINE__, __FILE__);
		}
		return seq1._length;
	}
	inline size_t getSeq(Sequence* seqs, int maxSeqs) {
		int index;
		for (index = 0; index < maxSeqs; ++index) {
			if (!getSeq(seqs[index])) {
				break;
			}
		}

		return index;
	}

	inline size_t getSeq(Sequence* seqs1, Sequence* seqs2, int maxSeqs) {
		int index;
		for (index = 0; index < maxSeqs; ++index) {
			if (!getSeq(seqs1[index], seqs2[index])) {
				break;
			}
		}

		return index;
	}

	inline size_t getSeqLockFree(Sequence& seq) {
		size_t ret;

		/*read the sequence from the file*/
		if (_format == FILE_FORMAT_FASTA) {
			ret = getFastaSeq(seq);
		} else if (_format == FILE_FORMAT_FASTQ) {
			ret = getFastqSeq(seq);
		} else if (_format == FILE_FORMAT_BSAM) { /*BAM/SAM format*/
			ret = getBSamSeq(seq);
		} else {
			ret = getFifoSeq(seq);
		}

    /*check the sequence length*/
    if(seq._length > MAX_SEQ_LENGTH){
      Utils::log("trimm the reads (of %d bases) longer than %d\n", seq._length, MAX_SEQ_LENGTH);
      seq._length = MAX_SEQ_LENGTH;
    }

		/*compute the reverse complement*/
		if (ret > 0) {
			uint32_t numNs = 0;
			srand48(11);
			for (uint32_t i = 0; i < seq._length; i++) {
				if (seq._bases[i] == UNKNOWN_BASE) {
					++numNs;
				}
			}
			seq._tlength = seq._length - numNs;

			//compute the reverse complement of the sequence
			reverseComp(seq._rbases, seq._bases, seq._length);
		}

		return ret;
	}
	inline size_t getSeqLockFree(Sequence& seq1, Sequence& seq2) {
		if (!getSeqLockFree(seq1)) {
			return 0;
		}
		if (!getSeqLockFree(seq2)) {
			Utils::exit(
					"Paired-end reads are inconsistent at line %d in file %s\n",
					__LINE__, __FILE__);
		}
		return seq1._length;
	}
	inline size_t getSeqLockFree(Sequence* seqs, int maxSeqs) {
		int index;
		for (index = 0; index < maxSeqs; ++index) {
			if (!getSeqLockFree(seqs[index])) {
				break;
			}
		}

		return index;
	}

	inline size_t getSeqLockFree(Sequence* seqs1, Sequence* seqs2,
			int maxSeqs) {
		int index;
		for (index = 0; index < maxSeqs; ++index) {
			if (!getSeqLockFree(seqs1[index], seqs2[index])) {
				break;
			}
		}

		return index;
	}
	static inline void encode(uint8_t* s, size_t length) {
		uint8_t ch;
		for (size_t i = 0; i < length; i++) {
			ch = s[i];
			if (ch >= 'A' && ch <= 'Z') {
				ch -= 'A';
			} else if (ch >= 'a' && ch <= 'z') {
				ch -= 'a';
			} else {
				Utils::exit("Unexpected character %c\n", ch);
			}
			s[i] = _codeTab[ch];
		}
	}
	//file reading test
	static void test(const char* path);
private:
	/*private member functions*/
	void resizeBuffer(size_t nsize);
	size_t getFastaSeq(Sequence& seq);
	size_t getFastqSeq(Sequence& seq);
	size_t getBSamSeq(Sequence & seq);
	size_t getFifoSeq(Sequence& seq);
	inline void reverseComp(uint8_t* rbases, uint8_t* bases, size_t length) {
		size_t off;
		size_t halfLength = length / 2;

		for (size_t i = 0; i < halfLength; i++) {
			off = length - i - 1;
			rbases[off] = _complements[bases[i]];
			rbases[i] = _complements[bases[off]];
		}
		if (length & 1) {
			rbases[halfLength] = _complements[bases[halfLength]];
		}
	}
	inline void reverseComp(uint8_t* bases, size_t length) {
		uint8_t ch;
		size_t off;
		size_t halfLength = length / 2;

		for (size_t i = 0; i < halfLength; i++) {
			off = length - i - 1;
			ch = bases[off];
			bases[off] = _complements[bases[i]];
			bases[i] = _complements[ch];
		}
		if (length & 1) {
			bases[halfLength] = _complements[bases[halfLength]];
		}
	}

	inline void _lock() {
		if (_withLock) {
			pthread_mutex_lock(&_mutex);
		}
	}
	inline void _unlock() {
		if (_withLock) {
			pthread_mutex_unlock(&_mutex);
		}
	}

	/*buffered file operations*/
	inline int myfgetc(MyFilePt file) {
		/*check the end-of-file*/
		if (_fileBufferSentinel >= _fileBufferLength) {
			/*re-fill the buffer*/
			_fileBufferSentinel = 0;
			/*read file*/
			_fileBufferLength = myfread(_fileBuffer, 1, 4096, file);
			if (_fileBufferLength == 0) {
				/*reach the end of the file*/
				if (myfeof(file)) {
					return -1;
				} else {
					Utils::exit("File reading failed in function %s line %d\n",
							__FUNCTION__, __LINE__);
				}
			}
		}
		/*return the current character, and increase the sentinel position*/
		return _fileBuffer[_fileBufferSentinel++];
	}
	inline int myungetc(int ch, MyFilePt file) {
		if (_fileBufferSentinel >= 0) {
			_fileBuffer[--_fileBufferSentinel] = ch;
		} else {
			Utils::log("Two consecutive ungetc operations occurred\n");
			return -1; /*an error occurred, return end-of-file marker*/
		}
		return ch;
	}
private:
	/*private member variables*/
	//buffer for file reading
	uint8_t* _buffer;
	size_t _length;
	size_t _size;

	//FASTA/FASTQ file handler
	MyFilePt _fp;
	uint8_t* _fileBufferR;
	uint8_t* _fileBuffer;
	int _fileBufferLength;
	int _fileBufferSentinel;

	/*FIFO file handler*/
	static int _fifoFile;
	bool _fifoEndOfFile;

	//BAM/SAM file handler
	samfile_t* _samfp;
	bam1_t* _samentry;
	//
	int _format;
	pthread_mutex_t _mutex;
	bool _withLock;

	static const uint8_t _codeTab[26];
	static const uint8_t _decodeTab[5];
	static const uint8_t _complements[5];
};

#endif /* SEQFILEPARSER_H_ */

#include "SeqFileParser.h"
#include "Utils.h"
#include <zlib.h>

/*FIFO file descriptor*/
int SeqFileParser::_fifoFile = -1;

const uint8_t SeqFileParser::_codeTab[26] = { 0, 4, 1, 4, 4, 4, 2, //A -> G
		4, 4, 4, 4, 4, 4, 4, //H->N
		4, 4, 4, 4, 4, 3, 4, //O->U
		4, 4, 4, 4, 4 //V->Z
		};
const uint8_t SeqFileParser::_decodeTab[5] = { 'A', 'C', 'G', 'T', 'N' };

const uint8_t SeqFileParser::_complements[5] = { 3, 2, 1, 0, 4 };

SeqFileParser::SeqFileParser(const char* path, bool withLock, int type,
		size_t BUFFER_SIZE) {
	/*create mutex*/
	_withLock = withLock;
	pthread_mutex_init(&_mutex, NULL);

	//allocate buffer for file reading
	_size = BUFFER_SIZE;
	_length = 0;
	_buffer = new uint8_t[_size + 1];
	if (_buffer == NULL) {
		Utils::exit("Memory allocation failed in file %s in line %d\n",
				__FUNCTION__, __LINE__);
	}

	/*chech the file type*/
	if (type == FILE_TYPE_BAM || type == FILE_TYPE_SAM) {
		char modes[5];
		modes[0] = 'r'; /*means reading*/
		if (type == FILE_TYPE_BAM) {
			modes[1] = 'b';
			modes[2] = '\0';
		} else {
			modes[1] = '\0';
		}
		/*open the file*/
		_samfp = samopen(path, modes, NULL);
		if (!_samfp) {
			Utils::cmpExit("Failed to open file: %s\n", path);
		}
		/*check the file header*/
		if (_samfp->header == 0) {
			Utils::cmpExit("Failed to read the header from \"%s\"\n", path);
		}

		/*set the file format*/
		_format = FILE_FORMAT_BSAM;

		/*create entry*/
		_samentry = bam_init1();

		/*return*/
		return;
	} else if (type == FILE_TYPE_FIFO) {
		if (_fifoFile == -1) {
			/*open the FIFO*/
			_fifoFile = open(path, O_RDONLY);
			if (_fifoFile == -1) {
				Utils::exit(
						"Failed to open FIFO file %s at line %d in file %s\n",
						path, __LINE__, __FILE__);
			}
		}
		_format = FILE_FORMAT_FIFO;
		_fifoEndOfFile = false;

		return;
	}
	/*create the file buffer*/
	_fileBufferSentinel = 0;
	_fileBufferLength = 0;
	_fileBufferR = new uint8_t[4096 + 8];
	if (_fileBufferR == NULL) {
		Utils::exit("Memory allocation failed in file %s in line %d\n",
				__FUNCTION__, __LINE__);
	}
	_fileBuffer = _fileBufferR + 8; /*make it aligned*/

	/*open the input file*/
	if (strcmp(path, "-") == 0) {
		_fp = myopenstdin("rb");
	} else {
		_fp = myfopen(path, "rb");
	}
	if (_fp == NULL) {
		Utils::cmpExit("Failed to open file: %s\n", path);
	}

	//detecting the file format in the first line
	int ch;
	while ((ch = myfgetc(_fp)) != -1 && ch != '>' && ch != '@' && ch != '\n')
		;
	if (ch == -1 || ch == '\n') {
		Utils::exit("Unrecognized file format\n");
	} else if (ch == '>') {
		_format = FILE_FORMAT_FASTA;
		myungetc(ch, _fp);
		Utils::log("FASTA format identified\n");
	} else {
		_format = FILE_FORMAT_FASTQ;
		myungetc(ch, _fp);
		Utils::log("FASTQ format identified\n");
	}
}
SeqFileParser::~SeqFileParser() {
	if (_buffer) {
		delete[] _buffer;
	}
	/*check the file format*/
	if (_format == FILE_FORMAT_FASTA || _format == FILE_FORMAT_FASTQ) {
		//close the file
		myfclose(_fp);

		//release the buffer
		delete[] _fileBufferR;
	} else if (_format == FILE_FORMAT_BSAM) {
		samclose(_samfp);
		bam_destroy1(_samentry);
	}
	/*destroy mutex*/
	pthread_mutex_destroy(&_mutex);
}

void SeqFileParser::resizeBuffer(size_t nsize) {
	if (nsize <= _size) {
		return;
	}

	//allocate a new buffer
	_size = nsize * 2;
	uint8_t* nbuffer = new uint8_t[_size];
	if (!nbuffer) {
		Utils::exit("Memory reallocation failed in file %s in line %d\n",
				__FUNCTION__, __LINE__);
	}
	//copy the old data
	memcpy(nbuffer, _buffer, _length);

	//release the old buffer
	delete[] _buffer;
	_buffer = nbuffer;
}
size_t SeqFileParser::getFastaSeq(Sequence& seq) {
	int ch;

	//find the header
	while ((ch = myfgetc(_fp)) != -1 && ch != '>')
		;
	if (ch == -1) {
		return 0; //reach the end of file
	}
	//read the sequence name (only one line)
	_length = 0;
	while ((ch = myfgetc(_fp)) != -1 && ch != '\n') {
		if (_length >= _size) {
			resizeBuffer(_size + 256);
		}
		if (isspace(ch)) {
			ch = '\0';
		}
		_buffer[_length++] = ch;
	}
	if (ch == -1) {
		Utils::exit("Incomplete file\n");
	}
	_buffer[_length] = '\0';

	/*trim characters /[12]$ like BWA*/
	if (_length > 2 && _buffer[_length - 2] == '/'
			&& (_buffer[_length - 1] == '1' || _buffer[_length - 1] == '2')) {
		_length -= 2;
		_buffer[_length] = '\0';
	}

	//save the sequence name
	seq.setNameSize(_length + 1); /*adjust the name buffer size*/
	strcpy((char*) seq._name, (char*) _buffer);

	//read the sequence bases
	_length = 0;
	do {
		//filter out the blank lines
		while ((ch = myfgetc(_fp)) != -1 && (ch == '\r' || ch == '\n'))
			;
		if (ch == -1)
			break; //reach the end of file
		if (ch == '>') { //reaching another sequence
			myungetc(ch, _fp);
			break;
		}

		//encode and save the base
		if (ch >= 'A' && ch <= 'Z') {
			ch -= 'A';
		} else if (ch >= 'a' && ch <= 'z') {
			ch -= 'a';
		} else {
			Utils::exit("Unexpected character %c\n", ch);
		}
		//save the current encoded base
		if (_length >= _size) {
			resizeBuffer(_size + 256);
		}
		_buffer[_length++] = _codeTab[ch];
	} while (1);

	//save the sequence length and its bases
	seq._length = _length;
	if (_length > 0) {
		/*adjust the sequence buffer size*/
		seq.setSequenceSize(seq._length, false);
		memcpy(seq._bases, _buffer, _length);
	}

	return _length;
}

size_t SeqFileParser::getFastqSeq(Sequence& seq) {
	int ch;

	//find the header
	while ((ch = myfgetc(_fp)) != -1 && ch != '@')
		;
	if (ch == -1)
		return 0; //reach the end of file

	//read the sequence name (only one line)
	_length = 0;
	while ((ch = myfgetc(_fp)) != -1 && ch != '\n') {
		if (_length >= _size) {
			resizeBuffer(_size + 256);
		}
		if (isspace(ch)) {
			ch = '\0';
		}
		_buffer[_length++] = ch;
	}
	if (ch == -1) {
		Utils::exit("Incomplete file\n");
	}
	_buffer[_length] = '\0';

	/*trim characters /[12]$ like BWA*/
	if (_length > 2 && _buffer[_length - 2] == '/'
			&& (_buffer[_length - 1] == '1' || _buffer[_length - 1] == '2')) {
		_length -= 2;
		_buffer[_length] = '\0';
	}

	//save the sequence name
	seq.setNameSize(_length + 1); /*adjust the name buffer size*/
	strcpy((char*) seq._name, (char*) _buffer);

	//read the sequence bases
	_length = 0;
	do {
		//filter out the blank lines
		while ((ch = myfgetc(_fp)) != -1 && (ch == '\r' || ch == '\n'))
			;
		if (ch == -1)
			Utils::exit("Incomplete FASTQ file\n");
		if (ch == '+')
			break; //the comment line

		//encode and save the base
		if (ch >= 'A' && ch <= 'Z') {
			ch -= 'A';
		} else if (ch >= 'a' && ch <= 'z') {
			ch -= 'a';
		} else {
			Utils::exit("Unexpected character %c\n", ch);
		}
		//save the current encoded base
		if (_length >= _size) {
			resizeBuffer(_size + 256);
		}
		_buffer[_length++] = _codeTab[ch];

	} while (1);

	//save the sequence length and its bases
	seq._length = _length;
	assert(_length > 0);

	/*adjust the sequence buffer size*/
	seq.setSequenceSize(seq._length, true);
	memcpy(seq._bases, _buffer, _length);

	//read the comment line (only one line)
	while ((ch = myfgetc(_fp)) != -1 && ch != '\n')
		;
	if (ch == -1)
		Utils::exit("Incomplete FASTQ file\n");

	//read the quality scores
	_length = 0;
	while ((ch = myfgetc(_fp)) != -1 && ch != '\n') {
		if (_length >= _size) {
			resizeBuffer(_size + 256);
		}
		if (ch >= 33 && ch <= 127) {
			_buffer[_length++] = ch;
		}

		if (_length > seq._length)
			break;
	}
	if (seq._length != _length) {
		Utils::exit(
				"The number of bases is not equal to the number of quality scores\n");
	}
	/*copy the quality scores*/
	memcpy(seq._quals, _buffer, _length);

	return seq._length;
}
size_t SeqFileParser::getBSamSeq(Sequence& seq) {

	/*read an entry from the file*/
	int ret = samread(_samfp, _samentry);
	if (ret < 0) {
		if (ret < -1)
			Utils::exit("Truncated file\n");
		return 0;
	}

	/*convert the entry to a string*/
	char* entry = samget(_samfp->header, _samentry);

	/*extract the read from the entry*/
	int i;
	char delimit[] = "\t\n";
	char* tok = strtok(entry, delimit);

	/*get the query name*/
	seq.setNameSize(strlen(tok) + 1); /*adjust the name buffer size*/
	strcpy((char*) seq._name, tok);

	/*get the flag*/
	tok = strtok(NULL, delimit);
	int flag = atoi(tok);

	/*get the aligned strand of the read*/
	int rc = (flag & SAM_FSR) != 0;

	/*skip the reference sequence name*/
	tok = strtok(NULL, delimit);

	/*skip the mapping position*/
	tok = strtok(NULL, delimit);

	/*skip the mapping quality*/
	tok = strtok(NULL, delimit);

	/*skip the CIGAR region*/
	tok = strtok(NULL, delimit);

	/*skip the paired-end alignment information*/
	for (i = 0; i < 3; ++i) {
		tok = strtok(NULL, delimit);
	}

	/*get the read bases as per the strand*/
	tok = strtok(NULL, delimit);

	/*get the sequence length*/
	seq._length = strlen(tok);

	/*get a copy of the sequence*/
	char* tmpseq = strdup(tok);

	/*get the quality score if applicable*/
	bool hasquals = true;
	tok = strtok(NULL, delimit);
	if (!strcmp(tok, "*")) {
		hasquals = false;
		/*adjust the sequence buffer size*/
		seq.setSequenceSize(seq._length, false);
		memcpy(seq._bases, tmpseq, seq._length);
	} else {
		/*adjust the sequence buffer size*/
		seq.setSequenceSize(seq._length, true);
		memcpy(seq._bases, tmpseq, seq._length);
	}
	/*release the temp sequence*/
	free(tmpseq);

	/*encode the sequence*/
	encode(seq._bases, seq._length);
	if (rc)
		reverseComp((uint8_t*) seq._bases, seq._length);

	/*copy the quality scores*/
	if (hasquals) {
		memcpy(seq._quals, tok, seq._length);
		if (rc) {
			/*reverse quality scores*/
			for (size_t i = 0; i < seq._length / 2; ++i) {
				char ch = seq._quals[i];
				seq._quals[i] = seq._quals[seq._length - 1 - i];
				seq._quals[seq._length - 1 - i] = ch;
			}
		}
	}

	/*release entry*/
	free(entry);

	return seq._length;
}
size_t SeqFileParser::getFifoSeq(Sequence& seq) {

	/*if it reaches the end of file*/
	if (_fifoEndOfFile) {
		return 0;
	}
	/*read the data size*/
	if (read(_fifoFile, _buffer, 4) != 4) {
		Utils::exit("FIFO read failed at line %d in file %s\n", __LINE__,
				__FILE__);
	}
	_length = _buffer[0];
	_length = (_length << 8) + _buffer[1];
	_length = (_length << 8) + _buffer[2];
	_length = (_length << 8) + _buffer[3];

	/*check the data size*/
	if (_length == 0) {
		/*indicates the end of the input sequences*/
		_fifoEndOfFile = true;
		return 0;
	}

	/*resize the buffer*/
	if (_length > _size) {
		resizeBuffer(_length + 256);
	}

	/*read the sequence data from the FIFO*/
	if (read(_fifoFile, _buffer, _length) != _length) {
		Utils::exit("FIFO read failed at line %d in file %s\n", __LINE__,
				__FILE__);
	}

	/*decompose the data*/
	seq.decompose(_buffer, _length);

	return seq._length;
}
void SeqFileParser::test(const char* path) {
	SeqFileParser* parser = new SeqFileParser(path, false, FILE_TYPE_FASTX);

	Sequence seq;
	while (parser->getSeq(seq)) {
		//print out the sequence name
		fputs((const char*) seq._name, stderr);

		//print out the sequence bases
		for (size_t i = 0; i < seq._length; i++) {
			fputc("ACGTN"[seq._bases[i]], stderr);
		}
		fputc('\n', stderr);

		//printout the quality scores
		if (seq._quals) {
			for (size_t i = 0; i < seq._length; i++) {
				fputc(seq._quals[i], stderr);
			}
			fputc('\n', stderr);
		}
	}
	delete parser;
}

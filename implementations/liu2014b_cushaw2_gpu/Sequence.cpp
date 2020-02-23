/*
 * Sequence.cpp
 *
 *  Created on: Dec 23, 2011
 *      Author: yongchao
 */

#include "Sequence.h"
#include "Utils.h"

Sequence::Sequence() {
	_name = NULL;
	_bases = NULL;
	_rbases = NULL;
	_quals = NULL;
	_nameSize = _seqSize = 0;
	_length = 0;
	_tlength = 0;
}
Sequence::Sequence(const Sequence & s) {
	_length = s._length;
	_tlength = s._tlength;
	_nameSize = s._nameSize;
	_seqSize = s._seqSize;
	if (_length == 0) {
		_name = NULL;
		_bases = NULL;
		_rbases = NULL;
		_quals = NULL;
		_nameSize = 0;
		_seqSize = 0;
		return;
	}
	if (s._name) {
		_name = new uint8_t[_nameSize];
		if (_name == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		strcpy((char*) _name, (const char*) s._name);
	}
	if (s._bases) {
		_bases = new uint8_t[_seqSize];
		if (_bases == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		memcpy(_bases, s._bases, _length);
	}
	if (s._rbases) {
		_rbases = new uint8_t[_seqSize];
		if (_rbases == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		memcpy(_rbases, s._rbases, _length);
	}
	if (s._quals) {
		_quals = new uint8_t[_seqSize];
		if (_quals == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		memcpy(_quals, s._quals, _length);
	}
}
Sequence::~Sequence() {
	clear();
}
void Sequence::clear() {
	if (_name) {
		delete[] _name;
	}
	if (_bases) {
		delete[] _bases;
	}
	if (_rbases) {
		delete[] _rbases;
	}
	if (_quals) {
		delete[] _quals;
	}

	_name = NULL;
	_bases = NULL;
	_rbases = NULL;
	_quals = NULL;
	_length = 0;
	_tlength = 0;
	_nameSize = 0;
	_seqSize = 0;
}
void Sequence::setNameSize(size_t size) {
	if (size >= _nameSize) {
		_nameSize = size * 2;
		if (_name) {
			delete[] _name;
		}
		_name = new uint8_t[_nameSize];
		if (_name == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
	}
}
void Sequence::setSequenceSize(size_t size, bool quals) {
	if (size >= _seqSize) {
		_seqSize = size * 2;
		/*forward strand*/
		if (_bases) {
			delete[] _bases;
		}
		_bases = new uint8_t[_seqSize];
		if (_bases == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}
		/*reverse strand*/
		if (_rbases) {
			delete[] _rbases;
		}
		_rbases = new uint8_t[_seqSize];
		if (_rbases == NULL) {
			Utils::exit("Memory allocation failed in function %s line %d\n",
					__FUNCTION__, __LINE__);
		}

		/*allocate space for quality scores*/
		if (quals) {
			if (_quals) {
				delete[] _quals;
			}
			_quals = new uint8_t[_seqSize];
			if (_quals == NULL) {
				Utils::exit("Memory allocation failed in function %s line %d\n",
						__FUNCTION__, __LINE__);
			}
		}
	}
}
void Sequence::print(FILE* file) {
	//print the sequence name
	if (_quals) {
		fputc('@', file);
	} else {
		fputc('>', file);
	}
	fprintf(file, "%s\n", _name);

	//print the query sequence
	for (uint32_t i = 0; i < _length; ++i) {
		fputc(decode(_bases[i]), file);
	}
	fputc('\n', file);

	//print the quality scores if available
	if (_quals) {
		/*printout comments*/
		fputc('+', file);
		fputc('\n', file);
		for (uint32_t i = 0; i < _length; ++i) {
			fputc(_quals[i], file);
		}
	}
	fputc('\n', file);
}

void Sequence::decompose(uint8_t* buffer, size_t bufferLength) {
	uint32_t length;
	uint32_t offset = 0;
	bool hasQuals;

	/*get the name length*/
	length = buffer[offset++];
	length = (length << 8) + buffer[offset++];
	length = (length << 8) + buffer[offset++];
	length = (length << 8) + buffer[offset++];

	/*copy the name*/
	setNameSize(length + 1);
	memcpy(_name, buffer + offset, length);
	_name[length] = '\0';
	offset += length;

	/*get the quality score length*/
	length = buffer[offset++];
	length = (length << 8) + buffer[offset++];
	length = (length << 8) + buffer[offset++];
	length = (length << 8) + buffer[offset++];

	hasQuals = false;
	if (length > 0) {
		hasQuals = true;
		setSequenceSize(length, true);
		memcpy(_quals, buffer + offset, length);
		offset += length;
	}

	/*get the sequence length*/
	_length = buffer[offset++];
	_length = (_length << 8) + buffer[offset++];
	_length = (_length << 8) + buffer[offset++];
	_length = (_length << 8) + buffer[offset++];

	if (!hasQuals) {
		setSequenceSize(_length, false);
	}
	memcpy(_bases, buffer + offset, _length);
	offset += _length;

	memcpy(_rbases, buffer + offset, _length);
	offset += _length;

	/*get true length*/
	_tlength = buffer[offset++];
	_tlength = (_tlength << 8) + buffer[offset++];
	_tlength = (_tlength << 8) + buffer[offset++];
	_tlength = (_tlength << 8) + buffer[offset++];
}
int32_t Sequence::compose(uint8_t*& buffer, size_t& bufferSize) {
	uint32_t offset = 0;
	uint32_t nameLength = strlen((const char*) _name);
	uint32_t maxNumBytes = 3 * _length + nameLength + 5 * sizeof(uint32_t);

	/*resize the buffer*/
	if (bufferSize < maxNumBytes) {
		bufferSize = maxNumBytes + 256;
		uint8_t* newBuffer = new uint8_t[bufferSize];
		if (!newBuffer) {
			Utils::exit("Memory allocation failed at line %d in file %s\n",
					__LINE__, __FILE__);
		}
		if (buffer) {
			delete[] buffer;
		}
		buffer = newBuffer;
	}

	/*copy the data into the buffer*/
	offset = 4; /*start from index 4*/
	buffer[offset++] = (nameLength >> 24) & 0x0ff;
	buffer[offset++] = (nameLength >> 16) & 0x0ff;
	buffer[offset++] = (nameLength >> 8) & 0x0ff;
	buffer[offset++] = nameLength & 0x0ff;
	memcpy(buffer + offset, _name, nameLength);
	offset += nameLength;

	/*copy quality scores*/
	if (_quals) {

		buffer[offset++] = (_length >> 24) & 0x0ff;
		buffer[offset++] = (_length >> 16) & 0x0ff;
		buffer[offset++] = (_length >> 8) & 0x0ff;
		buffer[offset++] = _length & 0x0ff;
		memcpy(buffer + offset, _quals, _length);
		offset += _length;
	} else {
		buffer[offset++] = 0;
		buffer[offset++] = 0;
		buffer[offset++] = 0;
		buffer[offset++] = 0;
	}

	buffer[offset++] = (_length >> 24) & 0x0ff;
	buffer[offset++] = (_length >> 16) & 0x0ff;
	buffer[offset++] = (_length >> 8) & 0x0ff;
	buffer[offset++] = _length & 0x0ff;
	memcpy(buffer + offset, _bases, _length);
	offset += _length;

	memcpy(buffer + offset, _rbases, _length);
	offset += _length;

	buffer[offset++] = (_tlength >> 24) & 0x0ff;
	buffer[offset++] = (_tlength >> 16) & 0x0ff;
	buffer[offset++] = (_tlength >> 8) & 0x0ff;
	buffer[offset++] = _tlength & 0x0ff;

	/*write the number of bytes*/
	uint32_t numBytes = offset - 4;
	buffer[0] = (numBytes >> 24) & 0x0ff;
	buffer[1] = (numBytes >> 16) & 0x0ff;
	buffer[2] = (numBytes >> 8) & 0x0ff;
	buffer[3] = numBytes & 0x0ff;

	return offset;
}

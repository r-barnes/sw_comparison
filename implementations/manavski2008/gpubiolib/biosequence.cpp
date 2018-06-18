#include "biosequence.h"


#include <iostream>
#include <algorithm>
#include <cctype> 


#include <QtCore/QTime>

BioSequence::BioSequence()
{
}

BioSequence::BioSequence(const string &n)
{
	name = n;
	sequence = "";
}


BioSequence::BioSequence(const string &n, const string & q)
{
	name = n;
	sequence = q;
	std::transform(sequence.begin(), sequence.end(), sequence.begin(), (int(*)(int)) std::toupper);
}


BioSequence::~BioSequence()
{
	clear();
}

string BioSequence::fastaToName(const string &line)
{
	unsigned N = line.size();
	unsigned j=0;
	for (; j<N; ++j) {
		if ( !(line[j] != ' ' && line[j] != '\n') ) break;
	}
	return line.substr(0, j);
}


void BioSequence::clear()
{
}

unsigned BioSequence::getSize() const
{
	return sequence.size();
}

const string & BioSequence::getName() const
{
	return name;
}

void BioSequence::setSequence(const string &s)
{
	sequence = s;
}

const string & BioSequence::getSequence() const
{
	return sequence;
}

const char * BioSequence::sequenceData() const
{
	return sequence.data();
}

string BioSequence::getSegment(unsigned pos, unsigned len) const
{
	return sequence.substr(pos, len);
}

void BioSequence::pushFront(const string &s) {
	sequence = s + sequence;
}

void BioSequence::toLower() {
	std::transform(sequence.begin(), sequence.end(), sequence.begin(), (int(*)(int)) std::tolower);
}

void BioSequence::toUpper() {
	std::transform(sequence.begin(), sequence.end(), sequence.begin(), (int(*)(int)) std::toupper);
}

void BioSequence::reverse()
{
	string outseq = "";
	int insz = sequence.size();
	for (int j=insz-1; j>=0; --j) {
		switch (sequence[j]) {
			case 'A':
			case 'a':
				outseq += 'T';
				break;
			case 'T':
			case 't':
				outseq += 'A';
				break;
			case 'C':
			case 'c':
				outseq += 'G';
				break;
			case 'G':
			case 'g':
				outseq += 'C';
				break;
			default:
				outseq += 'N';
		}
	}
	//cout << outseq << endl;
	
	sequence = outseq;
}







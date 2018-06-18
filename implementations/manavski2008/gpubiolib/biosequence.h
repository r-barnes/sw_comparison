
#ifndef BIOSEQUENCE_H
#define BIOSEQUENCE_H

/**
	@author Svetlin Manavski <svetlin.a@manavski.com>
*/

#include <string>
using namespace std;

class BioSequence {
	public:
		BioSequence();
		BioSequence(const string &n);
		BioSequence(const string &n, const string &q);
		~BioSequence();

		static string fastaToName(const string &);
		
		void clear();
		unsigned getSize() const;
		const string& getName() const;

		void setSequence(const string &);
		const string &getSequence() const;
		const char *sequenceData() const;
		string getSegment(unsigned pos, unsigned len) const;
	
		void pushFront(const string &s);
		void toLower();
		void toUpper();
		void reverse();
	protected:
		string name;
		string sequence;
};


#endif

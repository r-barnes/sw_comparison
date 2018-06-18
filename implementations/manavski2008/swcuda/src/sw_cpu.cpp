
/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/


#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <QtCore/QFile>

#include "sw_cpu.h"
#include "blosum.h"

#define maxx(a, b) ( (a) > (b) ) ? (a) : (b)

using namespace std;

int sbt(char A, char B) {
   return cpu_blosum50[A-60][B-60];
}

const unsigned MAXSEQSIZE = 10240;

int Hmat[MAXSEQSIZE][MAXSEQSIZE];
int Emat[MAXSEQSIZE][MAXSEQSIZE];
int Fmat[MAXSEQSIZE][MAXSEQSIZE];

void swRect( const string &inseqA, const string &inseqB, int alpha, int beta, int* result) {
	int score = 0;
	const char *seqA = inseqA.c_str();
	const char *seqB = inseqB.c_str();

	unsigned sizeA = inseqA.size();
	unsigned sizeB = inseqB.size();

	for(unsigned j=0; j<sizeA; j++) Hmat[0][j] = Emat[0][j] = Fmat[0][j] = 0;

	for(unsigned i=1; i<sizeB; ++i) {
		unsigned j=1; 
		Hmat[i][0]=Emat[i][0]=Fmat[i][0]=0;
		for(; j<sizeA; ++j) {
			// calcolo di f
			int f1 = Hmat[(i-1)][j] - alpha;
			int f2 = Fmat[(i-1)][j] - beta;
			int f = maxx(f1, f2); //f = maxx( f, 0 );
			Fmat[i][j] = f;

			//calcolo di e
			int e1 = Hmat[i][(j-1)] - alpha;
			int e2 = Emat[i][(j-1)] - beta;
			int e = maxx(e1, e2); //e = maxx( e, 0 );
			Emat[i][j] = e;

			//calcolo di h
			int h1 = Hmat[(i-1)][(j-1)] + sbt(seqA[j], seqB[i]);
			Hmat[i][j] = maxx(0, h1); Hmat[i][j] = maxx(Hmat[i][j], e); 
			Hmat[i][j] = maxx(Hmat[i][j], f);

// 			if (Hmat[i][j]>score) 
// 				printf("i=%u j=%u Hmat[i][j]=%d\n", i,j,Hmat[i][j]);
			if (sizeA > MAXSEQSIZE)
				score = 0;
			else
				score = maxx(score, Hmat[i][j]);
			

			//cout << Hmat[i][j] << ", ";
		}
		// cout << endl;
	}


// 	unsigned diag =65; //da 2 fino a 65
// 	cout << endl << "diag=" << diag << "> " << endl;
// 	
// 	cout << endl << "Hdiag=" << diag << endl;
// 	cout << "0, ";
// 	for (unsigned j=1; j<diag; ++j) {
// 		cout << Hmat[j][diag-j] << ", ";
// 	}
// 	cout << endl;

// 	cout << endl << "PAM x diag " << diag << endl;
// 	for (unsigned j=diag-1; j>0; --j) {
// 		cout << seqA[j] << " " <<seqB[diag-j] << " " <<sbt(seqA[j], seqB[diag-j]) << endl;
// 	}
// 
// 	cout << endl << "Hdiag=" << diag-1 << endl;
// 	cout << "0, ";
// 	for (unsigned j=1; j<diag-1; ++j) {
// 		cout << Hmat[j][diag-1-j] << ", ";
// 	}
// 
// 	cout << endl << "Hdiag=" << diag-2 << endl;
// 	cout << "0, ";
// 	for (unsigned j=1; j<diag-2; ++j) {
// 		cout << Hmat[j][diag-2-j] << ", ";
// 	}
// 	cout << endl << "Ediag=" << diag-1 << endl;
// 	cout << "0, ";
// 	for (unsigned j=1; j<diag-1; ++j) {
// 		cout << Emat[j][diag-1-j] << ", ";
// 	}
// 	cout << endl << "Fdiag=" << diag-1 << endl;
// 	cout << "0, ";
// 	for (unsigned j=1; j<diag-1; ++j) {
// 		cout << Fmat[j][diag-1-j] << ", ";
// 	}


// 	unsigned diag = 65;		//deve rimanere 65
// 	unsigned offset =10240;/*sizeA - 1 - 64;*/		//da 1 a 64
// 	cout << endl << "diag=" << diag+offset << "> " ;
// 	for (unsigned j=1; j<diag; ++j) {
// 		cout << Hmat[j][diag+offset-j] << ", ";
// 	}

/*
   unsigned diag = 65;
   cout << endl << "E diag=" << diag << "> " ;
   for (unsigned j=1; j<diag; ++j) {
       cout << Emat[j][diag-j] << ", ";
   }
   cout << endl << "F diag=" << diag << "> " ;
   for (unsigned j=1; j<diag; ++j) {
       cout << Fmat[j][diag-j] << ", ";
   }
*/

// 	unsigned diag = 64; // da 2 fino a 64 (sembra che diag corrisponde all'indice in cuda dove fermarsi)
// 	cout << endl << "diag=" << (diag) << "> " ;
// 	for (unsigned j=diag; j<65; ++j) {
// 		cout << Hmat[j][sizeA -1 + diag - j] << ", ";
// 	}

	result[0] = score;
	result[1] = Hmat[64][sizeA -1];
}

void filterdb(const string &dbName, unsigned threshold) {
	
	unsigned MAXBUF = 1024;
	char buf[MAXBUF];

	vector<string> seqLib;
	vector<string> nameSeq;
	vector<string> defSeqLib;
	vector<string> defNameSeq;

	QFile qf2(dbName.c_str());
	qf2.open(QIODevice::ReadOnly);
	qf2.readLine(buf, MAXBUF);
	nameSeq.push_back(buf);
	qf2.close();

	QFile qf(dbName.c_str());
	qf.open(QIODevice::ReadOnly);

	string curseq = "";

	for(;;) {
		long nr = qf.readLine(buf, MAXBUF);
		if (nr>0 && buf[0]!='>') {
			buf[nr-1] = '\0';
			curseq += buf;
		} else if (nr>0) {
			if ( (curseq.size() > 0) ) {
				seqLib.push_back(curseq);
				nameSeq.push_back(buf);
				curseq = "";
			}
		} else {
			if ( (curseq.size() > 0) ) {
				seqLib.push_back(curseq);
			}
			break;
		}
 	}
	qf.close();

	for(unsigned cnt=0; cnt<seqLib.size(); ++cnt) {
		if (seqLib[cnt].size() < threshold + 1) {
			defSeqLib.push_back(seqLib[cnt]);
			nameSeq[cnt].erase(nameSeq[cnt].size()-1);
			defNameSeq.push_back(nameSeq[cnt]);
		}
	}

	QFile qf3("newtest64.fasta");
	qf3.open(QIODevice::WriteOnly);
	
	string newLine("\n");

	for(unsigned cnt=0; cnt<defNameSeq.size(); ++cnt) {
		qf3.write(defNameSeq[cnt].c_str(), qstrlen(defNameSeq[cnt].c_str()));
		qf3.write(newLine.c_str(), qstrlen(newLine.c_str()));
		qf3.write(defSeqLib[cnt].c_str(), qstrlen(defSeqLib[cnt].c_str()));
		qf3.write(newLine.c_str(), qstrlen(newLine.c_str()));
	}

	cout << " read " << seqLib.size() << " sequences..." << endl;
	cout << " read " << defSeqLib.size() << " sequences shorter than " << threshold << " bytes." <<endl;

	cout << defNameSeq[1].c_str() << endl;
	cout << defSeqLib[1].c_str() << endl;
	
}





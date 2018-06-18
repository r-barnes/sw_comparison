
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


#include "inout.h"

#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>
using namespace std;

// includes e funzione per lettura del MAC
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <net/if.h>

void verifyMAC()
{
	///codice da forum
	int fd;
	struct ifreq ifr;

	fd = socket(AF_INET, SOCK_DGRAM, 0);

	ifr.ifr_addr.sa_family = AF_INET;
	strncpy(ifr.ifr_name, "eth1", IFNAMSIZ-1);

	ioctl(fd, SIOCGIFHWADDR, &ifr);

	close(fd);
	char MAC[18];

	sprintf(MAC, "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x\n", (unsigned char)ifr.ifr_hwaddr.sa_data[0], (unsigned char)ifr.ifr_hwaddr.sa_data[1], (unsigned char)ifr.ifr_hwaddr.sa_data[2], (unsigned char)ifr.ifr_hwaddr.sa_data[3], (unsigned char)ifr.ifr_hwaddr.sa_data[4], (unsigned char)ifr.ifr_hwaddr.sa_data[5]);
	MAC[17] = '\0';
	///fine codice da forum
	
	string sMAC(MAC);
	//00:02:44:bd:d1:45	mercurio
	//eth1 00:1a:92:4f:d9:77 macchina prof

	string hardCodedMAC("00:1a:92:4f:d9:77");

	if (sMAC!=hardCodedMAC) {
		string s("cannot run this sofware on a unauthorized machine, eth1:");
		s += MAC;
		throw s;
	}
}

void usage(){
	cout << "\nswcuda Smith-Waterman GPU engine, v.1.92 for Cuda 1.1" << endl;
	cout << "Version 1.92" << endl << endl;
	cout << "Usage: smithwaterman [OPTION] FILE1 FILE2 OFFSET" << endl << endl;
	cout << "FILE1 must contain the sequences to align in FASTA format" << endl;
	cout << "FILE2 must contain the library in FASTA format" << endl;
	cout << "The alignment starts from the sequence number OFFSET in FILE1 (0 - default value)" << endl << endl;
	exit(0);
}


unsigned commandLineManager(int argc, char** argv, std::string &seqFile, std::string &libFile) {
	
	unsigned alignOffsets = 0;
	
	if ( argc < 3 ) {
			if (argc < 2) {
				cout << "Insert the file containing the sequence to align: ";
				for(;;) {
					getline(cin, seqFile);
					QFile qf(seqFile.c_str());
					if (!qf.exists())
						cout << "Sequence file \"" << seqFile.c_str() << "\" not found. Please insert a valid file name: ";
					else
						break;
				}
		
				cout << "Insert the file containing the library: ";
				for(;;) {
					getline(cin, libFile);
					QFile qf(libFile.c_str());
					if (!qf.exists())
						cout << "Library file \"" << libFile.c_str() << "\" not found. Please insert a valid file name: ";
					else
						break;
				}
				cout << endl;
			} else {
				if ( ( strcmp(argv[1],"-h") == 0 ) || ( strcmp(argv[1],"--h") == 0 ) || ( strcmp(argv[1],"--help") == 0 ) ){
					usage();
				} else {
					QFile qf(argv[1]);
					if (!qf.exists())
						throw string("Sequence file not found.");
					seqFile = argv[1];
					
					cout << "Insert the file containing the library: ";
					for(;;) {
						getline(cin, libFile);
						QFile qf(libFile.c_str());
						if (!qf.exists())
							cout << "Library file \"" << libFile.c_str() << "\" not found. Please insert a valid file name: ";
						else
							break;
					}
					cout << endl;
				}
			}
	} else if ( argc == 3 ) {
		QFile qf(argv[1]);
		if (!qf.exists())
			throw string("Sequence file not found.");
		seqFile = argv[1];

		QFile qf2(argv[2]);
		if (!qf2.exists())
			throw string("Sequence file not found.");
		libFile = argv[2];

	} else if ( argc == 4 ) {
		
		QFile qf(argv[1]);
		if (!qf.exists())
			throw string("Sequence file not found.");
		seqFile = argv[1];

		QFile qf2(argv[2]);
		if (!qf2.exists())
			throw string("Sequence file not found.");
		libFile = argv[2];
		
		alignOffsets = atoi(argv[3]);
		
	} else
		usage();

	return alignOffsets;
}

void fastaVerifier(std::string &fileName) {

	const unsigned MAX_BUF = 1024;
	char buf[MAX_BUF];

	QFile qf1(fileName.c_str());
	if (!qf1.open(QIODevice::ReadOnly | QIODevice::Text))
		throw std::string("cannot read from file " + fileName);

	if (qf1.size() < 1)
		throw std::string("cannot use a file empty: " + fileName);

	int nr = qf1.readLine(buf, MAX_BUF);

	if (buf[0] != '>')
		throw std::string("cannot use a non fasta format for the file " + fileName);

	bool good = TRUE;
	unsigned majorCnt = 0;

	nr = qf1.readLine(buf, MAX_BUF);
	if (buf[0] == '>')
		throw std::string("cannot use a non fasta format for the file " + fileName);

	while (nr>0) {

		if (buf[0]!='>') {

			//if (nr > 61 || !good)
			//	throw std::string("cannot use a non fasta format for the file " + fileName);
			
			for (int cnt=0; cnt<nr; ++cnt){
				char c = buf[cnt];
				if (c!='A' && c!='B' && c!='C' && c!='D' && c!='E' && c!='F' && c!='G' && c!='H' && c!='I' && c!='K' && c!='L' && c!='M' && c!='N' && c!='P' && c!='Q' && c!='R' && c!='S' && c!='T' && c!='V' && c!='W' && c!='X' && c!='Y' && c!='Z' && c!='\n' && c!='*')
					throw std::string("cannot use a non fasta format for the file " + fileName);
	
				majorCnt = 0;
			}

		} else {
			//std::cout << buf[10] << buf[11] <<buf[12]<<buf[13]<<buf[14]<<buf[15]<<buf[16]<<buf[17]<<buf[18]<<endl;
			majorCnt++;
		}

		if (majorCnt>1)
			throw std::string("cannot use a non fasta format for the file " + fileName);

		if (buf[0]!='>' && nr < 61)
			good = FALSE;

		nr = qf1.readLine(buf, MAX_BUF);

		if ( buf[0]=='>' || nr<0)
			good = TRUE;
	}
}


unsigned readdb(string searched, const string &searchedName, const string &fname, char *seqlib, unsigned seqLibSize, unsigned *offsets, unsigned *sizes, std::vector<std::string> &seqNames ) {

	const unsigned MAX_BUF = 1024;
	char buf[MAX_BUF];
	
	unsigned totSeqRead = 0;
	unsigned totBytesUsed = 0;

	unsigned rsz = searched.size() % 65;
	if (rsz != 0) {
		for( unsigned j=0; j < 65 - rsz; ++j) {
			searched += "@";
		}
	}
	strcpy(seqlib, searched.c_str());
	seqNames.push_back(searchedName);
	sizes[totSeqRead] = searched.size();
	totBytesUsed += searched.size() + 1;
	offsets[totSeqRead] = 0;
	++totSeqRead;
	cout << "searched: " << searchedName << ", size=" << searched.size() << " >" << searched << endl;

	QFile qf(fname.c_str());
	qf.open(QIODevice::ReadOnly);

	string curseq = "@", curname = "";
	unsigned curSeqSize = 1;
	for(;;) {

		long nr = qf.readLine(buf, MAX_BUF);
		if (nr>0 && buf[0]!='>') {
			buf[nr-1] = '\0';
			curseq += buf;
			curSeqSize += static_cast<unsigned>(strlen(buf));
		} else if (nr>0) {
			if (curseq.size() > 1) {
				// completiamo con padding [ al multiplo di 65
				unsigned rsz = curSeqSize % 65;
				if (rsz != 0) {
					for( unsigned j=0; j < 65 - rsz; ++j) {
						curseq += "@"; ++curSeqSize;
					}
				}
				if ( seqLibSize > (totBytesUsed + curSeqSize + 1) ) {
					// add nella seqlib
					strcpy(seqlib+totBytesUsed, curseq.c_str());
					seqNames.push_back(curname);
					sizes[totSeqRead] = curSeqSize;
					offsets[totSeqRead] = totBytesUsed;
					totBytesUsed += curSeqSize + 1;
					++totSeqRead;
					cout << "read: " << curname << ", size=" << curSeqSize << " >" << curseq << endl;
				} else
					break;

				// azzeramento del contatore della sequenza corrente
				curseq = "@"; curname = "";
				curSeqSize = 1;
			}
			// e qui leggiamo il nome della nuova sequenza che sta iniziando
			size_t st = strlen(buf);
			(st>=7) ? curname.append(buf+1, 6) : curname.append(buf+1, st-1);

		} else {
			if (curseq.size() > 1 ) {
				unsigned rsz = curSeqSize % 65;
				if (rsz != 0) {
					for( unsigned j=0; j < 65 - rsz; ++j) {
						curseq += "@"; ++curSeqSize;
					}
				}
				if (seqLibSize > (totBytesUsed + curSeqSize + 1) ) {
					// c'e' un'ultima sequenza letta da inserire nel db e ci sta
					strcpy(seqlib+totBytesUsed, curseq.c_str());
					seqNames.push_back(curname);
					sizes[totSeqRead] = curSeqSize;
					offsets[totSeqRead] = totBytesUsed;
					totBytesUsed += curSeqSize + 1;
					++totSeqRead;
					cout << "read: " << curname << ", size=" << curSeqSize << " >" << curseq << endl;
				}
			}
			break;
		}
		
	}
	cout << endl << " read " << totSeqRead << " sequences. " << "Total bytes used: " << totBytesUsed << endl;
	return totSeqRead;
}




unsigned readdbOrdered( const std::string &fname, char *seqlib, unsigned seqLibSize, unsigned *offsets, unsigned *sizes, std::vector<std::string> &seqNames, std::vector<std::string> &seqNamesNotOrd, unsigned *sizesWithPad ) {
	
	const unsigned MAX_BUF = 1024;
	char buf[MAX_BUF];
	vector<dbLengthIdx> tempIndex(MAX_DB_STRUCT_SIZE);

	unsigned totSeqRead = 0;
	unsigned totBytesUsed = 0;

	vector<string> tempSeqLib;
	vector<string> tempSeqNames;
	vector<unsigned> tempOffsets;

	QFile qf(fname.c_str());
	if (!qf.exists())
		throw string("sequence file " + fname + " not found");
	if (!qf.open(QIODevice::ReadOnly))
		return totSeqRead;

	string curseq = "@", curname = "";
	unsigned curSeqSize = 1;

	for(;;) {

		long nr = qf.readLine(buf, MAX_BUF);
		if (nr>0 && buf[0]!='>') {
			if (buf[nr-1] == '\n' || buf[nr-1] == '\r') buf[--nr] = '\0';
			curseq += buf;
			curSeqSize += static_cast<unsigned>(strlen(buf));
		} else if (nr>0) {
			if (curseq.size() > 1) {
				// completiamo con padding [ al multiplo di 64
				tempIndex[totSeqRead].length = curSeqSize;
				unsigned rsz = (curSeqSize-1) % 64;
				if (rsz != 0) {
					for( unsigned j=0; j < 64 - rsz; ++j) {
						curseq += "@"; ++curSeqSize;
					}
				} 
				
				if ( seqLibSize > (totBytesUsed + curSeqSize + 1) ) {
					// add nella seqlib
					tempSeqLib.push_back(curseq.c_str());
					tempSeqNames.push_back(curname);
					tempIndex[totSeqRead].lengthWithPad = curSeqSize;
					tempIndex[totSeqRead].idx = totSeqRead;
					tempOffsets.push_back(totBytesUsed);
					totBytesUsed += curSeqSize + 1;
					++totSeqRead;
					if (totSeqRead >= MAX_DB_STRUCT_SIZE)
						throw string("not enough structure elements for the DB");
					//cout << "read: " << curname << ", size=" << curSeqSize << " >" << curseq << endl;
				} else
					break;

				// azzeramento del contatore della sequenza corrente
				curseq = "@"; curname = "";
				curSeqSize = 1;
			}
			// e qui leggiamo il nome della nuova sequenza che sta iniziando
			size_t st = strlen(buf);
			//(st>=7) ? curname.append(buf+1, 6) : curname.append(buf+1, st-1);
			curname.append(buf+1, st-1);

		} else {
			if (curseq.size() > 1 ) {
				// completiamo con padding [ al multiplo di 64
				tempIndex[totSeqRead].length = curSeqSize;
				unsigned rsz = (curSeqSize-1) % 64;
				if (rsz != 0) {
					for( unsigned j=0; j < 64 - rsz; ++j) {
						curseq += "@"; ++curSeqSize;
					}
				}

				if (seqLibSize > (totBytesUsed + curSeqSize + 1) ) {
					// c'e' un'ultima sequenza letta da inserire nel db e ci sta
					tempSeqLib.push_back(curseq.c_str());
					tempSeqNames.push_back(curname);
					tempIndex[totSeqRead].lengthWithPad = curSeqSize;
					tempIndex[totSeqRead].idx = totSeqRead;
					tempOffsets.push_back(totBytesUsed);
					totBytesUsed += curSeqSize + 1;
					++totSeqRead;
					if (totSeqRead >= MAX_DB_STRUCT_SIZE)
						throw string("not enough structure elements for the DB");
					//cout << "read: " << curname << ", size=" << curSeqSize << " >" << curseq << endl;
				}
			}
			break;
		}
		
	}
	//cout << "Read " << totSeqRead << " sequences from " << fname.c_str() << ". Total bytes used: " << totBytesUsed << endl<< endl;
	
	seqNamesNotOrd.resize(totSeqRead);
	copy (tempSeqNames.begin(), tempSeqNames.end(), seqNamesNotOrd.begin());

	vector<dbLengthIdx>::iterator itend = tempIndex.begin() + totSeqRead;
	sort( tempIndex.begin(), itend, Eless());

	strcpy(seqlib, tempSeqLib[tempIndex[0].idx].c_str());
	seqNames.push_back(tempSeqNames[tempIndex[0].idx].c_str());
	sizes[0] = tempIndex[0].length;
	sizesWithPad[0] = tempIndex[0].lengthWithPad;
	offsets[0] = 0;
	unsigned sizeTot = tempIndex[0].lengthWithPad;
	strcpy(seqlib+sizeTot, "");
	sizeTot++;

	for (unsigned j=1; j<totSeqRead; ++j) {
		strcpy(seqlib+sizeTot, tempSeqLib[tempIndex[j].idx].c_str());
		seqNames.push_back(tempSeqNames[tempIndex[j].idx].c_str());
		sizes[j] = tempIndex[j].length;
		sizesWithPad[j] = tempIndex[j].lengthWithPad;
		offsets[j] = offsets[j-1] + sizesWithPad[j-1] + 1;
		sizeTot = sizeTot + tempIndex[j].lengthWithPad;
		strcpy(seqlib+sizeTot, "");
		sizeTot++;
	}


// 	for (unsigned j=250290; j<250296; ++j) {
// 		cout << "idx=" << j << " " << seqNames[j] << " size=" << sizes[j] << ", sizes with pad=" << sizesWithPad[j] << ", offset=" << offsets[j] << endl;
// 		for (unsigned cnt=0; cnt<sizesWithPad[j]; ++cnt){
// 			cout << seqlib[(offsets[j] + cnt)];
// 		}
// 		cout << endl;
// 	}

	qf.close();

	return totSeqRead;

}

/**
qui cominciano una serie di funzioni tenute per eventuali scopi futuri ma che allo stato attuale non sono utilizzate
*/

//legge dal file che si suppone contenga una sequenza soltanto, la sequenza stessa
void readSeq(const std::string &fname, std::string &seq, std::string &seqName) {

	QFile qf(fname.c_str());
	if (!qf.exists())
		throw string("sequence file " + fname + " not found");
	if (!qf.open(QIODevice::ReadOnly))
		throw string("sequence file " + fname + " not found");

	long fileSize = qf.size();

	char buf[fileSize];

	long nr = qf.readLine(buf, fileSize);
	seqName.append(buf+1, buf+7);
	
	nr = qf.read(buf, (fileSize-nr));
	seq.append(buf, nr);

	string in("@");
	seq.insert(0, in);
	
// 	unsigned num = count(seq.begin(), seq.end(), '\n');
// 	cout << num << endl;
// 
// 	string::iterator iter = find(seq.begin(), seq.end(), '\n');
// 
// 	for (unsigned cnt=1; cnt <num; ++cnt) {
// 		string::iterator iter2 = find(iter+1, seq.end(), '\n');
// 		seq.erase(iter2);
// 	}

	qf.close();
}

void readSubstitutionMatrix(const string &fname, SubstitutionMatrixMap &mat) {

	const unsigned MAXBUF = 1024;
	char buf[MAXBUF];
	
	bool gotTitles = false;
	string titles = "";

	QFile qf(fname.c_str());
	qf.open(QIODevice::ReadOnly);

	for(;;) {

		long nr = qf.readLine(buf, MAXBUF);
		if (nr>0 && buf[0]!='#') {
			buf[nr-1] = '\0';
			if (gotTitles) {
				if (buf[0]==' ') continue;
				string ts = (buf+1);
				stringstream ss(ts);
				unsigned idx = 0;
				while (ss.good()) {
					int a;
					ss >> a; 
					string key=""; key+=buf[0]; key+=titles[idx++];
					mat[key]=a;
					//cout << key << "=" << a << ' ';
				}
			} else {
				if (buf[0]!=' ') throw string("incorrect substitution table format");
				string ts = buf;
				for(unsigned j=0; j<ts.size(); ++j) {
					if (ts[j]==' ') continue;
					titles += ts[j];
				}
				cout << "titles: " << titles << endl;
				gotTitles = true;
			}
		} else if (nr<1) 
			break;

	}

}

///BLOSUM 50
/*
  Matrix made by matblas from blosum50.iij
  BLOSUM Clustered Scoring Matrix in 1/3 Bit Units
  Blocks Database = /data/blocks_5.0/blocks.dat
  Cluster Percentage: >= 50
  Entropy =   0.4808, Expected =  -0.3573
*/
int abl50[450] = {
 /*A*/	 5,
 /*R*/	-2, 7,
 /*N*/	-1,-1, 7,
 /*D*/	-2,-2, 2, 8,
 /*C*/	-1,-4,-2,-4,13,
 /*Q*/	-1, 1, 0, 0,-3, 7,
 /*E*/	-1, 0, 0, 2,-3, 2, 6,
 /*G*/	 0,-3, 0,-1,-3,-2,-3, 8,
 /*H*/	-2, 0, 1,-1,-3, 1, 0,-2,10,
 /*I*/	-1,-4,-3,-4,-2,-3,-4,-4,-4, 5,
 /*L*/	-2,-3,-4,-4,-2,-2,-3,-4,-3, 2, 5,
 /*K*/	-1, 3, 0,-1,-3, 2, 1,-2, 0,-3,-3, 6,
 /*M*/	-1,-2,-2,-4,-2, 0,-2,-3,-1, 2, 3,-2, 7,
 /*F*/	-3,-3,-4,-5,-2,-4,-3,-4,-1, 0, 1,-4, 0, 8,
 /*P*/	-1,-3,-2,-1,-4,-1,-1,-2,-2,-3,-4,-1,-3,-4,10,
 /*S*/	 1,-1, 1, 0,-1, 0,-1, 0,-1,-3,-3, 0,-2,-3,-1, 5,
 /*T*/	 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 2, 5,
 /*W*/	-3,-3,-4,-5,-5,-1,-3,-3,-3,-3,-2,-3,-1, 1,-4,-4,-3,15,
 /*Y*/	-2,-1,-2,-3,-3,-1,-2,-3, 2,-1,-1,-2, 0, 4,-3,-2,-2, 2, 8,
 /*V*/	 0,-3,-3,-4,-1,-3,-3,-4,-4, 4, 1,-3, 1,-1,-3,-2, 0,-3,-1, 5,
 /*B*/	-2,-1, 4, 5,-3, 0, 1,-1, 0,-4,-4, 0,-3,-4,-2, 0, 0,-5,-3,-4, 5,
 /*Z*/	-1, 0, 0, 1,-3, 4, 5,-2, 0,-3,-3, 1,-1,-4,-1, 0,-1,-2,-2,-3, 2, 5,
 /*X*/	-1,-1,-1,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-1, 0,-3,-1,-1,-1,-1,-1,
 		-1,-1,-1,-1,-2,-1,-1,-2,-1,-1,-1,-1,-1,-2,-2,-1, 0,-3,-1,-1,-1,-1,-1, 7};

  	   //A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X


///BLOSUM 62
/*
 Matrix made by matblas from blosum62.iij
 * column uses minimum score
 BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
 Blocks Database = /data/blocks_5.0/blocks.dat
 Cluster Percentage: >= 62
 Entropy =   0.6979, Expected =  -0.5209
*/

int abl62[450] = {
/*A*/	 4,
/*R*/	-1, 5,
/*N*/	-2, 0, 6,
/*D*/	-2,-2, 1, 6,
/*C*/	 0,-3,-3,-3, 9,
/*Q*/	-1, 1, 0, 0,-3, 5,
/*E*/	-1, 0, 0, 2,-4, 2, 5,
/*G*/	 0,-2, 0,-1,-3,-2,-2, 6,
/*H*/	-2, 0, 1,-1,-3, 0, 0,-2, 8,
/*I*/	-1,-3,-3,-3,-1,-3,-3,-4,-3, 4,
/*L*/	-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,
/*K*/	-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,
/*M*/	-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5,
/*F*/	-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,
/*P*/	-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,
/*S*/	 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4,
/*T*/	 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,
/*W*/	-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11,
/*Y*/	-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,
/*V*/	 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4,
/*B*/	-2,-1, 3, 4,-3, 0, 1,-1, 0,-3,-4, 0,-3,-3,-2, 0,-1,-4,-3,-3, 4,
/*Z*/	-1, 0, 0, 1,-3, 3, 4,-2, 0,-3,-3, 1,-1,-3,-1, 0,-1,-3,-2,-2, 1, 4,
/*X*/	 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1,
		 0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2, 0, 0,-2,-1,-1,-1,-1,-1, 6};

	   //A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X

///BLOSUM 80
/* blosum80 in 1/2 bit units (previous versions had 1/3 bit units) */
/*
 Matrix made by matblas from blosum80.iij
 * column uses minimum score
 BLOSUM Clustered Scoring Matrix in 1/2 Bit Units
 Blocks Database = /data/blocks_5.0/blocks.dat
 Cluster Percentage: >= 80
 Entropy =   0.9868, Expected =  -0.7442
*/

int abl80[450] = {
/*A*/	 5,
/*R*/	-2, 6,
/*N*/	-2,-1, 6,
/*D*/	-2,-2, 1, 6,
/*C*/	-1,-4,-3,-4, 9,
/*Q*/	-1, 1, 0,-1,-4, 6,
/*E*/	-1,-1,-1, 1,-5, 2, 6,
/*G*/	 0,-3,-1,-2,-4,-2,-3, 6,
/*H*/	-2, 0, 0,-2,-4, 1, 0,-3, 8,
/*I*/	-2,-3,-4,-4,-2,-3,-4,-5,-4, 5,
/*L*/	-2,-3,-4,-5,-2,-3,-4,-4,-3, 1, 4,
/*K*/	-1, 2, 0,-1,-4, 1, 1,-2,-1,-3,-3, 5,
/*M*/	-1,-2,-3,-4,-2, 0,-2,-4,-2, 1, 2,-2, 6,
/*F*/	-3,-4,-4,-4,-3,-4,-4,-4,-2,-1, 0,-4, 0, 6,
/*P*/	-1,-2,-3,-2,-4,-2,-2,-3,-3,-4,-3,-1,-3,-4, 8,
/*S*/	 1,-1, 0,-1,-2, 0, 0,-1,-1,-3,-3,-1,-2,-3,-1, 5,
/*T*/	 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-2,-1,-1,-2,-2, 1, 5,
/*W*/	-3,-4,-4,-6,-3,-3,-4,-4,-3,-3,-2,-4,-2, 0,-5,-4,-4,11,
/*Y*/	-2,-3,-3,-4,-3,-2,-3,-4, 2,-2,-2,-3,-2, 3,-4,-2,-2, 2, 7,
/*V*/	 0,-3,-4,-4,-1,-3,-3,-4,-4, 3, 1,-3, 1,-1,-3,-2, 0,-3,-2, 4,
/*B*/	-2,-2, 4, 4,-4, 0, 1,-1,-1,-4,-4,-1,-3,-4,-2, 0,-1,-5,-3,-4, 4,
/*Z*/	-1, 0, 0, 1,-4, 3, 4,-3, 0,-4,-3, 1,-2,-4,-2, 0,-1,-4,-3,-3, 0, 4,
/*X*/	-1,-1,-1,-2,-3,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,-3,-2,-1,-2,-1,-1,
		-1,-1,-1,-2,-3,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,-3,-2,-1,-2,-1,-1, 6};

	   //A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X

void readSubstitutionMatrix(SubstitutionMatrixMap &mat) {

	unsigned matidx = 0;
	
	string titles = "ARNDCQEGHILKMFPSTWYVBZX";

	for (unsigned j=0; j<titles.size(); ++j) {
		for (unsigned i=0; i<=j; ++i) {
			int a = abl50[matidx++];
			string key; key += titles[j]; key += titles[i];
			mat[key]=a;
			//cout << key << "=" << a << ' ';

			if ( i != j ) {
				key = ""; key += titles[i]; key += titles[j];
				mat[key]=a;
				//cout << key << "=" << a << ' ';
			}
		}
	}
}


void readSubstitutionMatrix(SubstitutionMatrixArray &mat) {

	unsigned matidx = 0;
	
	string titles = "ARNDCQEGHILKMFPSTWYVBZX";

	for (unsigned j=0; j<titles.size(); ++j) {
		for (unsigned i=0; i<=j; ++i) {
			int a = abl50[matidx++];
			mat.values[(titles[j])][(titles[i])]=a;
			//cout << key << "=" << a << ' ';

			if ( i != j ) {
				mat.values[(titles[i])][(titles[j])]=a;
				//cout << key << "=" << a << ' ';
			}
		}
	}
}

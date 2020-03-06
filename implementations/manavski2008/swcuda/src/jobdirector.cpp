
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


#include "smithwaterman.h"
#include "hardwarearchitecturenet.h"
#include "hardwarearchitecturecuda.h"
#include "hardwarearchitecturecudaprof.h"
#include "hardwarearchitecturecpuprof.h"
#include "hardwarearchitecturecpusse2.h"
#include "sw_cpu.h"

#include "jobdirector.h"

#include <iostream>
#include <algorithm>
#include <map>
#include <math.h>
#include <unistd.h>
#include <QtCore/QTime>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QDate>
#include <QtCore/QDate>
#include <QtCore/QFileInfo>

using namespace std;

struct scoresOrd
{
  bool operator()(const SWResults num1, const SWResults num2) const
  {
    return (num1.score > num2.score);
  }
};

JobDirector::JobDirector() : seqlib(NULL), offsets(NULL), sizes(NULL), sizesPad(NULL), scores(NULL), end_positions(NULL), numSeqs(0), totBytesUsed(0), devs(0),  numCalls(0), dbName(""), normFact(1.0)
{
	seqlib			= new char[MAXLIBSIZE];
	offsets			= new unsigned[(MAXLIBSIZE / 64)];
	sizes			= new unsigned[(MAXLIBSIZE / 64)];
	sizesPad		= new unsigned[(MAXLIBSIZE / 64)];
}

JobDirector::~JobDirector()
{	
	clear();
}

void JobDirector::init( const std::string dbpath, const BioConfig &cf ) {
	
	bool cpu = cf.getCpuValue();
	bool gpu = cf.getGpuValue();
	unsigned cpuNum = cf.getCpuNumValue();
	unsigned gpuNum = cf.getGpuNumValue();


	unsigned gpuAvailable = HardwareArchitectureCUDA::getAvailableDevicesNumber();
	if (gpu) {
		if ( gpuAvailable >= gpuNum )
			gpuAvailable = gpuNum;
		else
			cout << "WARNING: cannot activate more than " << gpuAvailable << " GPUs. The application will activate only " << gpuAvailable << " devices" << endl;
	}

	if (cf.isEndPositionRequested()) {
		if (cf.getLastKrnlValue())
			throw string("The LAST_KRNL = T implementation does not support COMPUTE_ENDPOSITIONS = T");
		if (cf.getSSE2Value())
			throw string("The SSE2 implementation does not support COMPUTE_ENDPOSITIONS = T");
	}
	
	numSeqs = readdbOrdered( dbpath, seqlib, MAXLIBSIZE, offsets, sizes, seqNamesOrdered, seqNamesNotOrdered, sizesPad );
	dbName = dbpath;

	totBytesUsed = offsets[(numSeqs-1)] + sizesPad[(numSeqs-1)] + 1;

	if (numSeqs < 1) {
		throw string("db contains no sequences");
	}

	if (scores) delete []scores;
	scores = new int[numSeqs];
	memset(scores, 0, numSeqs*sizeof(int));
	if (end_positions) delete []end_positions;
	end_positions = new unsigned[numSeqs];
	memset(end_positions, 0, numSeqs*sizeof(int));
	

	unsigned totd = 0;

	unsigned numDevCPU = ( cpu ) ? cpuNum : 0;
	unsigned numDevGPU = ( gpu ) ? gpuAvailable : 0;
	unsigned numDevNET = 0;

	//controllare che in totale non vengano lanciati più di NUM_MAX_CPU_CORES threads. (i dispositivi di rete sono satti ignorati per ovvi motivi)
	unsigned bits = static_cast<unsigned>(log10(NUM_MAX_CPU_CORES)*log10(2));
	char buf[bits];

	if ( sprintf(buf, "%d", NUM_MAX_CPU_CORES-numDevGPU) < 0 )
		throw string("cannot convert numbers to string");

	if (numDevCPU > NUM_MAX_CPU_CORES-numDevGPU) {
		string ex ("WARNING: cannot run more than ");
		ex = ex + buf + " threads on the CPU. The application will run only " + buf + " threads on the CPU";
		cout << ex << endl;
		numDevCPU = NUM_MAX_CPU_CORES-numDevGPU;
	}


	if ( cf.getDebugValue() != 0 )  {
		cout << "Found " << HardwareArchitectureCUDA::getAvailableDevicesNumber() << " CUDA devices. " << endl << endl;
	}


	//total number of devices available
	totd = numDevCPU + numDevGPU + numDevNET;

	if (totd < 1)
		throw string("there aren't available devices");
	
	devs.resize(totd);
	
	char str[10];

	unsigned off = 0;

	//creation of GPU objects
	for (unsigned cnt = 0; cnt<numDevGPU; ++cnt) {
		if (cnt) {
			devs[off].device = HardwareArchitectureCUDA::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizesPad, scores, end_positions);
			devs[off].megaCUPS = 1450;
		} else {
			if (cf.getLastKrnlValue())
				//nella nuova versione con profile passiamo direttamente le size senza padding
				devs[off].device = HardwareArchitectureCUDAProf::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizes, scores, end_positions);
			else
				devs[off].device = HardwareArchitectureCUDA::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizes, scores, end_positions);
			
			devs[off].megaCUPS = 1750;
		}
		if ( sprintf(str,"%d",cnt) < 0 )
			throw string("cannot convert numbers to string");
		devs[off].type.append("GPU");
		devs[off].type.append(str);
		
		off++;
	}

	//creation of CPU objects
	for (unsigned cnt = 0; cnt<numDevCPU; ++cnt) {

		if ( cf.getSSE2Value() )
			devs[off].device = HardwareArchitectureCPUSSE2::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizes, scores, end_positions);
		else
			devs[off].device = HardwareArchitectureCPU::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizes, scores, end_positions);

		devs[off].megaCUPS = 40;
		if ( sprintf(str,"%d",cnt) < 0 )
			throw string("cannot convert numbers to string");
		devs[off].type.append("CPU");
		devs[off].type.append(str);
		
		off++;
	}

	//creation of NET objects
	for (unsigned cnt = 0; cnt<numDevNET; ++cnt) {
		devs[off].device = HardwareArchitectureNet::getDevice(cnt, seqlib, totBytesUsed, numSeqs, offsets, sizes, scores, end_positions);
		devs[off].megaCUPS = 100;
		off++;
	}
}

unsigned JobDirector::getSequenceCount( ) const
{
	return numSeqs;
}


void JobDirector::smithWatermanDyn(const char * strToAlign, const int alpha, const int beta, const std::string subMat, const BioConfig &cf, const unsigned startPos, const unsigned stopPos, string &searchedStringName, string &outFile, unsigned seqPos, unsigned alignOffsets)
{
	///sezione old-cpu
// 	FILE* outStream2 = fopen("cpuTest.dat", "w");
// 	int result[2];
// 	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
// 		string sb = seqlib + offsets[cnt];
// 		swRect( sb, strToAlign, alpha, beta, result);
// 		fprintf(outStream2, "seq %u name %s SCORE = %u\n", cnt, seqNames[cnt].c_str(), result[0]);
// 	}
// 	fclose(outStream2);
	///fine

	QTime tott;
	tott.start();

	numCalls++;

	// ripartizione carico
	repartition(startPos, stopPos, cf);

	for (unsigned cnt=0; cnt<devs.size(); ++cnt) {
		if (cf.getDebugValue()!=0)
			cout << "Start and stop positions for " << devs[cnt].type << ": " << devs[cnt].startPos << ", " << devs[cnt].stopPos << endl;
	}

	//making padding for the GPU
	string strToAlignPadded = strToAlign;
	unsigned rsz = (strToAlignPadded.size()-1) % 64;
	if (rsz != 0) {
		for( unsigned j=0; j < 64 - rsz; ++j) {
			strToAlignPadded += "@";
		}
	}

	//chiamata delle varie calc
	for (unsigned cnt=0; cnt<devs.size(); ++cnt) {
		if (!devs[cnt].sw) {
			devs[cnt].sw = new SmithWaterman(devs[cnt].device);
			devs[cnt].sw->start();
		}
		
		if (cf.getDebugValue()!=0)
			cout << "size last seq " << devs[cnt].type << "= " << sizes[devs[cnt].stopPos] << endl;

		devs[cnt].sw->setJob(strToAlignPadded.c_str(), strlen(strToAlign), alpha, beta, subMat, devs[cnt].startPos, devs[cnt].stopPos, cf.isEndPositionRequested(), cf.getDebugValue());
	}

	for (unsigned cnt=0; cnt<devs.size() ; ++cnt)  {
		if (cf.getDebugValue()!=0)
			cout << "waiting " << devs[cnt].type << " ..." << endl;
		devs[cnt].sw->wait();
	}

	///stampa output
	scoresToFile (outFile, searchedStringName, cf, seqPos);

	//registro i megaCUPS
	for (unsigned cnt=0; cnt<devs.size() ; ++cnt)  {
		devs[cnt].megaCUPS = ( (devs[cnt].megaCUPS*numCalls) + devs[cnt].sw->getMegaCUPS() ) / (numCalls+1);
		if (cf.getDebugValue()!=0)
			cout << "MegaCUPS for " << devs[cnt].type << ": " << devs[cnt].sw->getMegaCUPS() << " (last run)" << endl;
	}

	//calolo megaCUPS totali
	int tot_time = tott.elapsed();
	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	tot_time = (tot_time > 1) ? tot_time : 1;

	unsigned mc = static_cast<unsigned>((static_cast<double>(seqLibSize) / tot_time) * (strlen(strToAlign)/1048.576));

	QString currDate = (QDate::currentDate()).toString("dd.MM.yyyy");
	QString currTime = (QTime::currentTime()).toString("hh:mm:ss.zzz");

	cout << endl << "Results for sequence " << (numCalls-1+alignOffsets) << "\n";
	cout << "\tAlignment finished on " << currDate.toStdString().c_str() << " at " << currTime.toStdString().c_str() << "\n";

	cout << "\tElapsed time (s): " << tot_time/1000.0 << "\n";
	cout << "\tMegaCUPS: " << mc << endl << endl;

}

void JobDirector::smithWatermanMultiSeq ( const string &seqFileName, const string &libFileName, const int alpha, const int beta, const std::string subMat, const BioConfig &cf, const unsigned startPos, const unsigned stopPos, unsigned alignOffsets) {
	
	const unsigned MAX_BUF = 1024;
	char buf[MAX_BUF];

	QFile qf(seqFileName.c_str());
	if (!qf.exists())
		throw string("sequence file \"" + seqFileName + "\" not found");
	if (!qf.open(QIODevice::ReadOnly))
		throw string("sequence file \"" + seqFileName + "\" not found");

	string curseq = "@";
	string curname = "";
	string curTotName = "";
	string outFile = "";
	unsigned counter;

	unsigned totSeqRead = 0;

	QTime tott;
	tott.start();

	unsigned doneSeqs = 0;
	unsigned ignSeqs = 0;

	unsigned seqLibSize = 0;
	for (unsigned cnt=startPos; cnt<=stopPos; ++cnt) {
		seqLibSize += sizes[cnt];
	}

	QString currDateInit = (QDate::currentDate()).toString("dd_MM_yyyy");
	QString currTimeInit = (QTime::currentTime()).toString("hh_mm_ss");

	QFileInfo seqFileInfo(seqFileName.c_str());
	QString seqBase = seqFileInfo.baseName();
	QFileInfo libFileInfo(libFileName.c_str());
	QString libBase = libFileInfo.baseName();

	//cominciamo la lettura delle sequenze da allineare. Bisogna tener conto dell'eventuale offset richiesto dall'utente.
	for(;;) {

		long nr = qf.readLine(buf, MAX_BUF);
		if (nr>0 && buf[0]!='>') {
			buf[nr-1] = '\0';
			curseq += buf;
		} else if (nr>0) {
			if ( (totSeqRead-1) >= alignOffsets  ) {
				if ( curseq.size() > 1  ) {
	
					QString currDate = (QDate::currentDate()).toString("dd.MM.yyyy");
					QString currTime = (QTime::currentTime()).toString("hh:mm:ss.zzz");
	
					cout << endl << "###########################ALIGNMENT_FEATURES###########################" << endl << endl;
					cout << "Aligned sequence: " << curname.c_str() << " -> " << curseq.c_str() << endl << endl;
					cout << "Matrix used: " << subMat.c_str() << endl << endl;
					cout << "Database used:" << endl;
					cout << "\tName: " << dbName.c_str() << endl;
					cout << "\tSequences number: " << numSeqs << endl;
					cout << "\tResidues number: " << seqLibSize << endl << endl;
					cout << "Alignment started on " << currDate.toStdString().c_str() << " at " << currTime.toStdString().c_str() << endl << endl;
					cout << "-----------------------------------------------------------------------" << endl << endl;
	
					counter = doneSeqs + alignOffsets;

					outFile = seqBase.toStdString() + "_vs_" + libBase.toStdString() + "_" + currDateInit.toStdString() + "_" + currTimeInit.toStdString() + ".out";

					//porzione aggiunta per la normalizzazione dei risultati
					if (cf.getScalingFact()) {
						int result[2];
						swRect( curseq, curseq, alpha, beta, result);
						normFact = result[0];
					}
	
					if (curseq.size() > MAX_SEARCHED_SEQUENCE_LENGTH) {
						cout << "ERROR: Sequence " << curname << " is longer than " << MAX_SEARCHED_SEQUENCE_LENGTH << ". It will be ignored." << endl;
						++ignSeqs;
					} else {
						smithWatermanDyn( curseq.c_str(), alpha, beta, subMat, cf, startPos, stopPos, curTotName, outFile, counter, alignOffsets);
						++doneSeqs;
					}
	
					// azzeramento del contatore della sequenza corrente
					curseq = "@";
					curname = "";
					outFile = "";
				}
			} else {
				curseq = "@";
				curname = "";
				outFile = "";
			}
			
			// e qui leggiamo il nome della nuova sequenza che sta iniziando
			size_t st = strlen(buf);
			(st>=7) ? curname.append(buf+1, 6) : curname.append(buf+1, st-1);
			curTotName = "";
			curTotName.append(buf+1, st-1);

			++totSeqRead;

		} else {
			if ( (totSeqRead-1) >= alignOffsets  ) {
				if ( curseq.size() > 1 ) {
	
					QString currDate = (QDate::currentDate()).toString("dd.MM.yyyy");
					QString currTime = (QTime::currentTime()).toString("hh:mm:ss.zzz");
					
					cout << endl << "###########################ALIGNMENT_FEATURES###########################" << endl << endl;
					cout << "Aligned sequence: " << curname.c_str() << " -> " << curseq.c_str() << endl << endl;
					cout << "Matrix used: " << subMat.c_str() << endl << endl;
					cout << "Database used:" << endl;
					cout << "\tName: " << dbName.c_str() << endl;
					cout << "\tSequences number: " << numSeqs << endl;
					cout << "\tResidues number: " << seqLibSize << endl << endl;
					cout << "Alignment started on " << currDate.toStdString().c_str() << " at " << currTime.toStdString().c_str() << endl << endl;
					cout << "-----------------------------------------------------------------------" << endl << endl;
	
					counter = doneSeqs + alignOffsets;
					
					outFile = seqBase.toStdString() + "_vs_" + libBase.toStdString() + "_" + currDateInit.toStdString() + "_" + currTimeInit.toStdString() + ".out";
					
					//porzione aggiunta per la normalizzazione dei risultati
					if (cf.getScalingFact()) {
						int result[2];
						swRect( curseq, curseq, alpha, beta, result);
						normFact = result[0];
					}

					if (curseq.size() > MAX_SEARCHED_SEQUENCE_LENGTH) {
						cout << "ERROR: Sequence " << curname << " is longer than " << MAX_SEARCHED_SEQUENCE_LENGTH << ". It will be ignored." << endl;
					} else {
						smithWatermanDyn( curseq.c_str(), alpha, beta, subMat, cf, startPos, stopPos, curTotName, outFile, counter, alignOffsets);
						++doneSeqs;
					}
				}
			} 
			break;
		}
	}

	int tot_time = tott.elapsed();
	cout << "########################################################################" << endl;

	cout << endl << "###########################FINAL_OVERVIEW###########################" << endl << endl;
	cout << "Sequences aligned: " << doneSeqs << endl;
	cout << "Sequences ignored: " << ignSeqs << endl;
	cout << "Total time for " << doneSeqs << " sequences (s): " << tot_time/1000.0 << endl << endl;

	// si chiede ai worker di smith-waterman di uscire perchè non c'e' piu' lavoro
	for (unsigned cnt=0; cnt<devs.size() ; ++cnt)  {
		if (devs[cnt].sw)
			devs[cnt].sw->quit();
	}
	sleep(1);

	if (cf.getDebugValue()) printf("JobDirector ha segnalato l'uscita ai workers\n");
}

void JobDirector::scoresToFile (const std::string &outFile, const string &searchedStringName, const BioConfig &cf, const unsigned seqPos) {
	
	multimap<SWResults, unsigned, scoresOrd> scores_Idx;

	for (unsigned cnt=0; cnt<numSeqs; ++cnt) {
//printf("%d\t", scores[cnt]);
		if (scores[cnt]/normFact >= cf.getScoresThrld()) {
			SWResults r(scores[cnt]/normFact, end_positions[cnt]);
			scores_Idx.insert(make_pair(r, cnt));
		}
	}

	string outPath("");

	outPath = cf.getOutDirValue() + "/" + outFile;

	QFile outStream(outPath.c_str());
	if ( !outStream.open(QIODevice::Append) )
		throw string("cannot write into the file " + outPath);

	{
		
		std::string buf(MAX_NUM_CHAR, ' ');
		std::string reset(MAX_NUM_CHAR, ' ');
 
		//contiene il numero di query allineata
		char counter[MAX_NUM_SEQUENCE_ALIGNED_COUNTER];
		if ( sprintf(counter, "%d", seqPos) < 0 )
			throw string("cannot convert numbers to string");

		if (outStream.write("\n------------------------------------------------------------------------------------------------------\n\n") < 0 )
			throw string("cannot write into the file " + outPath);

		std::string outString("QUERY N° ");
		outString.append(counter);
		outString.append(" -> " + searchedStringName);

		if (outStream.write(outString.c_str()) < 0)
			throw string("cannot write into the file " + outPath);
		if (outStream.write("\n------------------------------------------------------------------------------------------------------\n\n") < 0 )
			throw string("cannot write into the file " + outPath);
		if ( outStream.write("SCORE     \t     Q_END\t     S_END\tNAME\n") < 0 )
			throw string("cannot write into the file " + outPath);

		unsigned cnt = 0;
		for (multimap<SWResults, unsigned>::iterator it=scores_Idx.begin(); it != scores_Idx.end(); ++it) {
			unsigned idx = (*it).second;
			double score = (*it).first.score;
			unsigned q_endpos = (*it).first.q_endpos;
			unsigned s_endpos = (*it).first.s_endpos;
			
			//resetto il buffer
			buf.assign(reset);

			//salvo i primi MAX_NUM_CHAR del nome della sequenza
			(seqNamesOrdered[idx].size()<MAX_NUM_CHAR) ? copy(seqNamesOrdered[idx].begin(), seqNamesOrdered[idx].end()-1, buf.begin()) : copy(seqNamesOrdered[idx].begin(), seqNamesOrdered[idx].begin()+MAX_NUM_CHAR, buf.begin());
 
			char outBuf[MAX_NUM_CHAR+100];

			if ( sprintf(outBuf, "%f\t%10u\t%10u\t%s\n", score, q_endpos, s_endpos, buf.c_str()) < 0 )
				throw string("cannot convert numbers to string for the file " + outPath);

			if ( outStream.write(outBuf) < 0 )
				throw string("cannot write into the file " + outPath);
			++cnt;
		}

	}
	outStream.close();

}

void JobDirector::clear( )
{
	if (seqlib) {
		delete [] seqlib;
		seqlib = NULL;
		delete [] offsets;
		delete [] sizes;
		delete [] sizesPad;
	}
	if (scores) {
		delete []scores;
		scores = NULL;
	}
	if (end_positions) {
		delete []end_positions;
		end_positions = NULL;
	}

}

unsigned JobDirector::repartition(const unsigned startPos, const unsigned stopPos, const BioConfig &cf) {

	cout << "Repartition a total of " << devs.size() << " devices. " << endl << endl;

	unsigned totMegaCUPS = 0;

	//calcolo il numero totale di CUPS
	for (unsigned cnt=0; cnt<devs.size(); ++cnt)
		totMegaCUPS += devs[cnt].megaCUPS;
	cout << "Total average MegaCUPS power available: " << totMegaCUPS << endl;

	//calcolo i bytes da fare per ogni devices spezzando il for per permettere all'ultimo device di prendere tutti i bytes residui
	unsigned tempBytesGiven = 0;
	for (unsigned cnt=0; cnt<devs.size()-1; ++cnt) {

		double devPerCent = (devs[cnt].megaCUPS * 100.0) / static_cast<double>(totMegaCUPS);
		devs[cnt].bytesToDo = static_cast<unsigned>((totBytesUsed / 100.0 ) * devPerCent);
		tempBytesGiven += devs[cnt].bytesToDo;
	}
	devs[devs.size()-1].bytesToDo = totBytesUsed - tempBytesGiven;

	for (unsigned cnt=0; cnt<devs.size(); ++cnt) {
		if (cf.getDebugValue()!=0)
			cout << "Bytes to do for " << devs[cnt].type << ": " << devs[cnt].bytesToDo << endl;
	}

	//initializing the start and stop positions for all devices
	unsigned lastIdx = startPos;
	unsigned bytes = 0;
	for (unsigned cnt=0; cnt<devs.size()-1; ++cnt) {
		devs[cnt].startPos = lastIdx;
		
		bytes += devs[cnt].bytesToDo;

		unsigned* libOffset = lower_bound(offsets, offsets+numSeqs, bytes);

		unsigned dis = distance(offsets,libOffset-1);

		devs[cnt].stopPos = dis;
		
		lastIdx = dis + 1;
	}
	devs[devs.size()-1].startPos = lastIdx;
	devs[devs.size()-1].stopPos = stopPos;

	return totMegaCUPS;
}




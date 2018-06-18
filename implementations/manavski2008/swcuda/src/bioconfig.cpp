
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


#include "bioconfig.h"

#include <iostream>


BioConfig::BioConfig(const QString & configFile) : ConfigFile(configFile), cpu(), gpu(), cpuNum(), gpuNum(), debug(), mat(), gapFirst(), gapNext(), scoreThrld(), outDir() {
	if (!isGood())
		throw std::string("cannot open configuration file: config.ini");

	QString tempValue;

	tempValue = Value("main", "CPU", "T");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("CPU parameter must be T or F");
	cpu = (tempValue!="T") ? 0 : 1;
	if (!cpu)
		std::cout << "WARNING: the application is running with no CPU activated. This could cause serious problems if input sequences are longer than 360!" << std::endl;

	tempValue = Value("main", "GPU", "T");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("GPU parameter must be T or F");
	gpu = (tempValue!="T") ? 0 : 1;
	
	tempValue = Value("main", "CPUNUM", "1");
	cpuNum = tempValue.toUInt();
		
	tempValue = Value("main", "GPUNUM", "1");
	gpuNum = tempValue.toUInt();
	
	tempValue = Value("main", "DEBUG", "0");
	debug = tempValue.toUInt();
	
	tempValue = Value("main", "SSE2", "F");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("SSE2 parameter must be T or F");
	sse2 = (tempValue!="T") ? 0 : 1;
	
	tempValue = Value("main", "LAST_KRNL", "F");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("LAST_KRNL parameter must be T or F");
	lastKrnl = (tempValue!="T") ? 0 : 1;
	
	mat = Value("main", "MAT", "BL50");
	if (mat != "BL50" && mat!="BL62" && mat!="BL90" && mat != "DNA1")
		throw std::string("unknown substitution matrix. Allowed matrix: BL50, BL62, BL90, DNA1.");	

	tempValue = Value("main", "GAP_FIRST", "10");
	gapFirst = tempValue.toUInt();
	if (gapFirst>128)
		throw std::string("invalid gap extension");
	
	tempValue = Value("main", "GAP_NEXT", "2");
	gapNext = tempValue.toUInt();
	if (gapNext<1 || gapNext>128)
		throw std::string("invalid gap extension");
	
	tempValue = Value("main", "SCORES_THRLD", "0");
	scoreThrld = tempValue.toDouble();
	
	outDir = Value("main", "OUTDIR", "result");
	QDir dir(outDir);
	if (!dir.exists()) {
			throw std::string("cannot find output directory");
	}

	if ( cpu  && cpuNum < 1 )
		throw std::string("cannot activate CPU without running at least one thread on it (CPUNUM=0)");
	if ( gpu  && gpuNum < 1 )
		throw std::string("cannot activate GPU without using at least one device (GPUNUM=0)");
	if ( !cpu && !gpu )
		throw std::string("cannot run the application without any device activated");
	if (debug < 1 && (!gpu || gpuNum<1))
		throw std::string("cannot run this sofware without activating at least a GPU");

	tempValue = Value("main", "SCORE_SCALING_FACTOR", "F");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("SCORE_SCALING_FACTOR parameter must be T or F");
	scalFact = (tempValue!="T") ? 0 : 1;


	tempValue = Value("main", "COMPUTE_ENDPOSITIONS", "F");
	if (tempValue!="T" && tempValue!="F")
		throw std::string("COMPUTE_ENDPOSITIONS parameter must be T or F");
	endbleEndpos = (tempValue!="T") ? 0 : 1;
}

BioConfig::~BioConfig() {
}

bool BioConfig::getCpuValue() const {
	return cpu;
}

bool BioConfig::getGpuValue() const {
	return gpu;
}

unsigned BioConfig::getCpuNumValue() const {
	return cpuNum;
}

unsigned BioConfig::getGpuNumValue() const {
	return gpuNum;
}

unsigned BioConfig::getDebugValue() const {
	return debug;
}

bool BioConfig::getSSE2Value() const {
	return sse2;
}

bool BioConfig::getLastKrnlValue() const {
	return lastKrnl;
}

std::string BioConfig::getMatValue() const {
	return mat.toStdString();
}

unsigned BioConfig::getGapFirst() const {
	return gapFirst;
}

unsigned BioConfig::getGapNext() const {
	return gapNext;
}

double BioConfig::getScoresThrld() const {
	return scoreThrld;
}

std::string BioConfig::getOutDirValue() const {
	return outDir.toStdString();
}

bool BioConfig::getScalingFact() const {
	return scalFact;
}

bool BioConfig::isEndPositionRequested() const
{
	return endbleEndpos;
}



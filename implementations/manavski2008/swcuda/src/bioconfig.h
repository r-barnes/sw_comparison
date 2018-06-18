

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

#ifndef BIOCONFIG_H
#define BIOCONFIG_H

#include <gpubiolib/ConfigFile.h>
#include <QtCore/QDir>

/**
	@author Svetlin Manavski <svetlin.a@manavski.com>
*/

/**
Questa classe gestisce il file di configurazione (config.ini).
Ne preleva tutti i valori assegnandoli a variabili membro ed esegue vari controlli.
Dettaglio importante: se non trova un dato campo nel file assegna alla variabile relativa un valore di default.
*/

const unsigned NUM_MAX_CPU_CORES = 4;

class BioConfig : protected ConfigFile
{
public:
    BioConfig(const QString & configFile);

    ~BioConfig();

	bool getCpuValue() const;
	bool getGpuValue() const;
	unsigned getCpuNumValue() const;
	unsigned getGpuNumValue() const;
	unsigned getDebugValue() const;
	bool getSSE2Value() const;
	bool getLastKrnlValue() const;
	std::string getMatValue() const;
	unsigned getGapFirst() const;
	unsigned getGapNext() const;
	double getScoresThrld() const;
	std::string getOutDirValue() const;
	bool getScalingFact() const;
	bool isEndPositionRequested() const;

private:
	bool cpu;
	bool gpu;
	unsigned cpuNum;
	unsigned gpuNum;
	unsigned debug;
	bool sse2;
	bool lastKrnl;
	QString mat;
	unsigned gapFirst;
	unsigned gapNext;
	double scoreThrld;
	QString outDir;
	bool scalFact;
	bool endbleEndpos;
};

#endif

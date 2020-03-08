/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2020 UniPro <ugene@unipro.ru>
 * http://ugene.net
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */

#include <U2Core/DNAAlphabet.h>
#include <U2Core/U2SafePoints.h>
#include <U2Core/U2SequenceDbi.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2DbiUtils.h>

#include <math.h>

#include "DNAStatisticsTask.h"

namespace U2 {

DNAStatistics::DNAStatistics() {
    clear();
}

void DNAStatistics::clear() {
    length = 0;

    gcContent = 0;
    molarWeight = 0;
    molarExtCoef = 0;
    meltingTm = 0;

    nmoleOD260 = 0;
    mgOD260 = 0;

    molecularWeight = 0;
    isoelectricPoint = 0;
}

static QVector<double> createProteinMWMap(){
    QVector<double> mwMap(256, 0);
    mwMap['A'] = 89.09; // ALA
    mwMap['R'] = 174.20; // ARG
    mwMap['N'] = 132.12; // ASN
    mwMap['D'] = 133.10; // ASP
    mwMap['B'] = 132.61; // ASX
    mwMap['C'] = 121.15; // CYS
    mwMap['Q'] = 146.15; // GLN
    mwMap['E'] = 147.13; // GLU
    mwMap['Z'] = 146.64; // GLX
    mwMap['G'] = 75.07; // GLY
    mwMap['H'] = 155.16; // HIS
    mwMap['I'] = 131.17; // ILE
    mwMap['L'] = 131.17; // LEU
    mwMap['K'] = 146.19; // LYS
    mwMap['M'] = 149.21; // MET
    mwMap['F'] = 165.19; // PHE
    mwMap['P'] = 115.13; //PRO
    mwMap['S'] = 105.09; // SER
    mwMap['T'] = 119.12; // THR
    mwMap['W'] = 204.23; // TRP
    mwMap['Y'] = 181.19; // TYR
    mwMap['V'] = 117.15; // VAL
    return mwMap;
}

static QVector<double> createPKAMap() {
    QVector<double> res(256, 0);
    res['D'] = 4.0;
    res['C'] = 8.5;
    res['E'] = 4.4;
    res['Y'] = 10.0;
    res['c'] = 3.1; // CTERM
    res['R'] = 12.0;
    res['H'] = 6.5;
    res['K'] = 10.4;
    res['n'] = 8.0; // NTERM
    return res;
}

static QVector<int> createChargeMap() {
    QVector<int> res(256, 0);
    res['D'] = -1;
    res['C'] = -1;
    res['E'] = -1;
    res['Y'] = -1;
    res['c'] = -1; // CTERM
    res['R'] = 1;
    res['H'] = 1;
    res['K'] = 1;
    res['n'] = 1; // NTERM
    return res;
}


QVector<double> DNAStatisticsTask::pMWMap = createProteinMWMap();
QVector<double> DNAStatisticsTask::pKaMap = createPKAMap();
QVector<int> DNAStatisticsTask::pChargeMap = createChargeMap();

DNAStatisticsTask::DNAStatisticsTask(const DNAAlphabet* alphabet,
                                     const U2EntityRef seqRef,
                                     const QVector<U2Region>& regions)
    : BackgroundTask< DNAStatistics > (tr("Calculate sequence statistics"), TaskFlag_None),
      alphabet(alphabet),
      seqRef(seqRef),
      regions(regions),
      nA(0),
      nC(0),
      nG(0),
      nT(0)
{
    SAFE_POINT_EXT(alphabet != NULL, setError(tr("Alphabet is NULL")), );
}

void DNAStatisticsTask::run() {
    computeStats();
}

void DNAStatisticsTask::computeStats() {
    U2OpStatus2Log os;
    DbiConnection dbiConnection(seqRef.dbiRef, os);
    CHECK_OP(os, );

    U2SequenceDbi* sequenceDbi = dbiConnection.dbi->getSequenceDbi();
    CHECK(sequenceDbi != NULL, );
    SAFE_POINT_EXT(alphabet != NULL, setError(tr("Alphabet is NULL")), );
    qint64 totalLength = U2Region::sumLength(regions);
    qint64 processedLength = 0;

    foreach (const U2Region& region, regions) {
        QList<U2Region> blocks = U2Region::split(region, 1024 * 1024);
        foreach(const U2Region& block, blocks) {
            if (isCanceled() || hasError()) {
                break;
            }
            QByteArray seqBlock = sequenceDbi->getSequenceData(seqRef.entityId, block, os);
            CHECK_OP(os,);
            const char* sequenceData = seqBlock.constData();
            for (int i = 0, n = seqBlock.size(); i < n; i++) {
                char c = sequenceData[i];
                if (c == 'A') {
                    nA++;
                } else if (c == 'G') {
                    nG++;
                } else if (c == 'T' || c == 'U') {
                    nT++;
                } else if (c == 'C') {
                    nC++;
                }

                if (alphabet->isAmino()) {
                    result.molecularWeight += pMWMap.value(c);
                }
            }
            processedLength += block.length;
            stateInfo.setProgress(processedLength * 100 / totalLength);
            CHECK_OP(stateInfo,);
        }
    }

    result.length = totalLength;

    // get alphabet type
    if (alphabet->isNucleic()) {
        result.gcContent = 100.0 * (nG + nC) / (double) totalLength;

        // Calculating molar weight
        // Source: http://www.basic.northwestern.edu/biotools/oligocalc.html
        if (alphabet->isRNA()) {
            result.molarWeight = nA * 329.21 + nT * 306.17 + nC * 305.18 + nG * 345.21 + 159.0;
        } else {
            result.molarWeight = nA * 313.21 + nT * 304.2 + nC * 289.18 + nG * 329.21 + 79;
        }

        result.molarExtCoef = nA*15400 + nT*8800 + nC*7300 + nG*11700;

        if (totalLength < 15) {
            result.meltingTm = (nA + nT) * 2 + (nG + nC) * 4;
        } else if (nA + nT + nG + nC != 0) {
            result.meltingTm = 64.9 + 41 * (nG + nC - 16.4) / (double) (nA + nT + nG + nC);
        }

        if (result.molarExtCoef != 0) {
            result.nmoleOD260 = (double)1000000 / result.molarExtCoef;
        }

        result.mgOD260 = result.nmoleOD260 * result.molarWeight * 0.001;

    } else if (alphabet->isAmino()) {
        static const double MWH2O = 18.0;
        result.molecularWeight = result.molecularWeight - (totalLength - 1) * MWH2O;
        result.isoelectricPoint = calcPi(sequenceDbi);
    }
}

double DNAStatisticsTask::calcPi(U2SequenceDbi* sequenceDbi) {
    U2OpStatus2Log os;
    QVector<qint64> countMap(256, 0);
    foreach (const U2Region& region, regions) {
        QList<U2Region> blocks = U2Region::split(region, 1024 * 1024);
        foreach(const U2Region& block, blocks) {
            if (isCanceled() || hasError()) {
                break;
            }
            QByteArray seqBlock = sequenceDbi->getSequenceData(seqRef.entityId, block, os);
            CHECK_OP(os, 0);
            const char* sequenceData = seqBlock.constData();
            for (int i = 0, n = seqBlock.size(); i < n; i++) {
                char c = sequenceData[i];
                if (pKaMap[c] != 0) {
                    countMap[c]++;
                }
            }
            CHECK_OP(stateInfo, 0);
        }
    }

    countMap['c'] = 1;
    countMap['n'] = 1;

    static const double CUTOFF = 0.001;
    static const double INITIAL_CUTOFF = 2.0;

    double step = INITIAL_CUTOFF;
    double pH = 0;
    while (step > CUTOFF) {
        if (calcChargeState(countMap, pH) > 0) {
            pH += step;
        } else {
            step *= 0.5;
            pH -= step;
        }
    }
    return pH;
}

double DNAStatisticsTask::calcChargeState(const QVector<qint64>& countMap, double pH ) {
    double chargeState = 0.;
    for (int i = 0; i < countMap.length(); i++) {
        if (isCanceled() || hasError()) {
            break;
        }
        double pKa = pKaMap[i];
        double charge = pChargeMap[i];
        chargeState += countMap[i] * charge / (1 + pow(10.0, charge * (pH - pKa)));
    }
    return chargeState;
}


} // namespace

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

#ifndef _U2_SPADES_GENOME_ASSEMBLY_DIALOG_FILLER_H_
#define _U2_SPADES_GENOME_ASSEMBLY_DIALOG_FILLER_H_

#include "utils/GTUtilsDialog.h"

namespace U2 {
using namespace HI;

class SpadesGenomeAssemblyDialogFiller : public Filler {
public:
    SpadesGenomeAssemblyDialogFiller(HI::GUITestOpStatus &os, QString _library, QStringList _leftReads, QStringList _rightReads, QString _output, QString _datasetType = "", QString _runningMode = "", QString _kmerSizes = "", int _numThreads = 0, int _memLimit = 0)
        : Filler(os, "GenomeAssemblyDialog"),
          library(_library),
          leftReads(_leftReads),
          rightReads(_rightReads),
          output(_output),
          datasetType(_datasetType),
          runningMode(_runningMode),
          kmerSizes(_kmerSizes),
          numThreads(_numThreads),
          memLimit(_memLimit) {
    }
    virtual void commonScenario();

protected:
    QString library;
    QStringList leftReads;
    QStringList rightReads;
    QString output;
    QString datasetType;
    QString runningMode;
    QString kmerSizes;
    int numThreads;
    int memLimit;
};

}    // namespace U2

#endif    // SPADESGENOMEASSEMBLYDIALOGFILLER_H

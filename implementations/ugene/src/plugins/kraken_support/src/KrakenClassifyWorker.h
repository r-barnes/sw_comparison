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

#ifndef _U2_KRAKEN_CLASSIFY_WORKER_H_
#define _U2_KRAKEN_CLASSIFY_WORKER_H_

#include <U2Lang/LocalDomain.h>

#include "../ngs_reads_classification/src/TaxonomySupport.h"
#include "KrakenClassifyTask.h"

namespace U2 {
namespace LocalWorkflow {

class KrakenClassifyWorker : public BaseWorker {
    Q_OBJECT
public:
    KrakenClassifyWorker(Actor *actor);

    void init();
    Task *tick();
    void cleanup();

private slots:
    void sl_taskFinished(Task *task);

private:
    bool isReadyToRun() const;
    bool dataFinished() const;

    KrakenClassifyTaskSettings getSettings(U2OpStatus &os);
    IntegralBus *input;
    IntegralBus *output;

    bool pairedReadsInput;

    static const QString KRAKEN_DIR;
};

}    // namespace LocalWorkflow
}    // namespace U2

#endif    // _U2_KRAKEN_CLASSIFY_WORKER_H_

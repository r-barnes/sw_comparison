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

#include "CutadaptSupport.h"

#include <U2Core/AppContext.h>
#include <U2Core/AppSettings.h>
#include <U2Core/DataPathRegistry.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2SafePoints.h>

#include "python/PythonSupport.h"

namespace U2 {

const QString CutadaptSupport::ET_CUTADAPT = "cutadapt";
const QString CutadaptSupport::ET_CUTADAPT_ID = "USUPP_CUTADAPT";
const QString CutadaptSupport::ADAPTERS_DIR_NAME = "adapters";
const QString CutadaptSupport::ADAPTERS_DATA_NAME = "Adapters file";

CutadaptSupport::CutadaptSupport(const QString &id, const QString &name, const QString &path)
    : ExternalTool(id, name, path) {
    if (AppContext::getMainWindow()) {
        icon = QIcon(":external_tool_support/images/cmdline.png");
        grayIcon = QIcon(":external_tool_support/images/cmdline_gray.png");
        warnIcon = QIcon(":external_tool_support/images/cmdline_warn.png");
    }
    executableFileName = "cutadapt.py";
    validMessage = "cutadapt version";
    description = tr("<i>cutadapt</i> removes adapter sequences from high-throughput sequencing data. This is necessary when the reads are longer than the molecule that is sequenced, such as in microRNA data.");

    versionRegExp = QRegExp("cutadapt version (\\d+.\\d+.\\d+)");
    validationArguments << "--help";
    toolKitName = "cutadapt";

    U2DataPathRegistry *dpr = AppContext::getDataPathRegistry();
    if (dpr != NULL) {
        U2DataPath *dp = new U2DataPath(ADAPTERS_DATA_NAME, QString(PATH_PREFIX_DATA) + ":" + ADAPTERS_DIR_NAME, "", U2DataPath::CutFileExtension);
        dpr->registerEntry(dp);
    }

    toolRunnerProgram = PythonSupport::ET_PYTHON_ID;
    dependencies << PythonSupport::ET_PYTHON_ID;
}

}    //namespace U2

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

#include "ExternalToolUtils.h"

#include <QFile>
#include <QFileInfo>
#include <QMessageBox>

#include <U2Core/AppContext.h>
#include <U2Core/DataPathRegistry.h>
#include <U2Core/ExternalToolRegistry.h>
#include <U2Core/QObjectScopedPointer.h>
#include <U2Core/Settings.h>
#include <U2Core/U2SafePoints.h>

#include <U2Gui/AppSettingsGUI.h>

#include "ExternalToolSupportSettingsController.h"

namespace U2 {

const QString ExternalToolUtils::CISTROME_DATA_DIR = "CISTROME_DATA_DIR";

void ExternalToolUtils::checkExtToolsPath(const QStringList &ids) {
    QStringList missingTools;
    foreach (const QString &id, ids) {
        ExternalTool *tool = AppContext::getExternalToolRegistry()->getById(id);
        SAFE_POINT(nullptr != tool, QString("External tool with ID '%1' not found in the registry").arg(id), );
        if (tool->getPath().isEmpty()) {
            missingTools << tool->getName();
        }
    }
    if (!missingTools.isEmpty()) {
        QString mergedNames = missingTools.join(", ");

        QObjectScopedPointer<QMessageBox> msgBox = new QMessageBox;
        msgBox->setWindowTitle("BLAST: " + QString(mergedNames));
        msgBox->setText(tr("Paths for the following tools are not selected: %1.").arg(mergedNames));
        msgBox->setInformativeText(tr("Do you want to select it now?"));
        msgBox->setStandardButtons(QMessageBox::Yes | QMessageBox::No);
        msgBox->setDefaultButton(QMessageBox::Yes);
        const int ret = msgBox->exec();
        CHECK(!msgBox.isNull(), );

        switch (ret) {
        case QMessageBox::Yes:
            AppContext::getAppSettingsGUI()->showSettingsDialog(ExternalToolSupportSettingsPageId);
            break;
        case QMessageBox::No:
            return;
        default:
            assert(false);
            break;
        }
    }
}

void ExternalToolUtils::addDefaultCistromeDirToSettings() {
    QString cistromeDefaultPath;
    QString customDataDir = qgetenv("UGENE_DATA_PATH");
    if (!customDataDir.isEmpty()) {
        cistromeDefaultPath = QFileInfo(customDataDir + "/cistrome").absoluteFilePath();
    } else {
        cistromeDefaultPath = QFileInfo(QString(PATH_PREFIX_DATA) + QString(":") + "cistrome").absoluteFilePath();
    }

    bool defaultExists = QFile::exists(cistromeDefaultPath);
    QString savedValue = AppContext::getSettings()->getValue(CISTROME_DATA_DIR).toString();

    bool addNew = savedValue.isEmpty() && defaultExists;
    bool removeOld = !savedValue.isEmpty() && !QFile::exists(savedValue);
    bool replaceOld = removeOld && defaultExists;

    if (addNew || replaceOld) {
        AppContext::getSettings()->setValue(CISTROME_DATA_DIR, cistromeDefaultPath);
    } else if (removeOld) {
        AppContext::getSettings()->remove(CISTROME_DATA_DIR);
    }
}

void ExternalToolUtils::addCistromeDataPath(const QString &dataName, const QString &dirName, bool entriesAreFolders) {
    U2DataPathRegistry *dpr = AppContext::getDataPathRegistry();
    CHECK(NULL != dpr, );

    const QString dataPath = AppContext::getSettings()->getValue(CISTROME_DATA_DIR).toString() + QDir::separator() + dirName;
    U2DataPath *dp = new U2DataPath(dataName, dataPath, "", U2DataPath::CutFileExtension | (entriesAreFolders ? U2DataPath::AddOnlyFolders : U2DataPath::None));
    bool ok = dpr->registerEntry(dp);
    if (!ok) {
        delete dp;
    }
}

}    // namespace U2

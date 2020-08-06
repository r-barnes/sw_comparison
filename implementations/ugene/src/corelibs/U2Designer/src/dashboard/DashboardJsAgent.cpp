/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2017 UniPro <ugene@unipro.ru>
 * http://ugene.unipro.ru
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

#include "DashboardJsAgent.h"

#include <QApplication>
#include <QClipboard>
#include <QDesktopServices>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>

#include <U2Core/AppContext.h>
#include <U2Core/AppSettings.h>
#include <U2Core/ProjectModel.h>
#include <U2Core/UserApplicationsSettings.h>

#include <U2Gui/MainWindow.h>

#include <U2Lang/Actor.h>
#include <U2Lang/Dataset.h>
#include <U2Lang/URLContainer.h>
#include <U2Lang/WorkflowSettings.h>
#include <U2Lang/WorkflowUtils.h>

#include "Dashboard.h"
#include "DashboardPageController.h"

namespace U2 {

const QString DashboardJsAgent::ID = "agent";

DashboardJsAgent::DashboardJsAgent(Dashboard *parent)
    : JavaScriptAgent(parent),
      monitor(parent->getMonitor()) {
    fillWorkerParamsInfo();
}

const QString &DashboardJsAgent::getId() const {
    return ID;
}

void DashboardJsAgent::sl_onJsError(const QString &errorMessage) {
    coreLog.error(errorMessage);
}

void DashboardJsAgent::openUrl(const QString &relative) {
    QString url = absolute(relative);
    QVariantMap hints;
    hints[ProjectLoaderHint_OpenBySystemIfFormatDetectionFailed] = true;
    Task *task = AppContext::getProjectLoader()->openWithProjectTask(url, hints);
    if (NULL != task) {
        AppContext::getTaskScheduler()->registerTopLevelTask(task);
    }
}

void DashboardJsAgent::openByOS(const QString &relative) {
    QString url = absolute(relative);
    if (!QFile::exists(url)) {
        QMessageBox::critical((QWidget *)AppContext::getMainWindow()->getQMainWindow(), tr("Error"), tr("The file does not exist"));
        return;
    }
    QDesktopServices::openUrl(QUrl("file:///" + url));
}

QString DashboardJsAgent::absolute(const QString &url) {
    if (QFileInfo(url).isAbsolute()) {
        return url;
    }
    return qobject_cast<Dashboard *>(parent())->directory() + url;
}

void DashboardJsAgent::loadSchema() {
    Dashboard *dashboard = qobject_cast<Dashboard *>(parent());
    dashboard->sl_loadSchema();
}

void DashboardJsAgent::setClipboardText(const QString &text) {
    QApplication::clipboard()->setText(text);
}

void DashboardJsAgent::hideLoadButtonHint() {
    Dashboard *dashboard = qobject_cast<Dashboard *>(parent());
    SAFE_POINT(NULL != dashboard, "NULL dashboard!", );
    dashboard->initiateHideLoadButtonHint();
}

QString DashboardJsAgent::getLogsFolderUrl() const {
    return monitor->getLogsDir();
}

QString DashboardJsAgent::getLogUrl(const QString &actorId, int actorRunNumber, const QString &toolName, int toolRunNumber, int contentType) const {
    return monitor->getLogUrl(actorId, actorRunNumber, toolName, toolRunNumber, contentType);
}

QString DashboardJsAgent::getWorkersParamsInfo() {
    return workersParamsInfo;
}

bool DashboardJsAgent::getShowHint() {
    return WorkflowSettings::isShowLoadButtonHint();
}

//Worker parameters initialization
void DashboardJsAgent::fillWorkerParamsInfo() {
    CHECK(monitor, );
    QJsonArray localWorkersParamsInfo;
    QList<Monitor::WorkerParamsInfo> workersParamsList = monitor->getWorkersParameters();
    foreach (Monitor::WorkerParamsInfo workerInfo, workersParamsList) {
        QJsonObject workerInfoJS;
        workerInfoJS["workerName"] = workerInfo.workerName;
        workerInfoJS["actor"] = workerInfo.actor->getLabel();
        QJsonArray parameters;
        foreach (Attribute *parameter, workerInfo.parameters) {
            QJsonObject parameterJS;
            parameterJS["name"] = parameter->getDisplayName();
            QVariant paramValueVariant = parameter->getAttributePureValue();
            if (paramValueVariant.canConvert<QList<Dataset>>()) {
                QList<Dataset> sets = paramValueVariant.value<QList<Dataset>>();
                foreach (const Dataset &set, sets) {
                    QString paramName = parameter->getDisplayName();
                    if (sets.size() > 1) {
                        paramName += ": <i>" + set.getName() + "</i>";
                    }
                    parameterJS["name"] = paramName;
                    QStringList urls;
                    foreach (URLContainer *c, set.getUrls()) {
                        urls << c->getUrl();
                    }
                    parameterJS["value"] = urls.join(";");
                    parameterJS["isDataset"] = true;
                    parameters.append(parameterJS);
                }
            } else {
                parameterJS["value"] = WorkflowUtils::getStringForParameterDisplayRole(paramValueVariant);
                UrlAttributeType type = WorkflowUtils::isUrlAttribute(parameter, workerInfo.actor);
                if (type == NotAnUrl || QString::compare(paramValueVariant.toString(), "default", Qt::CaseInsensitive) == 0) {
                    parameterJS["isUrl"] = false;
                } else {
                    parameterJS["isUrl"] = true;
                    if (type == InputDir || type == OutputDir) {
                        parameterJS["type"] = "Dir";
                    }
                }
                parameters.append(parameterJS);
            }
        }
        workerInfoJS["parameters"] = parameters;
        localWorkersParamsInfo.append(workerInfoJS);
        QJsonDocument doc(localWorkersParamsInfo);
        workersParamsInfo = doc.toJson(QJsonDocument::Compact);
    }
}

}    // namespace U2

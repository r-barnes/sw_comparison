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

#ifndef _U2_DASHBOARD_JS_AGENT_H_
#define _U2_DASHBOARD_JS_AGENT_H_

#include <QPointer>

#include <U2Lang/WorkflowMonitor.h>

#include "webview/JavaScriptAgent.h"

namespace U2 {

class Dashboard;

class DashboardJsAgent : public JavaScriptAgent {
    Q_OBJECT

    Q_PROPERTY(QString workersParamsInfo READ getWorkersParamsInfo CONSTANT)
    Q_PROPERTY(bool showHint READ getShowHint CONSTANT)

public:
    DashboardJsAgent(Dashboard *parent);

    const QString &getId() const;

    Q_INVOKABLE QString getWorkersParamsInfo();

public slots:
    void sl_onJsError(const QString &errorMessage);

    void openUrl(const QString &url);
    void openByOS(const QString &url);
    QString absolute(const QString &url);
    void loadSchema();
    void setClipboardText(const QString &text);
    void hideLoadButtonHint();
    QString getLogsFolderUrl() const;
    QString getLogUrl(const QString &actorId, int actorRunNumber, const QString &toolName, int toolRunNumber, int contentType) const;

signals:
    void si_progressChanged(int progress);
    void si_taskStateChanged(QString state);
    void si_newProblem(QString problem);    //workaround for Qt5.4 and Qt5.5, sould be simple QJsonObject. More info see https://bugreports.qt.io/browse/QTBUG-48198
    void si_workerStatsInfoChanged(QString info);    //workaround for Qt5.4 and Qt5.5, sould be simple QJsonObject. More info see https://bugreports.qt.io/browse/QTBUG-48198
    void si_workerStatsUpdate(QString workersStatisticsInfo);    //workaround for Qt5.4, sould be simple QJsonArray.
    void si_onLogChanged(QString logEntry);    //workaround for Qt5.4 and Qt5.5, sould be simple QJsonObject. More info see https://bugreports.qt.io/browse/QTBUG-48198
    void si_newOutputFile(QString fileInfo);    //workaround for Qt5.4 and Qt5.5, sould be simple QJsonObject. More info see https://bugreports.qt.io/browse/QTBUG-48198
    void si_switchTab(QString tabId);    //workaround for Qt5.4 and Qt5.5, sould be simple QJsonObject. More info see https://bugreports.qt.io/browse/QTBUG-48198
    void si_createOutputWidget();

private:
    Q_INVOKABLE bool getShowHint();
    void fillWorkerParamsInfo();

    QString workersParamsInfo;
    const QPointer<const WorkflowMonitor> monitor;

    static const QString ID;
};

}    // namespace U2

#endif    // _U2_DASHBOARD_JS_AGENT_H_

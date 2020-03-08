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

#ifndef _U2_DASHBOARD_PAGE_CONTROLLER_H_
#define _U2_DASHBOARD_PAGE_CONTROLLER_H_

#include <U2Gui/SimpleWebViewBasedWidgetController.h>

#include <U2Lang/WorkflowMonitor.h>

namespace U2 {

class Dashboard;
class DashboardJsAgent;
class WebViewController;

class DashboardPageController : public SimpleWebViewBasedWidgetController {
    Q_OBJECT
public:
    DashboardPageController(Dashboard *dashboard);

    DashboardJsAgent *getAgent();

private slots:
    void sl_pageIsAboutToBeInitialized();
    void sl_pageInitialized();
    void sl_progressChanged(int progress);
    void sl_taskStateChanged(Monitor::TaskState state);
    void sl_newNotification(const WorkflowNotification &notification, int count);
    void sl_workerInfoChanged(const QString &actorId, const Monitor::WorkerInfo &info);
    void sl_workerStatsUpdate();
    void sl_onLogChanged(const Monitor::LogEntry &entry);
    void sl_newOutputFile(const Monitor::FileInfo &info);

private:
    void initData();

    QString serializeNotification(const WorkflowNotification &problem, int count) const;
    QString serializeWorkerInfo(const QString &actorId, const Monitor::WorkerInfo &info) const;
    QString serializeWorkerStatistics(const QMap<QString, Monitor::WorkerInfo> &workersStatistics) const;
    QString serializeLogEntry(const Monitor::LogEntry &entry) const;
    QString serializeFileInfo(const Monitor::FileInfo &info) const;

    int progress;
    QString state;
    QStringList problems;
    QStringList workerInfos;
    QStringList workersStatisticsInfos;
    QStringList logEntries;
    QStringList fileInfos;

    QPointer<Dashboard> dashboard;
    const QPointer<const WorkflowMonitor> monitor;
};

}   // namespace U2

#endif // _U2_DASHBOARD_PAGE_CONTROLLER_H_

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

#ifndef _U2_GUI_TEST_THREAD_H_
#define _U2_GUI_TEST_THREAD_H_

#include <core/GUITest.h>

#include <QThread>

#include <U2Core/global.h>

namespace U2 {

class Logger;
typedef QList<HI::GUITest *> GUITests;

class U2TEST_EXPORT GUITestThread : public QThread {
    Q_OBJECT
public:
    GUITestThread(HI::GUITest *test, bool isCleanupNeeded = true);

    void run();

    HI::GUITest *getTest() {
        return test;
    }
    QString getTestResult() {
        return testResult;
    }

private slots:
    void sl_testTimeOut();

private:
    QString launchTest(const GUITests &tests);

    static GUITests preChecks();
    static GUITests postChecks();
    static GUITests postActions();
    void clearSandbox();
    static void removeDir(const QString &dirName);
    void saveScreenshot();
    void cleanup();
    void writeTestResult();

    HI::GUITest *test;
    bool isRunPostActionsAndCleanup;
    QString testResult;
};

}    // namespace U2

#endif    // _U2_GUI_TEST_THREAD_H_

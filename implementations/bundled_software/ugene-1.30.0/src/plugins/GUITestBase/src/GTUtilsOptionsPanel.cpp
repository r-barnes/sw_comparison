/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2018 UniPro <ugene@unipro.ru>
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

#include <U2Core/global.h>
#include "GTUtilsOptionsPanel.h"
#include <drivers/GTMouseDriver.h>
#include <drivers/GTKeyboardDriver.h>
#include "utils/GTKeyboardUtils.h"
#include <primitives/GTWidget.h>
#include <primitives/GTTreeWidget.h>
#include "GTUtilsTaskTreeView.h"
#include "utils/GTUtilsApp.h"
#include "utils/GTThread.h"
#include <U2Core/ProjectModel.h>
#include <U2Core/U2OpStatus.h>
#include <U2Gui/MainWindow.h>
#include <QApplication>
#include <QMainWindow>
#include <QTreeWidget>

namespace U2 {
using namespace HI;

#define GT_CLASS_NAME "GTUtilsOptionsPanel"


#define GT_METHOD_NAME "runFindPatternWithHotKey"
void GTUtilsOptionsPanel::runFindPatternWithHotKey( const QString& pattern, HI::GUITestOpStatus& os){
    GTKeyboardDriver::keyClick( 'f', Qt::ControlModifier);
    GTGlobals::sleep();

    QWidget *w = QApplication::focusWidget();
    GT_CHECK(w && w->objectName()=="textPattern", "Focus is not on FindPattern widget");

    GTKeyboardDriver::keySequence(pattern);
    GTGlobals::sleep(1000);
    GTKeyboardDriver::keyClick(Qt::Key_Enter);
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}

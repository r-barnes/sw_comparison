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

#include <base_dialogs/GTFileDialog.h>
#include <base_dialogs/MessageBoxFiller.h>
#include <drivers/GTKeyboardDriver.h>
#include <drivers/GTMouseDriver.h>
#include <primitives/GTAction.h>
#include <primitives/GTCheckBox.h>
#include <primitives/GTComboBox.h>
#include <primitives/GTDoubleSpinBox.h>
#include <primitives/GTGroupBox.h>
#include <primitives/GTLineEdit.h>
#include <primitives/GTMenu.h>
#include <primitives/GTScrollBar.h>
#include <primitives/GTSpinBox.h>
#include <primitives/GTTabWidget.h>
#include <primitives/GTTableView.h>
#include <primitives/GTToolbar.h>
#include <primitives/GTTreeWidget.h>
#include <primitives/GTWidget.h>
#include <primitives/PopupChooser.h>
#include <utils/GTThread.h>

#include <QApplication>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QGraphicsView>
#include <QGroupBox>
#include <QListWidget>
#include <QMainWindow>
#include <QMessageBox>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QTableView>
#include <QTableWidget>
#include <QTextEdit>
#include <QToolButton>
#include <QTreeWidget>

#include <U2Core/AppContext.h>
#include <U2Core/U2SafePoints.h>

#include <U2View/MSAEditor.h>

#include "../../workflow_designer/src/WorkflowViewItems.h"
#include "GTUtilsMdi.h"
#include "GTUtilsWorkflowDesigner.h"
#include "api/GTGraphicsItem.h"
#include "runnables/ugene/corelibs/U2Gui/AppSettingsDialogFiller.h"
#include "runnables/ugene/plugins/workflow_designer/DatasetNameEditDialogFiller.h"
#include "runnables/ugene/plugins/workflow_designer/StartupDialogFiller.h"
#include "runnables/ugene/plugins/workflow_designer/WorkflowMetadialogFiller.h"

namespace U2 {
using namespace HI;

const int GTUtilsWorkflowDesigner::verticalShift = 35;
#define GT_CLASS_NAME "GTUtilsWorkflowDesigner"

#define GT_METHOD_NAME "getActiveWorkflowDesignerWindow"
QWidget *GTUtilsWorkflowDesigner::getActiveWorkflowDesignerWindow(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = nullptr;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && wdWindow == nullptr; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        MainWindow *mainWindow = AppContext::getMainWindow();
        QWidget *mdiWindow = mainWindow == nullptr ? nullptr : mainWindow->getMDIManager()->getActiveWindow();
        if (mdiWindow != nullptr && mdiWindow->objectName() == "Workflow Designer") {
            wdWindow = mdiWindow;
        }
    }
    GT_CHECK_RESULT(wdWindow != nullptr, "No active WD window!", nullptr);
    GTThread::waitForMainThread();
    return wdWindow;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "checkWorkflowDesignerWindowIsActive"
void GTUtilsWorkflowDesigner::checkWorkflowDesignerWindowIsActive(HI::GUITestOpStatus &os) {
    getActiveWorkflowDesignerWindow(os);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "openWorkflowDesigner"
void GTUtilsWorkflowDesigner::openWorkflowDesigner(HI::GUITestOpStatus &os) {
    StartupDialogFiller *filler = new StartupDialogFiller(os);
    GTUtilsDialog::waitForDialogWhichMayRunOrNot(os, filler);
    GTMenu::clickMainMenuItem(os, QStringList() << "Tools"
                                                << "Workflow Designer...");
    checkWorkflowDesignerWindowIsActive(os);
    GTUtilsDialog::removeRunnable(filler);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "currentTab"
GTUtilsWorkflowDesigner::tab GTUtilsWorkflowDesigner::currentTab(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTabWidget *tabs = qobject_cast<QTabWidget *>(GTWidget::findWidget(os, "tabs", wdWindow));
    GT_CHECK_RESULT(NULL != tabs, "tabs widget is not found", algorithms);
    return tab(tabs->currentIndex());
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setCurrentTab"
void GTUtilsWorkflowDesigner::setCurrentTab(HI::GUITestOpStatus &os, tab t) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTabWidget *tabs = qobject_cast<QTabWidget *>(GTWidget::findWidget(os, "tabs", wdWindow));
    GT_CHECK(NULL != tabs, "tabs widget is not found");
    GTTabWidget::setCurrentIndex(os, tabs, int(t));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "loadWorkflow"
void GTUtilsWorkflowDesigner::loadWorkflow(HI::GUITestOpStatus &os, const QString &fileUrl) {
    GTUtilsDialog::waitForDialog(os, new GTFileDialogUtils(os, fileUrl));
    QToolBar *wdToolbar = GTToolbar::getToolbar(os, "mwtoolbar_activemdi");
    GT_CHECK(wdToolbar, "Toolbar is not found");
    QWidget *loadButton = GTToolbar::getWidgetForActionName(os, wdToolbar, "Load workflow");
    GT_CHECK(loadButton, "Load button is not found");
    GTWidget::click(os, loadButton);
    GTGlobals::sleep();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "saveWorkflow"
void GTUtilsWorkflowDesigner::saveWorkflow(HI::GUITestOpStatus &os) {
    GTToolbar::clickButtonByTooltipOnToolbar(os, MWTOOLBAR_ACTIVEMDI, "Save workflow");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "saveWorkflowAs"
void GTUtilsWorkflowDesigner::saveWorkflowAs(HI::GUITestOpStatus &os, const QString &fileUrl, const QString &workflowName) {
    GTUtilsDialog::waitForDialog(os, new WorkflowMetaDialogFiller(os, fileUrl, workflowName));
    GTToolbar::clickButtonByTooltipOnToolbar(os, MWTOOLBAR_ACTIVEMDI, "Save workflow as");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "validateWorkflow"
void GTUtilsWorkflowDesigner::validateWorkflow(GUITestOpStatus &os) {
    GTWidget::click(os, GTAction::button(os, "Validate workflow"));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "runWorkflow"
void GTUtilsWorkflowDesigner::runWorkflow(HI::GUITestOpStatus &os) {
    GTWidget::click(os, GTAction::button(os, "Run workflow", GTUtilsMdi::activeWindow(os)));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "stopWorkflow"
void GTUtilsWorkflowDesigner::stopWorkflow(HI::GUITestOpStatus &os) {
    GTWidget::click(os, GTAction::button(os, "Stop workflow", GTUtilsMdi::activeWindow(os)));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "returnToWorkflow"
void GTUtilsWorkflowDesigner::returnToWorkflow(HI::GUITestOpStatus &os) {
    GTWidget::click(os, GTAction::button(os, GTAction::findActionByText(os, "To Workflow Designer")));
}
#undef GT_METHOD_NAME

namespace {
bool compare(QString s1, QString s2, bool exactMatch) {
    if (exactMatch) {
        return s1 == s2;
    } else {
        return s1.toLower().contains(s2.toLower());
    }
}
}    // namespace

#define GT_METHOD_NAME "findTreeItem"
QTreeWidgetItem *GTUtilsWorkflowDesigner::findTreeItem(HI::GUITestOpStatus &os, QString itemName, tab t, bool exactMatch, bool failIfNULL) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTreeWidgetItem *foundItem = NULL;
    QTreeWidget *w;
    if (t == algorithms) {
        w = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "WorkflowPaletteElements", wdWindow));
    } else {
        w = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "samples", wdWindow));
    }
    GT_CHECK_RESULT(w != NULL, "WorkflowPaletteElements is null", NULL);

    QList<QTreeWidgetItem *> outerList = w->findItems("", Qt::MatchContains);

    for (int i = 0; i < outerList.count(); i++) {
        QList<QTreeWidgetItem *> innerList;

        for (int j = 0; j < outerList.value(i)->childCount(); j++) {
            innerList.append(outerList.value(i)->child(j));
        }

        foreach (QTreeWidgetItem *item, innerList) {
            if (t == algorithms) {
                QString s = item->data(0, Qt::UserRole).value<QAction *>()->text();
                if (compare(s, itemName, exactMatch)) {
                    GT_CHECK_RESULT(foundItem == NULL, "several items have this discription", item);
                    foundItem = item;
                }
            } else {
                QString s = item->text(0);
                if (compare(s, itemName, exactMatch)) {
                    GT_CHECK_RESULT(foundItem == NULL, "several items have this discription", item);
                    foundItem = item;
                }
            }
        }
    }
    if (failIfNULL) {
        GT_CHECK_RESULT(foundItem != NULL, "Item \"" + itemName + "\" not found in treeWidget", NULL);
    }
    return foundItem;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getVisibleSamples"
QList<QTreeWidgetItem *> GTUtilsWorkflowDesigner::getVisibleSamples(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTreeWidget *w = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "samples", wdWindow));
    GT_CHECK_RESULT(w != NULL, "WorkflowPaletteElements is null", QList<QTreeWidgetItem *>());

    QList<QTreeWidgetItem *> outerList = w->findItems("", Qt::MatchContains);
    QList<QTreeWidgetItem *> resultList;
    for (int i = 0; i < outerList.count(); i++) {
        QList<QTreeWidgetItem *> innerList;

        for (int j = 0; j < outerList.value(i)->childCount(); j++) {
            innerList.append(outerList.value(i)->child(j));
        }

        foreach (QTreeWidgetItem *item, innerList) {
            if (!item->isHidden()) {
                resultList.append(item);
            }
        }
    }
    return resultList;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "addAlgorithm"
void GTUtilsWorkflowDesigner::addAlgorithm(HI::GUITestOpStatus &os, QString algName, bool exactMatch, bool useDragAndDrop) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    expandTabs(os);
    QTabWidget *tabs = qobject_cast<QTabWidget *>(GTWidget::findWidget(os, "tabs", wdWindow));
    GT_CHECK(tabs != nullptr, "tabs widget not found");

    GTTabWidget::setCurrentIndex(os, tabs, 0);

    QTreeWidgetItem *alg = findTreeItem(os, algName, algorithms, exactMatch);
    GT_CHECK(alg != nullptr, "algorithm is NULL");

    selectAlgorithm(os, alg);
    QWidget *w = GTWidget::findWidget(os, "sceneView", wdWindow);

    // Put the new worker in to the grid.
    int columnWidth = 250;
    int columnHeight = 250;
    int workersPerRow = 3;

    int numberOfWorkers = getWorkers(os).size();
    int currentWorkerRow = numberOfWorkers / workersPerRow;
    int currentWorkerColumn = numberOfWorkers % workersPerRow;
    QPoint newWorkerPosition(w->rect().topLeft() + QPoint(currentWorkerColumn * columnWidth, currentWorkerRow * columnHeight) + QPoint(100, 100));
    if (useDragAndDrop) {
        GTMouseDriver::dragAndDrop(GTMouseDriver::getMousePosition(), w->mapToGlobal(newWorkerPosition));
    } else {
        GTWidget::click(os, w, Qt::LeftButton, newWorkerPosition);
    }
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "addElement"
WorkflowProcessItem *GTUtilsWorkflowDesigner::addElement(HI::GUITestOpStatus &os, const QString &algName, bool exactMatch) {
    addAlgorithm(os, algName, exactMatch);
    CHECK_OP(os, NULL);
    return getWorker(os, algName);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "addElementByUsingNameFilter"
WorkflowProcessItem *GTUtilsWorkflowDesigner::addElementByUsingNameFilter(HI::GUITestOpStatus &os, const QString &elementName, bool exactMatch) {
    GTUtilsWorkflowDesigner::findByNameFilter(os, elementName);
    WorkflowProcessItem *item = GTUtilsWorkflowDesigner::addElement(os, elementName, exactMatch);
    GTUtilsWorkflowDesigner::cleanNameFilter(os);
    return item;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "selectAlgorithm"
void GTUtilsWorkflowDesigner::selectAlgorithm(HI::GUITestOpStatus &os, QTreeWidgetItem *algorithm) {
    GT_CHECK(algorithm != nullptr, "algorithm is nullptr");

    class MainThreadAction : public CustomScenario {
    public:
        MainThreadAction(QTreeWidgetItem *algorithm)
            : CustomScenario(), algorithm(algorithm) {
        }
        void run(HI::GUITestOpStatus &os) {
            Q_UNUSED(os);
            algorithm->treeWidget()->scrollToItem(algorithm, QAbstractItemView::PositionAtCenter);
        }
        QTreeWidgetItem *algorithm;
    };
    GTThread::runInMainThread(os, new MainThreadAction(algorithm));
    GTMouseDriver::moveTo(GTTreeWidget::getItemCenter(os, algorithm));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "addSample"
void GTUtilsWorkflowDesigner::addSample(HI::GUITestOpStatus &os, const QString &sampName, QWidget const *const parentWidget) {
    expandTabs(os, parentWidget);
    QTabWidget *tabs = qobject_cast<QTabWidget *>(GTWidget::findWidget(os, "tabs", parentWidget));
    GT_CHECK(tabs != NULL, "tabs widget not found");

    GTTabWidget::setCurrentIndex(os, tabs, 1);

    QTreeWidgetItem *samp = findTreeItem(os, sampName, samples);
    GTGlobals::sleep(100);
    GT_CHECK(samp != NULL, "sample is NULL");

    selectSample(os, samp, parentWidget);
    GTGlobals::sleep(500);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "selectSample"
void GTUtilsWorkflowDesigner::selectSample(HI::GUITestOpStatus &os, QTreeWidgetItem *sample, QWidget const *const parentWidget) {
    GT_CHECK(sample != nullptr, "sample is nullptr");
    QTreeWidget *paletteTree = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "samples", parentWidget));
    GT_CHECK(paletteTree != nullptr, "paletteTree is nullptr");

    class MainThreadAction : public CustomScenario {
    public:
        MainThreadAction(QTreeWidget *paletteTree, QTreeWidgetItem *sample)
            : CustomScenario(), paletteTree(paletteTree), sample(sample) {
        }
        void run(HI::GUITestOpStatus &os) {
            Q_UNUSED(os);
            paletteTree->scrollToItem(sample);
        }
        QTreeWidget *paletteTree;
        QTreeWidgetItem *sample;
    };
    GTThread::runInMainThread(os, new MainThreadAction(paletteTree, sample));

    GTMouseDriver::moveTo(GTTreeWidget::getItemCenter(os, sample));
    GTMouseDriver::doubleClick();
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "expandTabs"
void GTUtilsWorkflowDesigner::expandTabs(HI::GUITestOpStatus &os, QWidget const *const parentWidget) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QSplitter *splitter = qobject_cast<QSplitter *>(GTWidget::findWidget(os, "WorkflowViewMainSplitter", parentWidget == nullptr ? wdWindow : parentWidget));
    GT_CHECK(splitter, "splitter not found");
    QList<int> s;
    s = splitter->sizes();

    if (s.first() == 0) {    //expands tabs if collapsed
        QPoint p;
        p.setX(splitter->geometry().left() + 2);
        p.setY(splitter->geometry().center().y());
        GTMouseDriver::moveTo(p);
        GTGlobals::sleep(300);
        GTMouseDriver::press();
        p.setX(p.x() + 200);
        GTMouseDriver::moveTo(p);
        GTMouseDriver::release();
        GTThread::waitForMainThread();
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "findByNameFilter"
void GTUtilsWorkflowDesigner::findByNameFilter(HI::GUITestOpStatus &os, const QString &elementName) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QWidget *paletteWidget = GTWidget::findWidget(os, "palette", wdWindow);
    QLineEdit *nameFilterLineEdit = qobject_cast<QLineEdit *>(GTWidget::findWidget(os, "nameFilterLineEdit", paletteWidget));
    GT_CHECK(nameFilterLineEdit != NULL, "Filter name line edit is not found");

    const QPoint mappedLineEditPos = nameFilterLineEdit->mapToGlobal(nameFilterLineEdit->pos());
    const QPoint pos(mappedLineEditPos.x() + 75, mappedLineEditPos.y() + 10);
    GTMouseDriver::moveTo(pos);
    GTGlobals::sleep(500);
    GTMouseDriver::click();
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_Home);
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_End, Qt::ShiftModifier);
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_Backspace);
    GTGlobals::sleep(500);
    for (int i = 0; i < elementName.size(); i++) {
        GTKeyboardDriver::keyClick(elementName[i].toLatin1());
        GTGlobals::sleep(50);
    }
    GTGlobals::sleep(1000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "cleanNameFilter"
void GTUtilsWorkflowDesigner::cleanNameFilter(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QWidget *paletteWidget = GTWidget::findWidget(os, "palette", wdWindow);
    QLineEdit *nameFilterLineEdit = qobject_cast<QLineEdit *>(GTWidget::findWidget(os, "nameFilterLineEdit", paletteWidget));
    GT_CHECK(nameFilterLineEdit != NULL, "Filter name line edit is not found");

    const QPoint mappedLineEditPos = nameFilterLineEdit->mapToGlobal(nameFilterLineEdit->pos());
    const QPoint pos(mappedLineEditPos.x() + 75, mappedLineEditPos.y() + 10);
    GTMouseDriver::moveTo(pos);
    GTGlobals::sleep(500);
    GTMouseDriver::click();
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_Home);
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_End, Qt::ShiftModifier);
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_Backspace);
    GTGlobals::sleep(1000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickOnPalette"
void GTUtilsWorkflowDesigner::clickOnPalette(HI::GUITestOpStatus &os, const QString &itemName, Qt::MouseButton mouseButton) {
    selectAlgorithm(os, findTreeItem(os, itemName, algorithms, true));
    GTMouseDriver::click(mouseButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroup"
QTreeWidgetItem *GTUtilsWorkflowDesigner::getPaletteGroup(HI::GUITestOpStatus &os, const QString &groupName) {
    QTreeWidget *tree = getCurrentTabTreeWidget(os);
    GT_CHECK_RESULT(NULL != tree, "WorkflowPaletteElements is NULL", NULL);

    GTGlobals::FindOptions options;
    options.depth = 1;
    options.matchPolicy = Qt::MatchExactly;

    return GTTreeWidget::findItem(os, tree, groupName, NULL, 0, options);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroups"
QList<QTreeWidgetItem *> GTUtilsWorkflowDesigner::getPaletteGroups(HI::GUITestOpStatus &os) {
    QList<QTreeWidgetItem *> groupItems;

    QTreeWidget *tree = getCurrentTabTreeWidget(os);
    GT_CHECK_RESULT(NULL != tree, "WorkflowPaletteElements is NULL", groupItems);

    GTGlobals::FindOptions options;
    options.depth = 1;
    options.matchPolicy = Qt::MatchContains;

    return GTTreeWidget::findItems(os, tree, "", NULL, 0, options);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroupNames"
QStringList GTUtilsWorkflowDesigner::getPaletteGroupNames(HI::GUITestOpStatus &os) {
    QStringList groupNames;
    const QList<QTreeWidgetItem *> groupItems = getPaletteGroups(os);
    foreach (QTreeWidgetItem *groupItem, groupItems) {
        groupNames << groupItem->text(0);
    }
    return groupNames;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroupEntries"
QList<QTreeWidgetItem *> GTUtilsWorkflowDesigner::getPaletteGroupEntries(HI::GUITestOpStatus &os, QTreeWidgetItem *groupItem) {
    QList<QTreeWidgetItem *> items;

    GT_CHECK_RESULT(NULL != groupItem, "Group item is NULL", items);

    QTreeWidget *tree = getCurrentTabTreeWidget(os);
    GT_CHECK_RESULT(NULL != tree, "WorkflowPaletteElements is NULL", items);

    GTGlobals::FindOptions options;
    options.depth = 0;
    options.matchPolicy = Qt::MatchContains;

    return GTTreeWidget::findItems(os, tree, "", groupItem, 0, options);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroupEntries"
QList<QTreeWidgetItem *> GTUtilsWorkflowDesigner::getPaletteGroupEntries(HI::GUITestOpStatus &os, const QString &groupName) {
    return getPaletteGroupEntries(os, getPaletteGroup(os, groupName));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPaletteGroupEntriesNames"
QStringList GTUtilsWorkflowDesigner::getPaletteGroupEntriesNames(GUITestOpStatus &os, const QString &groupName) {
    QStringList entriesNames;
    foreach (QTreeWidgetItem *entryItem, getPaletteGroupEntries(os, groupName)) {
        entriesNames << entryItem->text(0);
    }
    return entriesNames;
}
#undef GT_METHOD_NAME

QPoint GTUtilsWorkflowDesigner::getItemCenter(HI::GUITestOpStatus &os, QString itemName) {
    QRect r = getItemRect(os, itemName);
    QPoint p = r.center();
    return p;
}

#define GT_METHOD_NAME "removeItem"
void GTUtilsWorkflowDesigner::removeItem(HI::GUITestOpStatus &os, QString itemName) {
    click(os, itemName);
    GTKeyboardDriver::keyClick(Qt::Key_Delete);
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

int GTUtilsWorkflowDesigner::getItemLeft(HI::GUITestOpStatus &os, QString itemName) {
    QRect r = getItemRect(os, itemName);
    int i = r.left();
    return i;
}

int GTUtilsWorkflowDesigner::getItemRight(HI::GUITestOpStatus &os, QString itemName) {
    QRect r = getItemRect(os, itemName);
    int i = r.right();
    return i;
}

int GTUtilsWorkflowDesigner::getItemTop(HI::GUITestOpStatus &os, QString itemName) {
    QRect r = getItemRect(os, itemName);
    int i = r.top();
    return i;
}

int GTUtilsWorkflowDesigner::getItemBottom(HI::GUITestOpStatus &os, QString itemName) {
    QRect r = getItemRect(os, itemName);
    int i = r.bottom();
    return i;
}
#define GT_METHOD_NAME "click"
void GTUtilsWorkflowDesigner::click(HI::GUITestOpStatus &os, QString itemName, QPoint p, Qt::MouseButton button) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    GT_CHECK(sceneView != NULL, "scene view is NULL");
    sceneView->ensureVisible(getWorker(os, itemName));
    GTThread::waitForMainThread();

    GTMouseDriver::moveTo(getItemCenter(os, itemName) + p);
    GTMouseDriver::click();
    if (Qt::RightButton == button) {
        GTMouseDriver::click(Qt::RightButton);
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "click"
void GTUtilsWorkflowDesigner::click(HI::GUITestOpStatus &os, QGraphicsItem *item, QPoint p, Qt::MouseButton button) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    GT_CHECK(sceneView != NULL, "scene view is NULL");
    sceneView->ensureVisible(item);
    QRect rect = GTGraphicsItem::getGraphicsItemRect(os, item);

    GTMouseDriver::moveTo(rect.center() + p);
    GTMouseDriver::click();
    if (Qt::RightButton == button) {
        GTMouseDriver::click(Qt::RightButton);
    }
    GTGlobals::sleep(200);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getWorker"
WorkflowProcessItem *GTUtilsWorkflowDesigner::getWorker(HI::GUITestOpStatus &os, QString itemName, const GTGlobals::FindOptions &options) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    GT_CHECK_RESULT(sceneView, "sceneView not found", nullptr);
    // Wait for the item up to GT_OP_WAIT_MILLIS.
    for (int time = 0; time < GT_OP_WAIT_MILLIS; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        QList<QGraphicsItem *> items = sceneView->items();
        foreach (QGraphicsItem *item, items) {
            QGraphicsObject *graphicsObject = item->toGraphicsObject();
            QGraphicsTextItem *graphicsTextItem = qobject_cast<QGraphicsTextItem *>(graphicsObject);
            if (graphicsTextItem != nullptr) {
                QString text = graphicsTextItem->toPlainText();
                int lineSeparatorIndex = text.indexOf('\n');
                if (lineSeparatorIndex == -1) {
                    continue;
                }
                text = text.left(lineSeparatorIndex);
                if (text == itemName) {
                    WorkflowProcessItem *result = qgraphicsitem_cast<WorkflowProcessItem *>(item->parentItem()->parentItem());
                    if (result != nullptr) {
                        return result;
                    }
                    break;
                }
            }
        }
        if (!options.failIfNotFound) {
            break;
        }
    }
    GT_CHECK_RESULT(!options.failIfNotFound, "Item '" + itemName + "' is not found", NULL);
    return nullptr;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getWorkerText"
QString GTUtilsWorkflowDesigner::getWorkerText(HI::GUITestOpStatus &os, QString itemName, const GTGlobals::FindOptions &options) {
    WorkflowProcessItem *worker = getWorker(os, itemName, options);
    foreach (QGraphicsItem *child, worker->childItems()) {
        foreach (QGraphicsItem *subchild, child->childItems()) {
            QGraphicsObject *graphObject = subchild->toGraphicsObject();
            QGraphicsTextItem *textItem = qobject_cast<QGraphicsTextItem *>(graphObject);
            if (NULL != textItem) {
                return textItem->toPlainText();
            }
        }
    }
    return QString();
}
#undef GT_METHOD_NAME

void GTUtilsWorkflowDesigner::clickLink(HI::GUITestOpStatus &os, QString itemName, Qt::MouseButton button, int step) {
    WorkflowProcessItem *worker = getWorker(os, itemName);

    int left = GTUtilsWorkflowDesigner::getItemLeft(os, itemName);
    int right = GTUtilsWorkflowDesigner::getItemRight(os, itemName);
    int top = GTUtilsWorkflowDesigner::getItemTop(os, itemName);
    int bottom = GTUtilsWorkflowDesigner::getItemBottom(os, itemName);
    for (int i = left; i < right; i += step) {
        for (int j = top; j < bottom; j += step) {
            GTMouseDriver::moveTo(QPoint(i, j));
            if (worker->cursor().shape() == Qt::PointingHandCursor) {
                GTMouseDriver::click(button);
                return;
            }
        }
    }
}

#define GT_METHOD_NAME "isWorkerExtended"
bool GTUtilsWorkflowDesigner::isWorkerExtended(HI::GUITestOpStatus &os, const QString &itemName) {
    return "ext" == getWorker(os, itemName)->getStyle();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPortById"
WorkflowPortItem *GTUtilsWorkflowDesigner::getPortById(HI::GUITestOpStatus &os, WorkflowProcessItem *worker, QString id) {
    QList<WorkflowPortItem *> list = getPorts(os, worker);
    foreach (WorkflowPortItem *p, list) {
        if (p && p->getPort()->getId() == id) {
            return p;
        }
    }
    GT_CHECK_RESULT(false, "port with id " + id + "not found", NULL);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getPorts"
QList<WorkflowPortItem *> GTUtilsWorkflowDesigner::getPorts(HI::GUITestOpStatus &os, WorkflowProcessItem *worker) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    GT_CHECK_RESULT(sceneView, "sceneView not found", QList<WorkflowPortItem *>())
    return worker->getPortItems();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getItemRect"
QRect GTUtilsWorkflowDesigner::getItemRect(HI::GUITestOpStatus &os, QString itemName) {
    //TODO: support finding items when there are several similar workers in scheme
    WorkflowProcessItem *w = getWorker(os, itemName);
    QRect result = GTGraphicsItem::getGraphicsItemRect(os, w);
    result.setTop(result.top() + verticalShift);
    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCurrentTabTreeWidget"
QTreeWidget *GTUtilsWorkflowDesigner::getCurrentTabTreeWidget(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    switch (currentTab(os)) {
    case algorithms:
        return GTWidget::findExactWidget<QTreeWidget *>(os, "WorkflowPaletteElements", wdWindow);
    case samples:
        return GTWidget::findExactWidget<QTreeWidget *>(os, "samples", wdWindow);
    default:
        os.setError("An unexpected current tab");
        return nullptr;
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "toggleDebugMode"
void GTUtilsWorkflowDesigner::toggleDebugMode(HI::GUITestOpStatus &os, bool enable) {
    class DebugModeToggleScenario : public CustomScenario {
    public:
        DebugModeToggleScenario(bool enable)
            : enable(enable) {
        }

        void run(HI::GUITestOpStatus &os) {
            QWidget *dialog = QApplication::activeModalWidget();
            GT_CHECK(dialog, "activeModalWidget is NULL");

            GTTreeWidget::click(os, GTTreeWidget::findItem(os, GTWidget::findExactWidget<QTreeWidget *>(os, "tree"), "  Workflow Designer"));
            GTCheckBox::setChecked(os, GTWidget::findExactWidget<QCheckBox *>(os, "debuggerBox"), enable);

            GTUtilsDialog::clickButtonBox(os, dialog, QDialogButtonBox::Ok);
        }

    private:
        bool enable;
    };

    GTUtilsDialog::waitForDialog(os, new AppSettingsDialogFiller(os, new DebugModeToggleScenario(enable)));
    GTMenu::clickMainMenuItem(os, QStringList() << "Settings"
                                                << "Preferences...");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "toggleBreakpointManager"
void GTUtilsWorkflowDesigner::toggleBreakpointManager(HI::GUITestOpStatus &os) {
    GTWidget::click(os, GTToolbar::getWidgetForActionTooltip(os, GTToolbar::getToolbar(os, MWTOOLBAR_ACTIVEMDI), "Show or hide breakpoint manager"));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setBreakpoint"
void GTUtilsWorkflowDesigner::setBreakpoint(HI::GUITestOpStatus &os, const QString &itemName) {
    click(os, itemName);
    GTWidget::click(os, GTToolbar::getWidgetForActionTooltip(os, GTToolbar::getToolbar(os, MWTOOLBAR_ACTIVEMDI), "Break at element"));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getBreakpointList"
QStringList GTUtilsWorkflowDesigner::getBreakpointList(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    return GTTreeWidget::getItemNames(os, GTWidget::findExactWidget<QTreeWidget *>(os, "breakpoints list", wdWindow));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getAllConnectionArrows"
QList<WorkflowBusItem *> GTUtilsWorkflowDesigner::getAllConnectionArrows(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    GT_CHECK_RESULT(sceneView, "sceneView not found", QList<WorkflowBusItem *>());

    QList<WorkflowBusItem *> result;

    foreach (QGraphicsItem *item, sceneView->items()) {
        WorkflowBusItem *arrow = qgraphicsitem_cast<WorkflowBusItem *>(item);
        if (arrow != NULL) {
            result.append(arrow);
        }
    };

    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "removeCmdlineWorkerFromPalette"
void GTUtilsWorkflowDesigner::removeCmdlineWorkerFromPalette(HI::GUITestOpStatus &os, const QString &workerName) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTabWidget *tabs = qobject_cast<QTabWidget *>(GTWidget::findWidget(os, "tabs", wdWindow));
    GT_CHECK(tabs != NULL, "tabs widget not found");

    GTTabWidget::setCurrentIndex(os, tabs, 0);

    QTreeWidget *w = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "WorkflowPaletteElements", wdWindow));
    GT_CHECK(w != NULL, "WorkflowPaletteElements is null");

    QTreeWidgetItem *foundItem = NULL;
    QList<QTreeWidgetItem *> outerList = w->findItems("", Qt::MatchContains);
    for (int i = 0; i < outerList.count(); i++) {
        QList<QTreeWidgetItem *> innerList;

        for (int j = 0; j < outerList.value(i)->childCount(); j++) {
            innerList.append(outerList.value(i)->child(j));
        }

        foreach (QTreeWidgetItem *item, innerList) {
            const QString s = item->data(0, Qt::UserRole).value<QAction *>()->text();
            if (s == workerName) {
                foundItem = item;
            }
        }
    }
    if (foundItem != NULL) {
        GTUtilsDialog::waitForDialog(os, new PopupChooserByText(os, QStringList() << "Remove"));
        GTUtilsDialog::waitForDialog(os, new MessageBoxDialogFiller(os, QMessageBox::Ok, "", "Remove element"));
        GTUtilsWorkflowDesigner::clickOnPalette(os, workerName, Qt::RightButton);
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "increaseOutputPortBoxHeight"
void GTUtilsWorkflowDesigner::changeInputPortBoxHeight(HI::GUITestOpStatus &os, const int offset) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTextEdit *doc = GTWidget::findExactWidget<QTextEdit *>(os, "doc", wdWindow);
    GT_CHECK(doc != NULL, "doc is not found");

    QGroupBox *paramBox = GTWidget::findExactWidget<QGroupBox *>(os, "paramBox", wdWindow);
    GT_CHECK(paramBox != NULL, "Param Box is not found");

    QGroupBox *inputPortBox = GTWidget::findExactWidget<QGroupBox *>(os, "inputPortBox", wdWindow);
    GT_CHECK(paramBox != NULL, "inputPortBox is not found");

    QPoint docGlobal = doc->mapToGlobal(doc->pos());
    QPoint bottomDevidePos(docGlobal.x() + (inputPortBox->width() / 2), docGlobal.y() + doc->height() + paramBox->height() + inputPortBox->height() + 10);
    QPoint newBottomDevidePos(bottomDevidePos.x(), bottomDevidePos.y() + offset);
    GTMouseDriver::dragAndDrop(bottomDevidePos, newBottomDevidePos);
    GTGlobals::sleep();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "importCmdlineBasedElement"
void GTUtilsWorkflowDesigner::importCmdlineBasedElement(GUITestOpStatus &os, const QString &path) {
    GTUtilsDialog::waitForDialog(os, new GTFileDialogUtils(os, path));
    GTToolbar::clickButtonByTooltipOnToolbar(os, MWTOOLBAR_ACTIVEMDI, "Add element with external tool");
    GTGlobals::sleep(500);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "connect"
void GTUtilsWorkflowDesigner::connect(HI::GUITestOpStatus &os, WorkflowProcessItem *from, WorkflowProcessItem *to) {
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(from->scene()->views().at(0));
    GT_CHECK(sceneView, "sceneView not found")
    QList<WorkflowPortItem *> fromList = from->getPortItems();
    QList<WorkflowPortItem *> toList = to->getPortItems();

    foreach (WorkflowPortItem *fromPort, fromList) {
        foreach (WorkflowPortItem *toPort, toList) {
            if (fromPort->getPort()->canBind(toPort->getPort())) {
                GTMouseDriver::moveTo(GTGraphicsItem::getItemCenter(os, fromPort));
                GTMouseDriver::press();
                GTMouseDriver::moveTo(GTGraphicsItem::getItemCenter(os, toPort));
                GTMouseDriver::release();
                GTGlobals::sleep(1000);
                return;
            }
        }
    }

    GT_CHECK(false, "no suitable ports to connect");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "disconnect"
void GTUtilsWorkflowDesigner::disconect(HI::GUITestOpStatus &os, WorkflowProcessItem *from, WorkflowProcessItem *to) {
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(from->scene()->views().at(0));
    GT_CHECK(sceneView, "sceneView not found");

    WorkflowBusItem *arrow = getConnectionArrow(os, from, to);
    QGraphicsTextItem *hint = getArrowHint(os, arrow);
    click(os, hint);

    GTKeyboardDriver::keyClick(Qt::Key_Delete);
    GTGlobals::sleep(1000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getConnectionArrow"
WorkflowBusItem *GTUtilsWorkflowDesigner::getConnectionArrow(HI::GUITestOpStatus &os, WorkflowProcessItem *from, WorkflowProcessItem *to) {
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(from->scene()->views().at(0));
    GT_CHECK_RESULT(sceneView, "sceneView not found", NULL)
    QList<WorkflowPortItem *> fromList = from->getPortItems();
    QList<WorkflowPortItem *> toList = to->getPortItems();

    QList<WorkflowBusItem *> arrows = getAllConnectionArrows(os);

    foreach (WorkflowPortItem *fromPort, fromList) {
        foreach (WorkflowPortItem *toPort, toList) {
            foreach (WorkflowBusItem *arrow, arrows) {
                if (arrow->getInPort() == toPort && arrow->getOutPort() == fromPort) {
                    return arrow;
                }
            }
        }
    }

    GT_CHECK_RESULT(false, "no suitable ports to connect", NULL);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getArrowHint"
QGraphicsTextItem *GTUtilsWorkflowDesigner::getArrowHint(HI::GUITestOpStatus &os, WorkflowBusItem *arrow) {
    GT_CHECK_RESULT(arrow != NULL, "arrow item is NULL", NULL);

    foreach (QGraphicsItem *item, arrow->childItems()) {
        QGraphicsTextItem *hint = qgraphicsitem_cast<QGraphicsTextItem *>(item);
        if (hint != NULL) {
            return hint;
        }
    }

    GT_CHECK_RESULT(false, "hint not found", NULL);
}
#undef GT_METHOD_NAME

QList<WorkflowProcessItem *> GTUtilsWorkflowDesigner::getWorkers(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QList<WorkflowProcessItem *> result;
    QGraphicsView *sceneView = qobject_cast<QGraphicsView *>(GTWidget::findWidget(os, "sceneView", wdWindow));
    QList<QGraphicsItem *> items = sceneView->items();
    foreach (QGraphicsItem *it, items) {
        WorkflowProcessItem *worker = qgraphicsitem_cast<WorkflowProcessItem *>(it);
        if (worker) {
            result.append(worker);
        }
    }
    return result;
}

#define GT_METHOD_NAME "getDatasetsListWidget"
QWidget *GTUtilsWorkflowDesigner::getDatasetsListWidget(GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    return GTWidget::findWidget(os, "DatasetsListWidget", wdWindow);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCurrentDatasetWidget"
QWidget *GTUtilsWorkflowDesigner::getCurrentDatasetWidget(GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTabWidget *datasetsTabWidget = GTWidget::findExactWidget<QTabWidget *>(os, "DatasetsTabWidget", wdWindow);
    return datasetsTabWidget->currentWidget();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setDatasetInputFile"
void GTUtilsWorkflowDesigner::setDatasetInputFile(GUITestOpStatus &os, const QString &filePath, bool pastePath) {
    QWidget *currentDatasetWidget = getCurrentDatasetWidget(os);
    GT_CHECK(currentDatasetWidget != nullptr, "Current dataset widget not found");

    QWidget *addFileButton = GTWidget::findWidget(os, "addFileButton", currentDatasetWidget);
    GT_CHECK(addFileButton, "addFileButton not found");

    GTFileDialogUtils::TextInput t = pastePath ? GTFileDialogUtils::CopyPaste : GTFileDialogUtils::Typing;

    GTFileDialogUtils *ob = new GTFileDialogUtils(os, filePath, GTGlobals::UseMouse, GTFileDialogUtils::Open, t);
    GTUtilsDialog::waitForDialog(os, ob);

    GTWidget::click(os, addFileButton);
    GTGlobals::sleep(3000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setDatasetInputFiles"
void GTUtilsWorkflowDesigner::setDatasetInputFiles(GUITestOpStatus &os, const QStringList &filePaths) {
    GTGlobals::sleep(200);
    QWidget *currentDatasetWidget = getCurrentDatasetWidget(os);
    GT_CHECK(nullptr != currentDatasetWidget, "Current dataset widget not found");

    QWidget *addFileButton = GTWidget::findWidget(os, "addFileButton", currentDatasetWidget);
    GT_CHECK(nullptr != addFileButton, "addFileButton not found");

    GTUtilsDialog::waitForDialog(os, new GTFileDialogUtils_list(os, filePaths));

    GTWidget::click(os, addFileButton);
    GTGlobals::sleep(3000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "addInputFile"
void GTUtilsWorkflowDesigner::addInputFile(HI::GUITestOpStatus &os, const QString &elementName, const QString &url) {
    click(os, elementName);
    CHECK_OP(os, );
    QFileInfo info(url);
    setDatasetInputFile(os, info.path() + "/" + info.fileName());
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "createDataset"
void GTUtilsWorkflowDesigner::createDataset(HI::GUITestOpStatus &os, QString datasetName) {
    QWidget *plusButton = GTWidget::findButtonByText(os, "+", getDatasetsListWidget(os));
    GT_CHECK(plusButton, "plusButton not found");

    GTUtilsDialog::waitForDialog(os, new DatasetNameEditDialogFiller(os, datasetName));

    GTWidget::click(os, plusButton);
    GTGlobals::sleep();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setDatasetInputFolder"
void GTUtilsWorkflowDesigner::setDatasetInputFolder(HI::GUITestOpStatus &os, QString filePath) {
    QWidget *currentDatasetWidget = getCurrentDatasetWidget(os);
    GT_CHECK(nullptr != currentDatasetWidget, "Current dataset widget not found");

    QWidget *addDirButton = GTWidget::findWidget(os, "addDirButton", currentDatasetWidget);
    GT_CHECK(addDirButton, "addFileButton not found");

    GTFileDialogUtils *ob = new GTFileDialogUtils(os, filePath, "", GTFileDialogUtils::Choose, GTGlobals::UseMouse);
    GTUtilsDialog::waitForDialog(os, ob);

    GTWidget::click(os, addDirButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setDatasetInputFolders"
void GTUtilsWorkflowDesigner::setDatasetInputFolders(GUITestOpStatus &os, const QStringList &dirPaths) {
    QWidget *currentDatasetWidget = getCurrentDatasetWidget(os);
    GT_CHECK(nullptr != currentDatasetWidget, "Current dataset widget not found");

    QWidget *addDirButton = GTWidget::findWidget(os, "addDirButton", currentDatasetWidget);
    GT_CHECK(nullptr != addDirButton, "addFileButton not found");

    GTUtilsDialog::waitForDialog(os, new GTFileDialogUtils_list(os, dirPaths));
    GTWidget::click(os, addDirButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setParameter"
static int getRowIndexOrFail(HI::GUITestOpStatus &os, QTableView *table, const QString &parameter) {
    QAbstractItemModel *model = table->model();
    int rowIndex = -1;
    for (int i = 0; i < model->rowCount(); i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (s.compare(parameter, Qt::CaseInsensitive) == 0) {
            rowIndex = i;
            break;
        }
    }
    GT_CHECK_RESULT(rowIndex != -1, QString("parameter not found: %1").arg(parameter), -1);
    return rowIndex;
}

void GTUtilsWorkflowDesigner::setParameter(HI::GUITestOpStatus &os, QString parameter, QVariant value, valueType type, GTGlobals::UseMethod method) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    CHECK_SET_ERR(table, "tableView not found");

    // Find cell. TODO: scroll to parameter by mouse/keyboard
    class MainThreadAction : public CustomScenario {
    public:
        MainThreadAction(QTableView *table, const QString &parameter)
            : CustomScenario(), table(table), parameter(parameter) {
        }
        void run(HI::GUITestOpStatus &os) {
            int rowIndex = getRowIndexOrFail(os, table, parameter);
            table->scrollTo(table->model()->index(rowIndex, 1));
        }
        QTableView *table;
        QString parameter;
    };
    GTThread::runInMainThread(os, new MainThreadAction(table, parameter));
    GTThread::waitForMainThread();

    int rowIndex = getRowIndexOrFail(os, table, parameter);
    GTMouseDriver::moveTo(GTTableView::getCellPosition(os, table, 1, rowIndex));
    GTMouseDriver::click();
    GTGlobals::sleep();

    //SET VALUE
    setCellValue(os, table, value, type, method);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setTableValue"
void GTUtilsWorkflowDesigner::setTableValue(HI::GUITestOpStatus &os, QString parameter, QVariant value, valueType type, QTableWidget *table, GTGlobals::UseMethod method) {
    int row = -1;
    const int rows = table->rowCount();
    for (int i = 0; i < rows; i++) {
        QString s = table->item(i, 0)->text();
        if (s == parameter) {
            row = i;
            break;
        }
    }
    GT_CHECK(row != -1, QString("parameter not found: %1").arg(parameter));

    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QScrollArea *scrollArea = qobject_cast<QScrollArea *>(GTWidget::findWidget(os, "inputScrollArea", wdWindow));
    GT_CHECK(scrollArea != NULL, "inputPortBox isn't found");
    if (!scrollArea->findChildren<QTableWidget *>().contains(table)) {
        scrollArea = qobject_cast<QScrollArea *>(GTWidget::findWidget(os, "outputScrollArea", wdWindow));
        GT_CHECK(scrollArea != NULL, "outputPortBox isn't found");
        GT_CHECK(scrollArea->findChildren<QTableWidget *>().contains(table), "The owner of the table widget isn't found");
    }
    QScrollBar *scrollBar = scrollArea->verticalScrollBar();
    GT_CHECK(scrollBar != NULL, "Horizontal scroll bar isn't found");

    QRect parentTableRect = scrollArea->rect();
    QPoint globalTopLeftParentTable = scrollArea->mapToGlobal(parentTableRect.topLeft());
    QPoint globalBottomRightParentTable = scrollArea->mapToGlobal(parentTableRect.bottomRight());
    QRect globalParentRect(globalTopLeftParentTable, globalBottomRightParentTable - QPoint(0, 1));

    QTableWidgetItem *item = table->item(row, 1);
    QRect rect = table->visualItemRect(item);
    QPoint globalP = table->viewport()->mapToGlobal(rect.center());

    while (!globalParentRect.contains(globalP)) {
        GTScrollBar::lineDown(os, scrollBar, method);
        rect = table->visualItemRect(item);
        globalP = table->viewport()->mapToGlobal(rect.center());
    }

    GTMouseDriver::moveTo(globalP);
    GTMouseDriver::click();
    GTGlobals::sleep(500);

    //SET VALUE
    setCellValue(os, table, value, type, method);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setCellValue"
void GTUtilsWorkflowDesigner::setCellValue(HI::GUITestOpStatus &os, QWidget *parent, QVariant value, valueType type, GTGlobals::UseMethod method) {
    checkWorkflowDesignerWindowIsActive(os);
    bool ok = true;
    switch (type) {
    case (comboWithFileSelector): {
        GTUtilsDialog::waitForDialog(os, new GTFileDialogUtils(os, value.toString()));
        GTWidget::click(os, GTWidget::findButtonByText(os, "...", parent));
#ifdef Q_OS_WIN
        //added to fix UGENE-3597
        GTKeyboardDriver::keyClick(Qt::Key_Enter);
#endif
        break;
    }
    case (lineEditWithFileSelector): {
        GTLineEdit::setText(os, GTWidget::findExactWidget<QLineEdit *>(os, "mainWidget", parent), value.toString());
        GTKeyboardDriver::keyClick(Qt::Key_Enter);
        break;
    }
    case (spinValue): {
        int spinVal = value.toInt(&ok);
        GT_CHECK(ok, "Wrong input. Int required for GTUtilsWorkflowDesigner::spinValue")
        QSpinBox *spinBox = GTWidget::findWidgetByType<QSpinBox *>(os, parent, "Cell has no QSpinBox widget");
        GTSpinBox::setValue(os, spinBox, spinVal, GTGlobals::UseKeyBoard);
        break;
    }
    case (doubleSpinValue): {
        double spinVal = value.toDouble(&ok);
        GT_CHECK(ok, "Wrong input. Double required for GTUtilsWorkflowDesigner::doubleSpinValue")
        QDoubleSpinBox *doubleSpinBox = GTWidget::findWidgetByType<QDoubleSpinBox *>(os, parent, "Cell has no QDoubleSpinBox widget");
        GTDoubleSpinbox::setValue(os, doubleSpinBox, spinVal, GTGlobals::UseKeyBoard);
        break;
    }
    case (comboValue): {
        int comboVal = value.toInt(&ok);
        QComboBox *comboBox = GTWidget::findWidgetByType<QComboBox *>(os, parent, "Cell has no QComboBox widget");
        if (!ok) {
            QString comboString = value.toString();
            GTComboBox::setIndexWithText(os, comboBox, comboString, true, method);
        } else {
            GTComboBox::setCurrentIndex(os, comboBox, comboVal, true, method);
        }
        break;
    }
    case (textValue): {
        QString lineVal = value.toString();
        QLineEdit *lineEdit = GTWidget::findWidgetByType<QLineEdit *>(os, parent, "Cell has no QLineEdit widget");
        GTLineEdit::setText(os, lineEdit, lineVal);
        GTKeyboardDriver::keyClick(Qt::Key_Enter);
        break;
    }
    case ComboChecks: {
        QStringList values = value.value<QStringList>();
        QComboBox *comboBox = GTWidget::findWidgetByType<QComboBox *>(os, parent, "Cell has no QComboBox/ComboChecks widget");
        GTComboBox::checkValues(os, comboBox, values);
#ifndef Q_OS_WIN
        GTKeyboardDriver::keyClick(Qt::Key_Escape);
#endif
        break;
    }
    case customDialogSelector: {
        GTWidget::click(os, GTWidget::findButtonByText(os, "...", parent));
        break;
    }
    }
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCellValue"
QString GTUtilsWorkflowDesigner::getCellValue(HI::GUITestOpStatus &os, QString parameter, QTableWidget *table) {
    Q_UNUSED(os);
    int row = -1;
    for (int i = 0; i < table->rowCount(); i++) {
        QString s = table->item(i, 0)->text();
        if (s == parameter) {
            row = i;
            break;
        }
    }
    GT_CHECK_RESULT(row != -1, QString("parameter not found: %1").arg(parameter), QString());

    QString result = table->item(row, 1)->text();
    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getInputPortsTable"
QTableWidget *GTUtilsWorkflowDesigner::getInputPortsTable(HI::GUITestOpStatus &os, int index) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QWidget *inputPortBox = GTWidget::findWidget(os, "inputPortBox", wdWindow);
    GTGroupBox::setChecked(os, "inputPortBox", true);
    QList<QTableWidget *> tables = inputPortBox->findChildren<QTableWidget *>();
    foreach (QTableWidget *w, tables) {
        if (!w->isVisible()) {
            tables.removeOne(w);
        }
    }
    int number = tables.count();
    GT_CHECK_RESULT(index < number, QString("there are %1 visiable tables for input ports").arg(number), NULL);
    return tables[index];
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getOutputPortsTable"
QTableWidget *GTUtilsWorkflowDesigner::getOutputPortsTable(GUITestOpStatus &os, int index) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QWidget *outputPortBox = GTWidget::findWidget(os, "outputPortBox", wdWindow);
    GTGroupBox::setChecked(os, "outputPortBox", true);
    QList<QTableWidget *> tables = outputPortBox->findChildren<QTableWidget *>();
    foreach (QTableWidget *w, tables) {
        if (!w->isVisible()) {
            tables.removeOne(w);
        }
    }
    int number = tables.count();
    GT_CHECK_RESULT(index < number, QString("there are %1 visables tables for output ports").arg(number), NULL);
    return tables[index];
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "scrollInputPortsWidgetToTableRow"
void GTUtilsWorkflowDesigner::scrollInputPortsWidgetToTableRow(GUITestOpStatus &os, int tableIndex, const QString &slotName) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QWidget *inputPortBox = GTWidget::findWidget(os, "inputPortBox", wdWindow);
    QTableWidget *table = getInputPortsTable(os, tableIndex);

    QList<QTableWidgetItem *> itemList = table->findItems(slotName, Qt::MatchFixedString);
    GT_CHECK(!itemList.isEmpty(), QString("Can't find item for slot name '%1'").arg(slotName));

    const QRect itemLocalRect = table->visualItemRect(itemList.first());
    const QRect itemPortWidgetRect = QRect(table->viewport()->mapTo(inputPortBox, itemLocalRect.topLeft()),
                                           table->viewport()->mapTo(inputPortBox, itemLocalRect.bottomRight()));

    bool isCenterVisible = inputPortBox->rect().contains(itemPortWidgetRect.center());
    if (isCenterVisible) {
        return;
    }

    QScrollArea *inputScrollArea = GTWidget::findExactWidget<QScrollArea *>(os, "inputScrollArea", inputPortBox);
    QScrollBar *scrollBar = inputScrollArea->verticalScrollBar();
    GTScrollBar::moveSliderWithMouseToValue(os, scrollBar, itemPortWidgetRect.center().y());
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getAllParameters"
QStringList GTUtilsWorkflowDesigner::getAllParameters(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QStringList result;
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", result);

    QAbstractItemModel *model = table->model();
    int iMax = model->rowCount();
    for (int i = 0; i < iMax; i++) {
        QString s = model->data(model->index(i, 0)).toString();
        result << s;
    }
    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getComboBoxParameterValues"
QStringList GTUtilsWorkflowDesigner::getComboBoxParameterValues(HI::GUITestOpStatus &os, QString parameter) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", QStringList());

    //FIND CELL
    QAbstractItemModel *model = table->model();
    int iMax = model->rowCount();
    int row = -1;
    for (int i = 0; i < iMax; i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (s.compare(parameter, Qt::CaseInsensitive) == 0) {
            row = i;
            break;
        }
    }
    GT_CHECK_RESULT(row != -1, QString("parameter not found: %1").arg(parameter), QStringList());
    table->scrollTo(model->index(row, 1));

    GTMouseDriver::moveTo(GTTableView::getCellPosition(os, table, 1, row));
    GTMouseDriver::click();
    GTGlobals::sleep();

    QComboBox *box = qobject_cast<QComboBox *>(table->findChild<QComboBox *>());
    GT_CHECK_RESULT(box, "QComboBox not found. Widget in this cell might be not QComboBox", QStringList());

    QStringList result;
    int valuesCount = box->count();
    for (int i = 0; i < valuesCount; i++) {
        result << box->itemText(i);
    }

    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCheckableComboboxValuesFromInputPortTable"
QList<QPair<QString, bool>> GTUtilsWorkflowDesigner::getCheckableComboboxValuesFromInputPortTable(GUITestOpStatus &os, int tableIndex, const QString &slotName) {
    QList<QPair<QString, bool>> result;

    QTableWidget *table = getInputPortsTable(os, tableIndex);
    GT_CHECK_RESULT(nullptr != table, "table is nullptr", result);

    scrollInputPortsWidgetToTableRow(os, tableIndex, slotName);

    QList<QTableWidgetItem *> itemList = table->findItems(slotName, Qt::MatchFixedString);
    GT_CHECK_RESULT(!itemList.isEmpty(), QString("Can't find item for slot name '%1'").arg(slotName), result);
    const int row = itemList.first()->row();

    GTMouseDriver::moveTo(GTTableView::getCellPosition(os, table, 1, row));
    GTMouseDriver::click();
    GTGlobals::sleep();

    QComboBox *box = qobject_cast<QComboBox *>(table->findChild<QComboBox *>());
    GT_CHECK_RESULT(box, "QComboBox not found. Widget in this cell might be not QComboBox", result);

    QStandardItemModel *checkBoxModel = qobject_cast<QStandardItemModel *>(box->model());
    GT_CHECK_RESULT(nullptr != checkBoxModel, "Unexpected checkbox model", result);

    for (int i = 0; i < checkBoxModel->rowCount(); ++i) {
        QStandardItem *item = checkBoxModel->item(i);
        result << qMakePair(item->data(Qt::DisplayRole).toString(), Qt::Checked == item->checkState());
    }

    return result;
}
#undef GT_METHOD_NAME

namespace {
bool equalStrings(const QString &where, const QString &what, bool exactMatch) {
    if (exactMatch) {
        return (where == what);
    } else {
        return where.contains(what, Qt::CaseInsensitive);
    }
}
}    // namespace

#define GT_METHOD_NAME "getParameter"
QString GTUtilsWorkflowDesigner::getParameter(HI::GUITestOpStatus &os, QString parameter, bool exactMatch) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", "");

    QAbstractItemModel *model = table->model();
    GT_CHECK_RESULT(model, "model not found", "");
    int iMax = model->rowCount();
    int row = -1;
    for (int i = 0; i < iMax; i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (equalStrings(s, parameter, exactMatch)) {
            row = i;
            break;
        }
    }
    GT_CHECK_RESULT(row != -1, "parameter " + parameter + " not found", "");
    QModelIndex idx = model->index(row, 1);

    QVariant var;

    class Scenario : public CustomScenario {
    public:
        Scenario(QAbstractItemModel *_model, QModelIndex _idx, QVariant &_result)
            : model(_model), idx(_idx), result(_result) {
        }
        void run(HI::GUITestOpStatus &os) {
            Q_UNUSED(os);
            result = model->data(idx);
            GTGlobals::sleep(100);
        }

    private:
        QAbstractItemModel *model;
        QModelIndex idx;
        QVariant &result;
    };

    GTThread::runInMainThread(os, new Scenario(model, idx, var));
    return var.toString();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isParameterEnabled"
bool GTUtilsWorkflowDesigner::isParameterEnabled(HI::GUITestOpStatus &os, QString parameter) {
    clickParameter(os, parameter);
    QWidget *w = QApplication::widgetAt(GTMouseDriver::getMousePosition());
    QString s = w->metaObject()->className();

    bool result = !(s == "QWidget");    //if parameter is disabled QWidget is under cursor
    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isParameterRequired"
bool GTUtilsWorkflowDesigner::isParameterRequired(HI::GUITestOpStatus &os, const QString &parameter) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", false);

    // find a cell
    QAbstractItemModel *model = table->model();
    int iMax = model->rowCount();
    int row = -1;
    for (int i = 0; i < iMax; i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (s.contains(parameter, Qt::CaseInsensitive)) {
            row = i;
        }
    }
    GT_CHECK_RESULT(row != -1, "parameter not found", false);
    table->scrollTo(model->index(row, 0));

    const QFont font = model->data(model->index(row, 0), Qt::FontRole).value<QFont>();
    return font.bold();
}
#undef GT_METHOD_NAME

namespace {

int getParameterRow(QTableView *table, const QString &parameter) {
    QAbstractItemModel *model = table->model();
    int iMax = model->rowCount();
    for (int i = 0; i < iMax; i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (s == parameter) {
            return i;
        }
    }
    return -1;
}

}    // namespace

#define GT_METHOD_NAME "clickParameter"
void GTUtilsWorkflowDesigner::clickParameter(HI::GUITestOpStatus &os, const QString &parameter) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", );

    //FIND CELL
    const int row = getParameterRow(table, parameter);
    GT_CHECK_RESULT(row != -1, "parameter not found", );

    QAbstractItemModel *model = table->model();
    table->scrollTo(model->index(row, 1));
    GTMouseDriver::moveTo(GTTableView::getCellPosition(os, table, 1, row));
    GTMouseDriver::click();
    GTGlobals::sleep(500);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isParameterVisible"
bool GTUtilsWorkflowDesigner::isParameterVisible(HI::GUITestOpStatus &os, const QString &parameter) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    GT_CHECK_RESULT(table, "tableView not found", false);
    return -1 != getParameterRow(table, parameter);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getParametersTable"
QTableView *GTUtilsWorkflowDesigner::getParametersTable(HI::GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    return qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setParameterScripting"
void GTUtilsWorkflowDesigner::setParameterScripting(HI::GUITestOpStatus &os, QString parameter, QString scriptMode, bool exactMatch) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QTableView *table = qobject_cast<QTableView *>(GTWidget::findWidget(os, "table", wdWindow));
    CHECK_SET_ERR(table, "tableView not found");

    //FIND CELL
    QAbstractItemModel *model = table->model();
    int row = -1;
    for (int i = 0; i < model->rowCount(); i++) {
        QString s = model->data(model->index(i, 0)).toString();
        if (equalStrings(s, parameter, exactMatch)) {
            row = i;
        }
    }
    GT_CHECK(row != -1, "parameter not found");

    class MainThreadAction : public CustomScenario {
    public:
        MainThreadAction(QTableView *table, int row)
            : CustomScenario(), table(table), row(row) {
        }
        void run(HI::GUITestOpStatus &os) {
            Q_UNUSED(os);
            QAbstractItemModel *model = table->model();
            table->scrollTo(model->index(row, 1));
        }
        QTableView *table;
        int row;
    };
    GTThread::runInMainThread(os, new MainThreadAction(table, row));

    GTMouseDriver::moveTo(GTTableView::getCellPosition(os, table, 2, row));
    GTMouseDriver::click();

    //SET VALUE
    QComboBox *box = qobject_cast<QComboBox *>(table->findChild<QComboBox *>());
    GT_CHECK(box != nullptr, "QComboBox not found. Scripting might be unavaluable for this parameter");
    GTComboBox::setIndexWithText(os, box, scriptMode, false);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "checkErrorList"
int GTUtilsWorkflowDesigner::checkErrorList(HI::GUITestOpStatus &os, QString error) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QListWidget *w = qobject_cast<QListWidget *>(GTWidget::findWidget(os, "infoList", wdWindow));
    GT_CHECK_RESULT(w, "ErrorList widget not found", 0);

    QList<QListWidgetItem *> list = w->findItems(error, Qt::MatchContains);
    return list.size();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getErrors"
QStringList GTUtilsWorkflowDesigner::getErrors(GUITestOpStatus &os) {
    QWidget *wdWindow = getActiveWorkflowDesignerWindow(os);
    QListWidget *w = GTWidget::findExactWidget<QListWidget *>(os, "infoList", wdWindow);
    GT_CHECK_RESULT(w, "ErrorList widget not found", QStringList());

    QStringList errors;
    for (int i = 0; i < w->count(); i++) {
        errors << w->item(i)->text();
    }
    return errors;
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}    // namespace U2

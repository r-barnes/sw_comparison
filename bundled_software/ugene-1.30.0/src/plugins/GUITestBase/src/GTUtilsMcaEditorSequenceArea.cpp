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

#include <QApplication>
#include <QMainWindow>
#include <QStyleOptionSlider>

#include <drivers/GTMouseDriver.h>
#include <primitives/GTScrollBar.h>

#include <U2Core/AppContext.h>
#include <U2Core/DNASequenceObject.h>
#include <U2Core/DNASequenceSelection.h>
#include <U2Core/U2OpStatusUtils.h>

#include <U2View/ADVSingleSequenceWidget.h>
#include <U2View/BaseWidthController.h>
#include <U2View/DrawHelper.h>
#include <U2View/McaEditor.h>
#include <U2View/McaEditorConsensusArea.h>
#include <U2View/McaEditorNameList.h>
#include <U2View/McaEditorReferenceArea.h>
#include <U2View/McaEditorSequenceArea.h>
#include <U2View/McaEditorConsensusArea.h>

#include <U2View/McaEditorWgt.h>
#include <U2View/PanView.h>
#include <U2View/RowHeightController.h>
#include <U2View/SequenceObjectContext.h>

#include "GTUtilsMcaEditor.h"
#include "GTUtilsMcaEditorSequenceArea.h"
#include "GTUtilsMdi.h"
#include "GTUtilsProjectTreeView.cpp"

namespace U2 {
using namespace HI;

#define GT_CLASS_NAME "GTUtilsMcaEditorSequenceArea"

#define GT_METHOD_NAME "getSequenceArea"
McaEditorSequenceArea* GTUtilsMcaEditorSequenceArea::getSequenceArea(GUITestOpStatus &os) {
    QWidget *activeWindow = GTUtilsMdi::activeWindow(os);
    CHECK_OP(os, NULL);

    McaEditorSequenceArea *result = qobject_cast<McaEditorSequenceArea*>(GTWidget::findWidget(os, "mca_editor_sequence_area", activeWindow));
    GT_CHECK_RESULT(NULL != result, "MsaEditorSequenceArea is not found", NULL);
    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getVisibleNames"
QStringList GTUtilsMcaEditorSequenceArea::getVisibleNames(GUITestOpStatus &os) {
    Q_UNUSED(os);
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    McaEditor* editor = mw->findChild<McaEditor*>();
    CHECK_SET_ERR_RESULT(editor != NULL, "McaEditor not found", QStringList());

    McaEditorNameList *nameListArea = GTUtilsMcaEditor::getNameListArea(os);
    CHECK_SET_ERR_RESULT(NULL != nameListArea, "Mca Editor name list area is NULL", QStringList());

    const QList<int> visibleRowsIndexes = editor->getUI()->getDrawHelper()->getVisibleRowsIndexes(nameListArea->height());

    QStringList visibleRowNames;
    foreach(const int rowIndex, visibleRowsIndexes) {
        visibleRowNames << editor->getMaObject()->getRow(rowIndex)->getName();
    }
    return visibleRowNames;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getRowHeight"
int GTUtilsMcaEditorSequenceArea::getRowHeight(GUITestOpStatus &os, int rowNumber) {
    QWidget* activeWindow = GTUtilsMdi::activeWindow(os);
    GT_CHECK_RESULT(activeWindow != NULL, "active mdi window is NULL", 0);
    McaEditorWgt* ui = GTUtilsMdi::activeWindow(os)->findChild<McaEditorWgt*>();
    return ui->getRowHeightController()->getRowHeightByNumber(rowNumber);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickToPosition"
void GTUtilsMcaEditorSequenceArea::clickToPosition(GUITestOpStatus &os, const QPoint &globalMaPosition) {
    McaEditorSequenceArea *mcaSeqArea = GTWidget::findExactWidget<McaEditorSequenceArea *>(os, "mca_editor_sequence_area", GTUtilsMdi::activeWindow(os));
    GT_CHECK(NULL != mcaSeqArea, "MCA Editor sequence area is not found");
    GT_CHECK(mcaSeqArea->isInRange(globalMaPosition), 
             QString("Position is out of range: [%1, %2], range: [%3, %4]")
             .arg(globalMaPosition.x()).arg(globalMaPosition.y())
             .arg(mcaSeqArea->getEditor()->getAlignmentLen()).arg(mcaSeqArea->getNumDisplayableSequences()));

    scrollToPosition(os, globalMaPosition);
    GTGlobals::sleep();

    const QPoint positionCenter(mcaSeqArea->getEditor()->getUI()->getBaseWidthController()->getBaseScreenCenter(globalMaPosition.x()),
        mcaSeqArea->getEditor()->getUI()->getRowHeightController()->getRowScreenRangeByNumber(globalMaPosition.y()).center());
    GT_CHECK(mcaSeqArea->rect().contains(positionCenter, false), "Position is not visible");

    GTMouseDriver::moveTo(mcaSeqArea->mapToGlobal(positionCenter));
    GTMouseDriver::click();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "scrollToPosition"
void GTUtilsMcaEditorSequenceArea::scrollToPosition(GUITestOpStatus &os, const QPoint &position) {
    McaEditorSequenceArea *mcaSeqArea = GTWidget::findExactWidget<McaEditorSequenceArea *>(os, "mca_editor_sequence_area", GTUtilsMdi::activeWindow(os));
    GT_CHECK(NULL != mcaSeqArea, "MSA Editor sequence area is not found");
    GT_CHECK(mcaSeqArea->isInRange(position), 
             QString("Position is out of range: [%1, %2], range: [%3, %4]")
             .arg(position.x()).arg(position.y())
             .arg(mcaSeqArea->getEditor()->getAlignmentLen()).arg(mcaSeqArea->getNumDisplayableSequences()));
    
    CHECK(!mcaSeqArea->isVisible(position, false), );

    if (GTUtilsProjectTreeView::isVisible(os)){
        GTUtilsProjectTreeView::toggleView(os);
    }

    if (!mcaSeqArea->isRowVisible(position.y(), false)) {
        GTUtilsMcaEditor::scrollToRead(os, position.y());
    }

    if (!mcaSeqArea->isPositionVisible(position.x(), false)) {
        scrollToBase(os, position.x());
    }

    CHECK_SET_ERR(mcaSeqArea->isVisible(position, false), "The position is still invisible after scrolling");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "scrollToBase"
void GTUtilsMcaEditorSequenceArea::scrollToBase(GUITestOpStatus &os, int position) {
    const int scrollBarValue = GTUtilsMcaEditor::getEditorUi(os)->getBaseWidthController()->getBaseGlobalRange(position).center() -
                               GTUtilsMcaEditor::getEditorUi(os)->getSequenceArea()->width() / 2;
    GTScrollBar::moveSliderWithMouseToValue(os,
                                            GTUtilsMcaEditor::getHorizontalScrollBar(os),
                                            scrollBarValue);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickCollapseTriangle"
void GTUtilsMcaEditorSequenceArea::clickCollapseTriangle(GUITestOpStatus &os, QString rowName, bool showChromatogram){
    McaEditorSequenceArea *mcaEditArea = qobject_cast<McaEditorSequenceArea*>(GTWidget::findWidget(os, "mca_editor_sequence_area"));
    GT_CHECK(mcaEditArea != NULL, "McaEditorSequenceArea not found");

    int rowNum = getVisibleNames(os).indexOf(rowName);
    GT_CHECK(rowNum != -1, "sequence not found in nameList");
    QWidget* nameList = GTWidget::findWidget(os, "mca_editor_name_list");
    int yPos = mcaEditArea->getEditor()->getUI()->getRowHeightController()->getRowScreenCenterByNumber(rowNum);
    if (showChromatogram) {
        yPos -= 65;
    }
    QPoint localCoord = QPoint(15, yPos);
    QPoint globalCoord = nameList->mapToGlobal(localCoord);
    GTMouseDriver::moveTo(globalCoord);
    GTMouseDriver::click();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isCollapsed"
bool GTUtilsMcaEditorSequenceArea::isChromatogramShown(GUITestOpStatus &os, QString rowName){
    McaEditorSequenceArea *mcaEditArea = qobject_cast<McaEditorSequenceArea*>(GTWidget::findWidget(os, "mca_editor_sequence_area"));
    GT_CHECK_RESULT(mcaEditArea != NULL, "McaEditorSequenceArea not found", false);
    int rowNum = GTUtilsMcaEditor::getReadsNames(os).indexOf(rowName);
    GT_CHECK_RESULT(rowNum != -1, "sequence not found in nameList", false);
    int rowHeight = mcaEditArea->getEditor()->getUI()->getRowHeightController()->getRowHeight(rowNum);
    bool isCollapsed = rowHeight > 100;
    return isCollapsed;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getNameList"
QStringList GTUtilsMcaEditorSequenceArea::getNameList(GUITestOpStatus &os) {
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    McaEditor* editor = mw->findChild<McaEditor*>();
    CHECK_SET_ERR_RESULT(editor != NULL, "MsaEditor not found", QStringList());

    QStringList result = editor->getMaObject()->getMultipleAlignment()->getRowNames();

    return result;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "callContextMenu"
void GTUtilsMcaEditorSequenceArea::callContextMenu(GUITestOpStatus &os, const QPoint &innerCoords) {
    if (innerCoords.isNull()) {
        GTWidget::click(os, getSequenceArea(os), Qt::RightButton);
    } else {
        moveTo(os, innerCoords);
        GTMouseDriver::click(Qt::RightButton);
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "moveTo"
void GTUtilsMcaEditorSequenceArea::moveTo(GUITestOpStatus &os, const QPoint &p)
{
    QPoint convP = convertCoordinates(os, p);

    GTMouseDriver::moveTo(convP);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "convertCoordinates"
QPoint GTUtilsMcaEditorSequenceArea::convertCoordinates(GUITestOpStatus &os, const QPoint p) {
    QWidget* activeWindow = GTUtilsMdi::activeWindow(os);
    McaEditorSequenceArea *mcaEditArea = qobject_cast<McaEditorSequenceArea*>(GTWidget::findWidget(os, "mca_editor_sequence_area", activeWindow));
    GT_CHECK_RESULT(mcaEditArea != NULL, "McaEditorSequenceArea not found", QPoint());

    const int posX = static_cast<int>(mcaEditArea->getEditor()->getUI()->getBaseWidthController()->getBaseGlobalRange(p.x()).center());
    const int posY = static_cast<int>(mcaEditArea->getEditor()->getUI()->getRowHeightController()->getRowGlobalRangeByNumber(p.y()).center());
    return mcaEditArea->mapToGlobal(QPoint(posX, posY));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReferenceReg"
QString GTUtilsMcaEditorSequenceArea::getReferenceReg(GUITestOpStatus &os, int num, int length) {
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    GT_CHECK_RESULT(mw != NULL, "QMainWindow not found", QString());
    McaEditor* editor = mw->findChild<McaEditor*>();
    GT_CHECK_RESULT(editor != NULL, "McaEditor not found", QString());
    MultipleChromatogramAlignmentObject* obj = editor->getMaObject();
    GT_CHECK_RESULT(obj != NULL, "MultipleChromatogramAlignmentObject not found", QString());

    U2OpStatus2Log status;
    QByteArray seq = obj->getReferenceObj()->getSequenceData(U2Region(num, length), status);
    CHECK_OP(status, QString());

    return seq;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReferenceReg"
QString GTUtilsMcaEditorSequenceArea::getSelectedReferenceReg(GUITestOpStatus &os) {
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    GT_CHECK_RESULT(mw != NULL, "QMainWindow not found", QString());
    McaEditor* editor = mw->findChild<McaEditor*>();
    GT_CHECK_RESULT(editor != NULL, "McaEditor not found", QString());
    MultipleChromatogramAlignmentObject* obj = editor->getMaObject();
    GT_CHECK_RESULT(obj != NULL, "MultipleChromatogramAlignmentObject not found", QString());

    U2Region sel = GTUtilsMcaEditorSequenceArea::getReferenceSelection(os);
    int num = sel.startPos;
    int length = sel.length;

    U2OpStatus2Log status;
    QByteArray seq = obj->getReferenceObj()->getSequenceData(U2Region(num, length), status);

    return seq;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "moveTheBorderBetweenAlignmentAndRead"
void GTUtilsMcaEditorSequenceArea::moveTheBorderBetweenAlignmentAndRead(HI::GUITestOpStatus &os, int shift) {
    QStringList visible = getVisibleNames(os);
    GT_CHECK_RESULT(visible.size() != 0, "No visible reads", );
    QString firstVisible = visible.first();

    const QRect sequenceNameRect = GTUtilsMcaEditor::getReadNameRect(os, firstVisible);
    GTMouseDriver::moveTo(QPoint (sequenceNameRect.right() + 2, sequenceNameRect.center().y()));
    GTMouseDriver::press(Qt::LeftButton);
    GTGlobals::sleep(1000);
    GTMouseDriver::moveTo(QPoint(sequenceNameRect.right() + 2 + shift, sequenceNameRect.center().y()));
    GTMouseDriver::release(Qt::LeftButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "dragAndDrop"
void GTUtilsMcaEditorSequenceArea::dragAndDrop(HI::GUITestOpStatus & /*os*/, const QPoint p) {
    GTMouseDriver::click();
    GTGlobals::sleep(1000);
    GTMouseDriver::press(Qt::LeftButton);
    GTGlobals::sleep(1000);
    GTMouseDriver::moveTo(p);
    GTMouseDriver::release(Qt::LeftButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getSelectedRowsNum"
U2Region GTUtilsMcaEditorSequenceArea::getSelectedRowsNum(GUITestOpStatus &os) {
    McaEditorNameList *mcaNameList = GTUtilsMcaEditor::getNameListArea(os);
    CHECK_SET_ERR_RESULT(mcaNameList != NULL, "McaEditorNameList not found", U2Region());

    U2Region selection = mcaNameList->getSelection();
    return selection;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getSelectedRowsNames"
QStringList GTUtilsMcaEditorSequenceArea::getSelectedRowsNames(GUITestOpStatus &os) {
    U2Region sel = getSelectedRowsNum(os);
    QStringList names = getNameList(os);

    QStringList res;
    for (int i = sel.startPos; i < sel.endPos(); i++) {
        res << names[i];
    }

    return res;
}
#undef GT_METHOD_NAME


#define GT_METHOD_NAME "getSelectedRect"
QRect GTUtilsMcaEditorSequenceArea::getSelectedRect(GUITestOpStatus &os) {
    McaEditorSequenceArea *mcaEditArea = qobject_cast<McaEditorSequenceArea*>(GTWidget::findWidget(os, "mca_editor_sequence_area"));
    GT_CHECK_RESULT(mcaEditArea != NULL, "McaEditorSequenceArea not found", QRect());

    return mcaEditArea->getSelection().getRect();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickToReferencePosition"
void GTUtilsMcaEditorSequenceArea::clickToReferencePosition(GUITestOpStatus &os, const qint64 num) {
    QPoint selectedPoint(num, 2);
    McaEditorSequenceArea *mcaSeqArea = GTWidget::findExactWidget<McaEditorSequenceArea *>(os, "mca_editor_sequence_area", GTUtilsMdi::activeWindow(os));
    GT_CHECK(NULL != mcaSeqArea, "MCA Editor sequence area is not found");
    GT_CHECK(mcaSeqArea->isInRange(selectedPoint), 
             QString("Position is out of range: [%1, %2], range: [%3, %4]")
             .arg(selectedPoint.x()).arg(selectedPoint.y())
             .arg(mcaSeqArea->getEditor()->getAlignmentLen()).arg(mcaSeqArea->getNumDisplayableSequences()));
    

    scrollToPosition(os, selectedPoint);

    const QPoint positionCenter(mcaSeqArea->getEditor()->getUI()->getBaseWidthController()->getBaseScreenCenter(selectedPoint.x()), 2);
    GT_CHECK(mcaSeqArea->rect().contains(positionCenter, false), "Position is not visible");

    PanView *panView = qobject_cast<PanView*>(GTWidget::findWidget(os, "mca_editor_reference_area"));
    GT_CHECK(panView != NULL, "Pan view area is not found");

    QPoint p = panView->mapToGlobal(positionCenter);

    GTMouseDriver::moveTo(p);
    GTMouseDriver::click();

}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCharacterModificationMode"
short GTUtilsMcaEditorSequenceArea::getCharacterModificationMode(GUITestOpStatus &os) {
    McaEditorSequenceArea* mcaSeqArea = GTUtilsMcaEditorSequenceArea::getSequenceArea(os);
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor sequence area is not found", 3);

    short mod = mcaSeqArea->getModInfo();
    return mod;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getSelectedChar"
char GTUtilsMcaEditorSequenceArea::getSelectedReadChar(GUITestOpStatus &os) {
    QRect selection = GTUtilsMcaEditorSequenceArea::getSelectedRect(os);
    GT_CHECK_RESULT(selection.width() > 0 && selection.height() > 0, "There is no selection", U2Mca::INVALID_CHAR);
    GT_CHECK_RESULT(selection.width() <= 1 && selection.height() <= 1, "The selection is too big", U2Mca::INVALID_CHAR);
    int rowNum = selection.y();
    qint64 pos = selection.x();

    McaEditorSequenceArea* mcaSeqArea = GTUtilsMcaEditorSequenceArea::getSequenceArea(os);
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor sequence area is not found", U2Mca::INVALID_CHAR);

    McaEditor* mcaEditor = mcaSeqArea->getEditor();
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor is not found", U2Mca::INVALID_CHAR);

    MultipleChromatogramAlignmentObject* mcaObj = mcaEditor->getMaObject();
    GT_CHECK_RESULT(mcaObj != NULL, "MCA Object is not found", U2Mca::INVALID_CHAR);

    const MultipleChromatogramAlignmentRow mcaRow = mcaObj->getRow(rowNum);

    char selectedChar = mcaRow->charAt(pos);
    return selectedChar;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReadCharByPos"
char GTUtilsMcaEditorSequenceArea::getReadCharByPos(GUITestOpStatus &os, const QPoint p) {
    int rowNum = p.y();
    qint64 pos = p.x();

    McaEditorSequenceArea* mcaSeqArea = GTUtilsMcaEditorSequenceArea::getSequenceArea(os);
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor sequence area is not found", U2Mca::INVALID_CHAR);

    McaEditor* mcaEditor = mcaSeqArea->getEditor();
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor is not found", U2Mca::INVALID_CHAR);

    MultipleChromatogramAlignmentObject* mcaObj = mcaEditor->getMaObject();
    GT_CHECK_RESULT(mcaObj != NULL, "MCA Object is not found", U2Mca::INVALID_CHAR);

    const MultipleChromatogramAlignmentRow mcaRow = mcaObj->getRow(rowNum);

    char selectedChar = mcaRow->charAt(pos);
    return selectedChar;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getRowLength"
qint64 GTUtilsMcaEditorSequenceArea::getRowLength(GUITestOpStatus &os, const int numRow) {
    McaEditorSequenceArea* mcaSeqArea = GTUtilsMcaEditorSequenceArea::getSequenceArea(os);
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor sequence area is not found", 0);

    McaEditor* mcaEditor = mcaSeqArea->getEditor();
    GT_CHECK_RESULT(mcaSeqArea != NULL, "MCA Editor is not found", 0);

    MultipleChromatogramAlignmentObject* mcaObj = mcaEditor->getMaObject();
    GT_CHECK_RESULT(mcaObj != NULL, "MCA Object is not found", 0);

    const MultipleChromatogramAlignmentRow mcaRow = mcaObj->getRow(numRow);

    qint64 rowLength = mcaRow->getCoreLength();
    return rowLength;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReferenceLength"
qint64 GTUtilsMcaEditorSequenceArea::getReferenceLength(GUITestOpStatus &os) {
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    GT_CHECK_RESULT(mw != NULL, "QMainWindow not found", 0);
    McaEditor* editor = mw->findChild<McaEditor*>();
    GT_CHECK_RESULT(editor != NULL, "McaEditor not found", 0);
    MultipleChromatogramAlignmentObject* obj = editor->getMaObject();
    GT_CHECK_RESULT(obj != NULL, "MultipleChromatogramAlignmentObject not found", 0);

    U2OpStatus2Log status;
    qint64 refLength = obj->getReferenceObj()->getSequenceLength();

    return refLength;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReferenceLength"
qint64 GTUtilsMcaEditorSequenceArea::getReferenceLengthWithGaps(GUITestOpStatus &os) {
    QMainWindow* mw = AppContext::getMainWindow()->getQMainWindow();
    GT_CHECK_RESULT(mw != NULL, "QMainWindow not found", 0);
    McaEditor* editor = mw->findChild<McaEditor*>();
    GT_CHECK_RESULT(editor != NULL, "McaEditor not found", 0);
    MultipleChromatogramAlignmentObject* obj = editor->getMaObject();
    GT_CHECK_RESULT(obj != NULL, "MultipleChromatogramAlignmentObject not found", 0);

    int length = obj->getReferenceLengthWithGaps();

    return length;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getReferenceSelection"
U2Region GTUtilsMcaEditorSequenceArea::getReferenceSelection(GUITestOpStatus &os) {
    McaEditorReferenceArea *mcaEditArea = qobject_cast<McaEditorReferenceArea*>(GTWidget::findWidget(os, "mca_editor_reference_area"));
    GT_CHECK_RESULT(mcaEditArea != NULL, "McaEditorReferenceArea not found", U2Region());

    SequenceObjectContext* seqContext = mcaEditArea->getSequenceContext();
    GT_CHECK_RESULT(seqContext != NULL, "SequenceObjectContext not found", U2Region());

    DNASequenceSelection* dnaSel = seqContext->getSequenceSelection();
    GT_CHECK_RESULT(dnaSel != NULL, "DNASequenceSelection not found", U2Region());

    QVector<U2Region> region = dnaSel->getSelectedRegions();

    CHECK(region.size() != 0, U2Region());

    GT_CHECK_RESULT(region.size() == 1, "Incorrect selected region", U2Region());

    return region.first();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getSelectedConsensusReg"
QString GTUtilsMcaEditorSequenceArea::getSelectedConsensusReg(GUITestOpStatus &os) {
    McaEditorConsensusArea* consArea = GTUtilsMcaEditor::getConsensusArea(os);
    GT_CHECK_RESULT(consArea != NULL, "Consensus area not found", QString());

    QSharedPointer<MSAEditorConsensusCache> consCache = consArea->getConsensusCache();

    U2Region sel = GTUtilsMcaEditorSequenceArea::getReferenceSelection(os);
    int start = sel.startPos;
    int length = sel.length;

    QString res;
    for (int i = 0; i < length; i++) {
        int pos = start + i;
        char ch = consCache->getConsensusChar(pos);
        res.append(ch);
    }
    return res;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getConsensusStringByPos"
QString GTUtilsMcaEditorSequenceArea::getConsensusStringByRegion(GUITestOpStatus &os, const U2Region reg) {
    McaEditorConsensusArea* consArea = GTUtilsMcaEditor::getConsensusArea(os);
    GT_CHECK_RESULT(consArea != NULL, "Consensus area not found", QString());

    QSharedPointer<MSAEditorConsensusCache> consCache = consArea->getConsensusCache();

    int start = reg.startPos;
    int length = reg.length;

    QString res;
    for (int i = 0; i < length; i++) {
        int pos = start + i;
        char ch = consCache->getConsensusChar(pos);
        res.append(ch);
    }
    return res;
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}//namespace

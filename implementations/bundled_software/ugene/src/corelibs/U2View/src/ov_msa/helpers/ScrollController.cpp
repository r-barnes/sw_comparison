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

#include <U2Core/MultipleAlignmentObject.h>
#include <U2Core/SignalBlocker.h>

#include "BaseWidthController.h"
#include "DrawHelper.h"
#include "RowHeightController.h"
#include "ScrollController.h"
#include "ov_msa/MaEditor.h"
#include "ov_msa/MSACollapsibleModel.h"
#include "ov_msa/view_rendering/MaEditorSequenceArea.h"
#include "ov_msa/view_rendering/MaEditorWgt.h"

namespace U2 {

ScrollController::ScrollController(MaEditor *maEditor, MaEditorWgt *maEditorUi, MSACollapsibleItemModel *collapsibleModel)
    : QObject(maEditorUi),
      maEditor(maEditor),
      ui(maEditorUi),
      collapsibleModel(collapsibleModel),
      savedFirstVisibleRowIndex(0),
      savedFirstVisibleRowAdditionalOffset(0)
{
    connect(this, SIGNAL(si_visibleAreaChanged()), maEditorUi, SIGNAL(si_completeRedraw()));
    connect(collapsibleModel, SIGNAL(si_aboutToBeToggled()), SLOT(sl_collapsibleModelIsAboutToBeChanged()));
    connect(collapsibleModel, SIGNAL(si_toggled()), SLOT(sl_collapsibleModelChanged()));
}

void ScrollController::init(GScrollBar *hScrollBar, GScrollBar *vScrollBar) {
    this->hScrollBar = hScrollBar;
    hScrollBar->setValue(0);
    connect(hScrollBar, SIGNAL(valueChanged(int)), SIGNAL(si_visibleAreaChanged()));

    this->vScrollBar = vScrollBar;
    vScrollBar->setValue(0);
    connect(vScrollBar, SIGNAL(valueChanged(int)), SIGNAL(si_visibleAreaChanged()));

    sl_updateScrollBars();
}

QPoint ScrollController::getScreenPosition() const {
    return QPoint(hScrollBar->value(), vScrollBar->value());
}

QPoint ScrollController::getGlobalMousePosition(const QPoint& mousePos) const {
    return mousePos + getScreenPosition();
}

void ScrollController::updateHorizontalScrollBar() {
    updateHorizontalScrollBarPrivate();
    emit si_visibleAreaChanged();
}

void ScrollController::updateVerticalScrollBar() {
    updateVerticalScrollBarPrivate();
    emit si_visibleAreaChanged();
}

void ScrollController::scrollToRowByNumber(int rowNumber, int widgetHeight) {
    const U2Region rowRegion = ui->getRowHeightController()->getRowGlobalRangeByNumber(rowNumber);
    const U2Region visibleRegion = getVerticalRangeToDrawIn(widgetHeight);
    if (rowRegion.startPos < visibleRegion.startPos) {
        vScrollBar->setValue(static_cast<int>(rowRegion.startPos));
    } else if (rowRegion.endPos() >= visibleRegion.endPos()) {
        if (rowRegion.length > visibleRegion.length) {
            vScrollBar->setValue(static_cast<int>(rowRegion.startPos));
        } else if (rowRegion.startPos > visibleRegion.startPos) {
            vScrollBar->setValue(static_cast<int>(rowRegion.endPos() - widgetHeight));
        }
    }
}

void ScrollController::scrollToBase(int baseNumber, int widgetWidth) {
    const U2Region baseRange = U2Region(ui->getBaseWidthController()->getBaseGlobalOffset(baseNumber), maEditor->getColumnWidth());
    const U2Region visibleRange = getHorizontalRangeToDrawIn(widgetWidth);
    if (baseRange.startPos < visibleRange.startPos) {
        hScrollBar->setValue(static_cast<int>(baseRange.startPos));
    } else if (baseRange.endPos() >= visibleRange.endPos()) {
        hScrollBar->setValue(static_cast<int>(baseRange.endPos() - widgetWidth));
    }
}

void ScrollController::scrollToPoint(const QPoint &maPoint, const QSize &screenSize) {
    scrollToBase(maPoint.x(), screenSize.width());
    scrollToRowByNumber(maPoint.y(), screenSize.height());
}

void ScrollController::centerBase(int baseNumber, int widgetWidth) {
    const U2Region baseGlobalRange = ui->getBaseWidthController()->getBaseGlobalRange(baseNumber);
    const U2Region visibleRange = getHorizontalRangeToDrawIn(widgetWidth);
    const int newScreenXOffset = baseGlobalRange.startPos - visibleRange.length / 2;
    hScrollBar->setValue(newScreenXOffset);
}

void ScrollController::centerRow(int rowNumber, int widgetHeight) {
    const U2Region rowGlobalRange = ui->getRowHeightController()->getRowGlobalRangeByNumber(rowNumber);
    const U2Region visibleRange = getVerticalRangeToDrawIn(widgetHeight);
    const int newScreenYOffset = rowGlobalRange.startPos - visibleRange.length / 2;
    vScrollBar->setValue(newScreenYOffset);
}

void ScrollController::centerPoint(const QPoint &maPoint, const QSize &widgetSize) {
    centerBase(maPoint.x(), widgetSize.width());
    centerRow(maPoint.y(), widgetSize.height());
}

void ScrollController::setHScrollbarValue(int value) {
    hScrollBar->setValue(value);
}

void ScrollController::setVScrollbarValue(int value) {
    vScrollBar->setValue(value);
}

void ScrollController::setFirstVisibleBase(int firstVisibleBase) {
    hScrollBar->setValue(ui->getBaseWidthController()->getBaseGlobalOffset(firstVisibleBase));
}

void ScrollController::setFirstVisibleRowByNumber(int firstVisibleRowNumber) {
    const int firstVisibleRowIndex = ui->getCollapseModel()->mapToRow(firstVisibleRowNumber);
    setFirstVisibleRowByIndex(firstVisibleRowIndex);
}

void ScrollController::setFirstVisibleRowByIndex(int firstVisibleRowIndex) {
    vScrollBar->setValue(ui->getRowHeightController()->getRowGlobalOffset(firstVisibleRowIndex));
}

void ScrollController::scrollSmoothly(const Directions &directions) {
    QAbstractSlider::SliderAction horizontalAction = QAbstractSlider::SliderNoAction;
    QAbstractSlider::SliderAction verticalAction = QAbstractSlider::SliderNoAction;

    if (directions.testFlag(Up)) {
        verticalAction = QAbstractSlider::SliderSingleStepSub;
    }
    if (directions.testFlag(Down)) {
        verticalAction = QAbstractSlider::SliderSingleStepAdd;
    }

    if (directions.testFlag(Left)) {
        horizontalAction = QAbstractSlider::SliderSingleStepSub;
    }
    if (directions.testFlag(Right)) {
        horizontalAction = QAbstractSlider::SliderSingleStepAdd;
    }

    if (verticalAction != vScrollBar->getRepeatAction()) {
        vScrollBar->setupRepeatAction(verticalAction, 500, 50);
    }

    if (horizontalAction != hScrollBar->getRepeatAction()) {
        hScrollBar->setupRepeatAction(horizontalAction, 500, 50);
    }
}

void ScrollController::stopSmoothScrolling() {
    hScrollBar->setupRepeatAction(QAbstractSlider::SliderNoAction);
    vScrollBar->setupRepeatAction(QAbstractSlider::SliderNoAction);
}

void ScrollController::scrollStep(ScrollController::Direction direction) {
    switch (direction) {
    case Up:
        vScrollBar->triggerAction(QAbstractSlider::SliderSingleStepSub);
        break;
    case Down:
        vScrollBar->triggerAction(QAbstractSlider::SliderSingleStepAdd);
        break;
    case Left:
        hScrollBar->triggerAction(QAbstractSlider::SliderSingleStepSub);
        break;
    case Right:
        hScrollBar->triggerAction(QAbstractSlider::SliderSingleStepAdd);
        break;
    default:
        FAIL("An unknown direction" ,);
        break;
    }
}

void ScrollController::scrollPage(ScrollController::Direction direction) {
    switch (direction) {
    case Up:
        vScrollBar->triggerAction(QAbstractSlider::SliderPageStepSub);
        break;
    case Down:
        vScrollBar->triggerAction(QAbstractSlider::SliderPageStepAdd);
        break;
    case Left:
        hScrollBar->triggerAction(QAbstractSlider::SliderPageStepSub);
        break;
    case Right:
        hScrollBar->triggerAction(QAbstractSlider::SliderPageStepAdd);
        break;
    default:
        FAIL("An unknown direction" ,);
        break;
    }
}

void ScrollController::scrollToEnd(ScrollController::Direction direction) {
    switch (direction) {
    case Up:
        vScrollBar->triggerAction(QAbstractSlider::SliderToMinimum);
        break;
    case Down:
        vScrollBar->triggerAction(QAbstractSlider::SliderToMaximum);
        break;
    case Left:
        hScrollBar->triggerAction(QAbstractSlider::SliderToMinimum);
        break;
    case Right:
        hScrollBar->triggerAction(QAbstractSlider::SliderToMaximum);
        break;
    default:
        FAIL("An unknown direction", );
        break;
    }
}

void ScrollController::scrollToMovedSelection(int deltaX, int deltaY) {
    const Direction direction = (deltaX != 0 ? (deltaX < 0 ? ScrollController::Left : ScrollController::Right)
                                             : (deltaY != 0 ? (deltaY < 0 ? ScrollController::Up
                                                                          : ScrollController::Down)
                                                            : ScrollController::None));
    scrollToMovedSelection(direction);
}

void ScrollController::scrollToMovedSelection(ScrollController::Direction direction) {
    U2Region fullyVisibleRegion;
    U2Region selectionRegion;
    const MaEditorSelection selection = ui->getSequenceArea()->getSelection();
    int selectionEdgePosition = 0;
    const QSize widgetWize = ui->getSequenceArea()->size();

    switch (direction) {
    case Up:
        fullyVisibleRegion = ui->getDrawHelper()->getVisibleRowsNumbers(widgetWize.height(), false, false);
        selectionRegion = selection.getYRegion();
        selectionEdgePosition = static_cast<int>(selectionRegion.startPos);
        break;
    case Down:
        fullyVisibleRegion = ui->getDrawHelper()->getVisibleRowsNumbers(widgetWize.height(), false, false);
        selectionRegion = selection.getYRegion();
        selectionEdgePosition = static_cast<int>(selectionRegion.endPos() - 1);
        break;
    case Left:
        fullyVisibleRegion = ui->getDrawHelper()->getVisibleBases(widgetWize.width(), false, false);
        selectionRegion = selection.getXRegion();
        selectionEdgePosition = static_cast<int>(selectionRegion.startPos);
        break;
    case Right:
        fullyVisibleRegion = ui->getDrawHelper()->getVisibleBases(widgetWize.width(), false, false);
        selectionRegion = selection.getXRegion();
        selectionEdgePosition = static_cast<int>(selectionRegion.endPos() - 1);
        break;
    case None:
        return;
    default:
        FAIL("An unknown direction", );
        break;
    }

    const bool selectionEdgeIsFullyVisible = fullyVisibleRegion.contains(selectionEdgePosition);
    if (!selectionEdgeIsFullyVisible) {
        switch (direction) {
        case Up:
        case Down:
            scrollToRowByNumber(static_cast<int>(selectionEdgePosition), widgetWize.height());
            break;
        case Left:
        case Right:
            scrollToBase(static_cast<int>(selectionEdgePosition), widgetWize.width());
            break;
        case None:
            return;
        default:
            FAIL("An unknown direction", );
            break;
        }
    }
}

int ScrollController::getFirstVisibleBase(bool countClipped) const {
    if (maEditor->getAlignmentLen() == 0) {
        return 0;
    }
    const bool removeClippedBase = !countClipped && (getAdditionalXOffset() != 0);
    const int firstVisibleBase = ui->getBaseWidthController()->globalXPositionToColumn(hScrollBar->value()) + (removeClippedBase ? 1 : 0);
    assert(firstVisibleBase < maEditor->getAlignmentLen());
    return qMin(firstVisibleBase, maEditor->getAlignmentLen() - 1);
}

int ScrollController::getLastVisibleBase(int widgetWidth, bool countClipped) const {
    const bool removeClippedBase = !countClipped && ((hScrollBar->value() + widgetWidth) % maEditor->getColumnWidth() != 0);
    const int lastVisiblebase = ui->getBaseWidthController()->globalXPositionToColumn(hScrollBar->value() + widgetWidth - 1) - (removeClippedBase ? 1 : 0);
    return qMin(lastVisiblebase, maEditor->getAlignmentLen() - 1);
}

int ScrollController::getFirstVisibleRowIndex(bool countClipped) const {
    const bool removeClippedRow = !(countClipped || getAdditionalYOffset() == 0);
    return ui->getRowHeightController()->globalYPositionToRowIndex(vScrollBar->value()) + (removeClippedRow ? 1 : 0);
}

int ScrollController::getFirstVisibleRowNumber(bool countClipped) const {
    return collapsibleModel->rowToMap(getFirstVisibleRowIndex(countClipped));
}

int ScrollController::getLastVisibleRowIndex(int widgetHeight, bool countClipped) const {
    return collapsibleModel->mapToRow(getLastVisibleRowNumber(widgetHeight, countClipped));
}

int ScrollController::getLastVisibleRowNumber(int widgetHeight, bool countClipped) const {
    int lastVisibleRowNumber = ui->getRowHeightController()->globalYPositionToRowNumber(vScrollBar->value() + widgetHeight);
    if (lastVisibleRowNumber < 0) {
        lastVisibleRowNumber = collapsibleModel->getDisplayableRowsCount() - 1;
    }
    const U2Region lastRowScreenRegion = ui->getRowHeightController()->getRowScreenRangeByNumber(lastVisibleRowNumber);
    const bool removeClippedRow = !countClipped && lastRowScreenRegion.endPos() > widgetHeight;
    return lastVisibleRowNumber - (removeClippedRow ? 1 : 0);
}

QPoint ScrollController::getMaPointByScreenPoint(const QPoint &point) const {
    const int columnNumber = ui->getBaseWidthController()->screenXPositionToColumn(point.x());
    int rowNumber = ui->getRowHeightController()->screenYPositionToRowNumber(point.y());
    if (-1 == rowNumber) {
        rowNumber = ui->getCollapseModel()->getDisplayableRowsCount();
    }
    return QPoint(columnNumber, rowNumber);
}

GScrollBar *ScrollController::getHorizontalScrollBar() const {
    return hScrollBar;
}

GScrollBar *ScrollController::getVerticalScrollBar() const {
    return vScrollBar;
}

void ScrollController::sl_zoomScrollBars() {
    zoomHorizontalScrollBarPrivate();
    zoomVerticalScrollBarPrivate();
    emit si_visibleAreaChanged();
}

void ScrollController::sl_updateScrollBars() {
    updateHorizontalScrollBarPrivate();
    updateVerticalScrollBarPrivate();
    emit si_visibleAreaChanged();
}

void ScrollController::sl_collapsibleModelIsAboutToBeChanged() {
    savedFirstVisibleRowIndex = getFirstVisibleRowIndex(true);
    savedFirstVisibleRowAdditionalOffset = getScreenPosition().y() - ui->getRowHeightController()->getRowGlobalOffset(savedFirstVisibleRowIndex);
}

void ScrollController::sl_collapsibleModelChanged() {
    const int newFirstVisibleRowIndex = collapsibleModel->rowToMap(savedFirstVisibleRowIndex);
    const int newFirstVisibleRowOffset = ui->getRowHeightController()->getRowGlobalOffset(newFirstVisibleRowIndex);
    setVScrollbarValue(newFirstVisibleRowOffset + savedFirstVisibleRowAdditionalOffset);
}

int ScrollController::getAdditionalXOffset() const {
    return hScrollBar->value() % maEditor->getColumnWidth();
}

int ScrollController::getAdditionalYOffset() const {
    const int firstVisibleRowIndex = ui->getRowHeightController()->globalYPositionToRowIndex(vScrollBar->value());
    const int firstVisibleRowOffset = ui->getRowHeightController()->getRowGlobalOffset(firstVisibleRowIndex);
    return vScrollBar->value() - firstVisibleRowOffset;
}

U2Region ScrollController::getHorizontalRangeToDrawIn(int widgetWidth) const {
    return U2Region(hScrollBar->value(), widgetWidth);
}

U2Region ScrollController::getVerticalRangeToDrawIn(int widgetHeight) const {
    return U2Region(vScrollBar->value(), widgetHeight);
}

void ScrollController::zoomHorizontalScrollBarPrivate() {
    CHECK(!maEditor->isAlignmentEmpty(), );
    SignalBlocker signalBlocker(hScrollBar);
    Q_UNUSED(signalBlocker);

    const int previousAlignmentWidth = hScrollBar->maximum() + ui->getSequenceArea()->width();
    const double previousRelation = static_cast<double>(hScrollBar->value()) / previousAlignmentWidth;
    updateHorizontalScrollBarPrivate();
    hScrollBar->setValue(previousRelation * ui->getBaseWidthController()->getTotalAlignmentWidth());
}

void ScrollController::zoomVerticalScrollBarPrivate() {
    CHECK(!maEditor->isAlignmentEmpty(), );
    SignalBlocker signalBlocker(vScrollBar);
    Q_UNUSED(signalBlocker);

    const int previousAlignmentHeight = vScrollBar->maximum() + ui->getSequenceArea()->height();
    const double previousRelation = static_cast<double>(vScrollBar->value()) / previousAlignmentHeight;
    updateVerticalScrollBarPrivate();
    vScrollBar->setValue(previousRelation * ui->getRowHeightController()->getTotalAlignmentHeight());
}

void ScrollController::updateHorizontalScrollBarPrivate() {
    SAFE_POINT(NULL != hScrollBar, "Horizontal scrollbar is not initialized", );
    SignalBlocker signalBlocker(hScrollBar);
    Q_UNUSED(signalBlocker);

    CHECK_EXT(!maEditor->isAlignmentEmpty(), hScrollBar->setVisible(false), );

    const int alignmentLength = maEditor->getAlignmentLen();
    const int columnWidth = maEditor->getColumnWidth();
    const int sequenceAreaWidth = ui->getSequenceArea()->width();

    hScrollBar->setMinimum(0);
    hScrollBar->setMaximum(qMax(0, alignmentLength * columnWidth - sequenceAreaWidth));
    hScrollBar->setSingleStep(columnWidth);
    hScrollBar->setPageStep(sequenceAreaWidth);

    const int numVisibleBases = getLastVisibleBase(sequenceAreaWidth) - getFirstVisibleBase();
    SAFE_POINT(numVisibleBases <= alignmentLength, "Horizontal scrollbar appears unexpectedly: numVisibleBases is too small", );
    hScrollBar->setVisible(numVisibleBases < alignmentLength);
}

void ScrollController::updateVerticalScrollBarPrivate() {
    SAFE_POINT(NULL != vScrollBar, "Vertical scrollbar is not initialized", );
    SignalBlocker signalBlocker(vScrollBar);
    Q_UNUSED(signalBlocker);

    CHECK_EXT(!maEditor->isAlignmentEmpty(), vScrollBar->setVisible(false), );

    const int totalDisplayableSequences = ui->getSequenceArea()->getNumDisplayableSequences();
    const int sequenceAreaHeight = ui->getSequenceArea()->height();
    const int totalAlignmentHeight = ui->getRowHeightController()->getTotalAlignmentHeight();

    vScrollBar->setMinimum(0);
    vScrollBar->setMaximum(qMax(0, totalAlignmentHeight - sequenceAreaHeight));
    vScrollBar->setSingleStep(ui->getRowHeightController()->getSequenceHeight());
    vScrollBar->setPageStep(sequenceAreaHeight);

    const int numVisibleSequences = getLastVisibleRowNumber(sequenceAreaHeight) - getFirstVisibleRowNumber() + 1;
    SAFE_POINT(numVisibleSequences <= totalDisplayableSequences, "Vertical scrollbar appears unexpectedly: numVisibleSequences is too small", );
    vScrollBar->setVisible(numVisibleSequences < totalDisplayableSequences);
}

}   // namespace U2

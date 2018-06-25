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

#include "RowHeightController.h"
#include "ScrollController.h"
#include "ov_msa/MaEditor.h"
#include "ov_msa/MSACollapsibleModel.h"
#include "ov_msa/view_rendering/MaEditorWgt.h"

namespace U2 {

RowHeightController::RowHeightController(MaEditorWgt *maEditorWgt)
    : QObject(maEditorWgt),
      ui(maEditorWgt)
{

}

int RowHeightController::getRowScreenOffset(int rowIndex) const {
    return getRowScreenRange(rowIndex).startPos;
}

int RowHeightController::getRowScreenOffsetByNumber(int rowNumber) const {
    return getRowScreenOffset(ui->getCollapseModel()->mapToRow(rowNumber));
}

int RowHeightController::getRowScreenCenterByNumber(int rowNumber) const {
    return getRowScreenOffsetByNumber(rowNumber) + getRowHeightByNumber(rowNumber) / 2;
}

int RowHeightController::getRowGlobalOffset(int rowIndex) const {
    int rowOffset = 0;
    for (int i = 0; i < rowIndex; i++) {
        rowOffset += getRowHeight(i);
    }
    return rowOffset;
}

int RowHeightController::getRowGlobalOffset(int rowIndex, const QList<int> &rowIndexes) const {
    int rowOffset = 0;
    foreach (const int currentIndex, rowIndexes) {
        if (currentIndex != rowIndex) {
            rowOffset += getRowHeight(currentIndex);
        } else {
            return rowOffset;
        }
    }
    FAIL(false, 0);
}

int RowHeightController::getFirstVisibleRowGlobalOffset(bool countClipped) const {
    return getRowGlobalOffset(ui->getScrollController()->getFirstVisibleRowIndex(countClipped));
}

int RowHeightController::getFirstVisibleRowScreenOffset(bool countClipped) const {
    const int firstVisibleRowGlobalOffset = getFirstVisibleRowGlobalOffset(countClipped);
    return firstVisibleRowGlobalOffset - ui->getScrollController()->getScreenPosition().y();
}

int RowHeightController::getRowHeightByNumber(int rowNumber) const {
    return getRowHeight(ui->getCollapseModel()->mapToRow(rowNumber));
}

int RowHeightController::getRowsHeight(const QList<int> &rowIndexes) const {
    int rowsHeight = 0;
    foreach (int rowIndex, rowIndexes) {
        rowsHeight += getRowHeight(rowIndex);
    }
    return rowsHeight;
}

int RowHeightController::getTotalAlignmentHeight() const {
    return static_cast<int>(getRowsGlobalRange(0, ui->getCollapseModel()->getDisplayableRowsCount()).length);
}

int RowHeightController::getSequenceHeight() const {
    const int fontHeight = QFontMetrics(ui->getEditor()->getFont(), ui).height();
    const float zoomMult = ui->getEditor()->zoomMult;
    return qRound(fontHeight * zoomMult);
}

int RowHeightController::globalYPositionToRowIndex(int y) const {
    return ui->getCollapseModel()->mapToRow(globalYPositionToRowNumber(y));
}

int RowHeightController::globalYPositionToRowNumber(int y) const {
    const int getDisplayableRowsCount = ui->getCollapseModel()->getDisplayableRowsCount();
    int accumulatedHeight = 0;
    for (int i = 0; i < getDisplayableRowsCount; i++) {
        const int rowHeight = getRowHeightByNumber(i);
        if (accumulatedHeight + rowHeight <= y) {
            accumulatedHeight += rowHeight;
        } else {
            return i;
        }
    }
    return -1;
}

int RowHeightController::screenYPositionToRowIndex(int y) const {
    return globalYPositionToRowIndex(y + ui->getScrollController()->getScreenPosition().y());
}

int RowHeightController::screenYPositionToRowNumber(int y) const {
    return globalYPositionToRowNumber(y + ui->getScrollController()->getScreenPosition().y());
}

U2Region RowHeightController::getRowGlobalRange(int rowIndex) const {
    return U2Region(getRowGlobalOffset(rowIndex), getRowHeight(rowIndex));
}

U2Region RowHeightController::getRowGlobalRange(int rowIndex, const QList<int> &rowIndexes) const {
    return U2Region(getRowGlobalOffset(rowIndex, rowIndexes), getRowHeight(rowIndex));
}

U2Region RowHeightController::getRowGlobalRangeByNumber(int rowNumber) const {
    return getRowGlobalRange(ui->getCollapseModel()->mapToRow(rowNumber));
}

U2Region RowHeightController::getRowsGlobalRange(int startRowNumber, int count) const {
    QList<int> rowIndexes;
    for (int i = startRowNumber; i < startRowNumber + count; i++) {
        rowIndexes << ui->getCollapseModel()->mapToRow(i);
    }
    return getRowsGlobalRange(rowIndexes);
}

U2Region RowHeightController::getRowsGlobalRange(const QList<int> &rowIndexes) const {
    CHECK(!rowIndexes.isEmpty(), U2Region());
    int length = 0;
    foreach (const int rowIndex, rowIndexes) {
        length += getRowHeight(rowIndex);
    }
    return U2Region(getRowGlobalRange(rowIndexes.first()).startPos, length);
}

U2Region RowHeightController::getRowScreenRange(int rowIndex) const {
    return getRowScreenRange(rowIndex, ui->getScrollController()->getScreenPosition().y());
}

U2Region RowHeightController::getRowScreenRange(int rowIndex, const QList<int> &rowIndexes, int screenYOrigin) const {
    const U2Region rowGlobalRange = getRowGlobalRange(rowIndex, rowIndexes);
    return U2Region(rowGlobalRange.startPos - screenYOrigin, rowGlobalRange.length);
}

U2Region RowHeightController::getRowScreenRange(int rowIndex, int screenYOrigin) const {
    const U2Region rowRange = getRowGlobalRange(rowIndex);
    return U2Region(rowRange.startPos - screenYOrigin, rowRange.length);
}

U2Region RowHeightController::getRowScreenRangeByNumber(int rowNumber) const {
    const int screenYOrigin = ui->getScrollController()->getScreenPosition().y();
    return getRowScreenRange(ui->getCollapseModel()->mapToRow(rowNumber), screenYOrigin);
}

U2Region RowHeightController::getRowScreenRangeByNumber(int rowNumber, int screenYOrigin) const {
    return getRowScreenRange(ui->getCollapseModel()->mapToRow(rowNumber), screenYOrigin);
}

U2Region RowHeightController::getRowsScreenRangeByNumbers(const U2Region &rowsNumbers) const {
    const QList<int> rowsIndexes = ui->getCollapseModel()->numbersToIndexes(rowsNumbers);
    const U2Region rowsGlobalRange = getRowsGlobalRange(rowsIndexes);
    return U2Region(rowsGlobalRange.startPos - ui->getScrollController()->getScreenPosition().y(), rowsGlobalRange.length);
}

}   // namespace U2

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

#include "BaseWidthController.h"
#include "DrawHelper.h"
#include "RowHeightController.h"
#include "ScrollController.h"
#include "ov_msa/MSACollapsibleModel.h"
#include "ov_msa/view_rendering/MaEditorSelection.h"
#include "ov_msa/view_rendering/MaEditorWgt.h"

namespace U2 {

DrawHelper::DrawHelper(MaEditorWgt *maEditorWgt) :
    ui(maEditorWgt),
    scrollController(maEditorWgt->getScrollController()),
    collapsibleModel(maEditorWgt->getCollapseModel())
{

}

U2Region DrawHelper::getVisibleBases(int widgetWidth, bool countFirstClippedBase, bool countLastClippedBase) const {
    const int firstVisibleBase = scrollController->getFirstVisibleBase(countFirstClippedBase);
    const int lastVisibleBase = scrollController->getLastVisibleBase(widgetWidth, countLastClippedBase);
    return U2Region(firstVisibleBase, lastVisibleBase - firstVisibleBase + 1);
}

U2Region DrawHelper::getVisibleRowsNumbers(int widgetHeight, bool countFirstClippedRow, bool countLastClippedRow) const {
    const int firstVisibleRowNumber = scrollController->getFirstVisibleRowNumber(countFirstClippedRow);
    const int lastVisibleRowNumber= scrollController->getLastVisibleRowNumber(widgetHeight, countLastClippedRow);
    return U2Region(firstVisibleRowNumber, lastVisibleRowNumber - firstVisibleRowNumber + 1);
}

QList<int> DrawHelper::getVisibleRowsIndexes(int widgetHeight, bool countFirstClippedRow, bool countLastClippedRow) const {
    QVector<U2Region> groupedRowsIndexes = getGroupedVisibleRowsIndexes(widgetHeight, countFirstClippedRow, countLastClippedRow);
    QList<int> rowsIndexes;
    foreach (const U2Region &group, groupedRowsIndexes) {
        for (qint64 rowIndex = group.startPos; rowIndex < group.endPos(); rowIndex++) {
            rowsIndexes << static_cast<int>(rowIndex);
        }
    }
    return rowsIndexes;
}

QVector<U2Region> DrawHelper::getGroupedVisibleRowsIndexes(int widgetHeight, bool countFirstClippedRow, bool countLastClippedRow) const {
    const int firstVisibleRowNumber = scrollController->getFirstVisibleRowNumber(countFirstClippedRow);
    const int lastVisibleRowNumber = scrollController->getLastVisibleRowNumber(widgetHeight, countLastClippedRow);
    QVector<U2Region> groupedRowsIndexes;
    collapsibleModel->getVisibleRows(firstVisibleRowNumber, lastVisibleRowNumber, groupedRowsIndexes);
    return groupedRowsIndexes;
}

int DrawHelper::getVisibleBasesCount(int widgetWidth, bool countFirstClippedBase, bool countLastClippedBase) const {
    return getVisibleBases(widgetWidth, countFirstClippedBase,countLastClippedBase).length;
}

int DrawHelper::getVisibleRowsCount(int widgetHeight, bool countFirstClippedRow, bool countLastClippedRow) const {
    return getVisibleRowsIndexes(widgetHeight, countFirstClippedRow,countLastClippedRow).length();
}

QRect DrawHelper::getSelectionScreenRect(const MaEditorSelection &selection) const {
    CHECK(!selection.getRect().isEmpty(), QRect());

    const U2Region xRange = ui->getBaseWidthController()->getBasesScreenRange(selection.getXRegion());
    CHECK(!xRange.isEmpty(), QRect());

    const U2Region yRange = ui->getRowHeightController()->getRowsScreenRangeByNumbers(selection.getYRegion());
    CHECK(!yRange.isEmpty(), QRect());

    return QRect(xRange.startPos, yRange.startPos, xRange.length - 1, yRange.length - 1);
}

}   // namespace U2

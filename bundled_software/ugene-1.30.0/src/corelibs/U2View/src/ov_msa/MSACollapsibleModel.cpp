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

#include "MSACollapsibleModel.h"
#include "MSAEditor.h"

#include <U2Core/Log.h>
#include <U2Core/MultipleSequenceAlignmentObject.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2Region.h>


namespace U2 {

//////////////////////////////////////////////////////////////////////////
/// MSACollapsableItem
//////////////////////////////////////////////////////////////////////////

MSACollapsableItem::MSACollapsableItem()
    : row(-1), numRows(-1), isCollapsed(false)
{

}

MSACollapsableItem::MSACollapsableItem(int startPos, int length)
    : row(startPos), numRows(length), isCollapsed(false)
{

}

bool MSACollapsableItem::isValid() const {
    return row != -1 && numRows != -1;
}

//////////////////////////////////////////////////////////////////////////
/// MSACollapsibleItemModel
//////////////////////////////////////////////////////////////////////////

MSACollapsibleItemModel::MSACollapsibleItemModel(MaEditorWgt *p)
    : QObject(p),
      ui(p),
      trivialGroupsPolicy(Allow),
      fakeModel(false)
{

}

void MSACollapsibleItemModel::reset(const QVector<U2Region>& itemRegions) {
    items.clear();
    positions.clear();
    foreach(const U2Region& r, itemRegions) {
        if (r.length < 1 || (r.length == 1 && trivialGroupsPolicy == Forbid)) {
            continue;
        }
        items.append(MSACollapsableItem(r.startPos, r.length));
        positions.append(r.startPos);
    }
    collapseAll(true);
}

void MSACollapsibleItemModel::reset() {
    const bool modelExists = ( !items.isEmpty( ) || !positions.isEmpty( ) );
    if ( modelExists ) {
        emit si_aboutToBeToggled();
        items.clear( );
        positions.clear( );
        emit si_toggled( );
    }
}

void MSACollapsibleItemModel::collapseAll(bool collapse) {
    emit si_aboutToBeToggled();
    int delta = 0;
    for (int i=0; i < items.size(); i++) {
        MSACollapsableItem& item = items[i];
        positions[i] = item.row - delta;
        item.isCollapsed = collapse;
        if (collapse) {
            delta += item.numRows - 1;
        }
    }
    emit si_toggled();
}

void MSACollapsibleItemModel::toggle(int pos) {
    emit si_aboutToBeToggled();
    QVector<int>::ConstIterator i = qBinaryFind(positions, pos);
    assert(i != positions.constEnd());
    int index = i - positions.constBegin();
    triggerItem(index);
    emit si_toggled();
}

void MSACollapsibleItemModel::triggerItem(int index) {
    MSACollapsableItem& item = items[index];
    item.isCollapsed = !item.isCollapsed;
    int delta = item.numRows - 1;
    CHECK(delta != 0, );
    assert(delta > 0);
    if (item.isCollapsed) {
        delta *= -1;
    }
    for (int j = index + 1; j < items.size(); j++ ) {
        positions[j] += delta;
    }
}

int MSACollapsibleItemModel::mapToRow(int lastItem, int pos) const {
    const MSACollapsableItem& item = items.at(lastItem);
    int row = item.row + pos - positions.at(lastItem);
    if (item.isCollapsed) {
        row += item.numRows - 1;
    }
    return row;
}

int MSACollapsibleItemModel::mapToRow(int pos) const {
    QVector<int>::ConstIterator i = qLowerBound(positions, pos);
    int idx = i - positions.constBegin() - 1;
    if (idx < 0) {
        return pos;
    } else {
        return mapToRow(idx, pos);
    }
}

U2Region MSACollapsibleItemModel::mapToRows(int pos) const {
    QVector<int>::ConstIterator i = qLowerBound(positions, pos);

    int idx = i - positions.constBegin();

    if (i < positions.constEnd() && *i == pos) {
        // 'pos' is top row in group
        const MSACollapsableItem& item = items.at(idx);
        if (item.isCollapsed) {
            return U2Region(item.row, item.numRows);
        }
        return U2Region(item.row, 1);
    }

    --idx;
    int startPos = 0;
    if (idx < 0) {
        startPos = pos;
    } else {
        startPos = mapToRow(idx, pos);
    }
    return U2Region(startPos, 1);
}

U2Region MSACollapsibleItemModel::mapSelectionRegionToRows(const U2Region& selectionRegion) const {
    if (selectionRegion.isEmpty()) {
        return U2Region();
    }

    if (!ui->isCollapsibleMode()) {
        return selectionRegion;
    }

    int startPos = selectionRegion.startPos;
    int endPos = startPos + selectionRegion.length - 1;

    int startSeq = 0;
    int endSeq = 0;

    int startItemIdx = itemForRow(startPos);

    if (startItemIdx >= 0) {
        const MSACollapsableItem& startItem = getItem(startItemIdx);
        startSeq = startItem.row;
    } else {
        startSeq = mapToRow(startPos);
    }

    int endItemIdx = itemForRow(endPos);

    if (endItemIdx >= 0) {
        const MSACollapsableItem& endItem = getItem(endItemIdx);
        endSeq = endItem.row + endItem.numRows;
    } else {
        endSeq = mapToRow(endPos) + 1;
    }

    return U2Region(startSeq, endSeq - startSeq);
}

QList<int> MSACollapsibleItemModel::numbersToIndexes(const U2Region &rowNumbers) {
    QList<int> rowsIndexes;
    for (int i = rowNumbers.startPos; i < rowNumbers.endPos(); i++) {
        rowsIndexes << mapToRow(i);
    }
    return rowsIndexes;
}

QList<int> MSACollapsibleItemModel::getDisplayableRowsIndexes() const {
    QList<int> displayableRowsIndexes;
    for (int rowNumber = 0; rowNumber < getDisplayableRowsCount(); rowNumber++) {
        displayableRowsIndexes << mapToRow(rowNumber);
    }
    return displayableRowsIndexes;
}

int MSACollapsibleItemModel::rowToMap(int rowIndex, bool failIfNotVisible) const {
    int invisibleRows = 0;
    for (QVector<MSACollapsableItem>::ConstIterator it = items.constBegin(); it < items.constEnd() && it->row < rowIndex; it++) {
        if (it->isCollapsed) {
            if (it->row + it->numRows > rowIndex && failIfNotVisible) {
                return -1;
            }
            invisibleRows += (it->row + it->numRows <= rowIndex) ? it->numRows - 1 : rowIndex - it->row;
        }
    }
    return rowIndex - invisibleRows;
}

void MSACollapsibleItemModel::getVisibleRows(int startPos, int endPos, QVector<U2Region>& range) const {
    if (items.isEmpty()) {
        CHECK(0 <= startPos && 0 <= endPos && startPos <= endPos, );
        range.append(U2Region(startPos, endPos - startPos + 1));
        return;
    }
    QVector<int>::ConstIterator i = qLowerBound(positions, startPos);
    int idx = i - positions.constBegin() - 1;
    int start = 0;
    if (idx < 0) {
        start = startPos;
    } else {
        start = mapToRow(idx, startPos);
    }

    int j = i - positions.constBegin();
    for (; j < items.size(); j++) {
        const MSACollapsableItem& item = items.at(j);
        if (positions[j] > endPos)
            break;
        if (item.isCollapsed) {
            range.append(U2Region(start, item.row - start + 1));
            start = item.row + item.numRows;
        }
    }

    int lastRow = 0;
    if (j - 1 < 0) {
        lastRow = endPos;
    } else {
        lastRow = mapToRow(j - 1, endPos);
    }

    MaEditor* ed = ui->getEditor();
    MultipleAlignmentObject* obj = ed->getMaObject();
    int alnNumRows = obj->getNumRows();
    lastRow = qMin(lastRow, alnNumRows - 1);
    int len = lastRow - start + 1;
    if (len>0) {
        range.append(U2Region(start, len));
    }
}


bool MSACollapsibleItemModel::isTopLevel(int rowNumber) const {
    QVector<int>::ConstIterator i = qBinaryFind(positions, rowNumber);
    if (i == positions.constEnd()) {
        return false;
    }
    return true;
}

bool MSACollapsibleItemModel::isRowInGroup(int rowNumber) const {
    return itemForRow(rowNumber) >= 0;
}

bool MSACollapsibleItemModel::isItemCollapsed(int rowIndex) const {
    const MSACollapsableItem item = ui->getCollapseModel()->getItemByRowIndex(rowIndex);
    return item.isValid() && item.isCollapsed;
}

bool MSACollapsibleItemModel::isRowVisible(int rowIndex) const {
    return isTopLevel(rowToMap(rowIndex, true)) || !isItemCollapsed(rowIndex);
}

int MSACollapsibleItemModel::itemForRow(int rowNumber) const {
    QVector<int>::ConstIterator i = qLowerBound(positions, rowNumber);

    if (i < positions.constEnd() && *i == rowNumber) {
        return i - positions.constBegin();
    }

    int closestItem = i - positions.constBegin() - 1;
    if (closestItem < 0) {
        return -1;
    }

    const MSACollapsableItem& item = items.at(closestItem);
    if (item.isCollapsed) {
        return -1;
    } else {
        int itBottom = positions.at(closestItem) + item.numRows - 1;
        if (rowNumber <= itBottom) {
            return closestItem;
        }
        return -1;
    }
}

int MSACollapsibleItemModel::getItemPos(int index) const {
    return positions.at(index);
}

MSACollapsableItem MSACollapsibleItemModel::getItem(int index) const {
    return items.at(index);
}

MSACollapsableItem MSACollapsibleItemModel::getItemByRowIndex(int rowIndex) const {
    const int itemNumber = itemForRow(rowToMap(rowIndex));
    CHECK(0 <= itemNumber && itemNumber < items.size(), MSACollapsableItem());
    return items[itemNumber];
}

int MSACollapsibleItemModel::getDisplayableRowsCount() const {
    MaEditor *ed = ui->getEditor();
    MultipleAlignmentObject *o = ed->getMaObject();
    int size = o->getNumRows();
    foreach (const MSACollapsableItem &item, items) {
        if (item.isCollapsed) {
            size -= item.numRows - 1;
        }
    }
    return size;
}

void MSACollapsibleItemModel::removeCollapsedForPosition(int index) {
    for (int i = 0, n = items.size(); i < n; ++i) {
        MSACollapsableItem &collapsedItem = items[i];

        int itemStart = collapsedItem.row;
        int itemEnd = itemStart + collapsedItem.numRows;

        if ((index >= itemStart) && (index < itemEnd)) {
            items.remove(i);

            int positionIndex = positions.indexOf(itemStart);
            positions.remove(positionIndex);
        }
    }
}

bool MSACollapsibleItemModel::isEmpty() const {
    return items.isEmpty();
}

void MSACollapsibleItemModel::setTrivialGroupsPolicy(TrivialGroupsPolicy policy) {
    trivialGroupsPolicy = policy;
}

void MSACollapsibleItemModel::setFakeCollapsibleModel(bool fakeModelStatus) {
    fakeModel = fakeModelStatus;
}

bool MSACollapsibleItemModel::isFakeModel() const {
    return fakeModel;
}

int MSACollapsibleItemModel::getItemSize() const {
    return items.size();
}

} // namespace U2

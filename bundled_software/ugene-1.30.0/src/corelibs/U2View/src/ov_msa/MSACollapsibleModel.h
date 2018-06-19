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

#ifndef _U2_MSA_COLLAPSIBLE_MODEL_H_
#define _U2_MSA_COLLAPSIBLE_MODEL_H_

#include <QObject>
#include <QVector>

#include <U2Core/MultipleSequenceAlignment.h>

namespace U2 {

#define MODIFIER "modifier"
#define MAROW_SIMILARITY_SORT "marow_similarity_sort"

class MSACollapsableItem {
public:
    MSACollapsableItem();
    MSACollapsableItem(int startPos, int length);

    bool isValid() const;

    int row;
    int numRows;
    bool isCollapsed;
};

class MaEditorWgt;
class MaModificationInfo;
class U2Region;

class U2VIEW_EXPORT MSACollapsibleItemModel : public QObject {
    Q_OBJECT
public:
    enum TrivialGroupsPolicy {
        Allow,
        Forbid
    };

    MSACollapsibleItemModel(MaEditorWgt *p);

    // creates model with every item collapsed
    // 'itemRegions' has to be sorted list of non-intersecting regions
    void reset(const QVector<U2Region> &itemRegions);

    void reset();

    void toggle(int pos);

    void collapseAll(bool collapse);

    int mapToRow(int pos) const;

    U2Region mapToRows(int pos) const;

    U2Region mapSelectionRegionToRows(const U2Region &selectionRegion) const;
    QList<int> numbersToIndexes(const U2Region &rowNumbers);        // invisible rows are not included to the result list
    QList<int> getDisplayableRowsIndexes() const;

    /**
    * The method converts the row position in the whole msa into its "visible" position (i.e.
    * the row position that takes into account collapsed items).
    * Returns -1 if the row is inside a collapsed item and @failIfNotVisible is true.
    */
    int rowToMap(int rowIndex, bool failIfNotVisible = false) const;

    /**
     * Returns rows indexes that are visible (that are not collapsed) grouped corresponding to the collapsing model
     * for the positions between @startPos and @endPos.
     */
    void getVisibleRows(int startPos, int endPos, QVector<U2Region> &rows) const;

    bool isTopLevel(int rowNumber) const;
    bool isRowInGroup(int rowNumber) const;
    bool isItemCollapsed(int rowIndex) const;
    bool isRowVisible(int rowIndex) const;

    /**
     * Returns the item which contains the row defined by @rowNumber
     * or -1, if the row is not in a collapsing group
     */
    int itemForRow(int rowNumber) const;

    int getItemPos(int index) const;

    MSACollapsableItem getItem(int index) const;
    MSACollapsableItem getItemByRowIndex(int rowIndex) const;

    /**
     * Returns count of rows that can be viewed (that are not collapsed).
     * Every group has at least one row to view.
     */
    int getDisplayableRowsCount() const;

    /** If there is a collapsible item at 'pos' position, it is removed. */
    void removeCollapsedForPosition(int pos);

    bool isEmpty() const;

    void setTrivialGroupsPolicy(TrivialGroupsPolicy policy);

    void setFakeCollapsibleModel(bool fakeModel);

    bool isFakeModel() const;

    int getItemSize() const;

signals:
    void si_aboutToBeToggled();
    void si_toggled();

private:
    void triggerItem(int index);
    int mapToRow(int lastItem, int pos) const;

private:
    MaEditorWgt* ui;
    QVector<MSACollapsableItem> items;
    QVector<int> positions;
    TrivialGroupsPolicy trivialGroupsPolicy;
    bool fakeModel;
};

} //namespace

#endif

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

#ifndef _U2_ROW_HEIGHT_CONTROLLER_H_
#define _U2_ROW_HEIGHT_CONTROLLER_H_

#include <U2Core/U2Region.h>

namespace U2 {

class MaEditorWgt;

class U2VIEW_EXPORT RowHeightController : public QObject{
    Q_OBJECT
public:
    RowHeightController(MaEditorWgt *ui);

    int getRowScreenOffset(int rowIndex) const;
    int getRowScreenOffsetByNumber(int rowNumber) const;

    int getRowScreenCenterByNumber(int rowNumber) const;

    int getRowGlobalOffset(int rowIndex) const;
    int getRowGlobalOffset(int rowIndex, const QList<int> &rowIndexes) const;

    int getFirstVisibleRowGlobalOffset(bool countClipped) const;
    int getFirstVisibleRowScreenOffset(bool countClipped) const;

    virtual int getRowHeight(int rowIndex) const = 0;
    int getRowHeightByNumber(int rowNumber) const;
    int getRowsHeight(const QList<int> &rowIndexes) const;

    U2Region getRowGlobalRange(int rowIndex) const;
    U2Region getRowGlobalRange(int rowIndex, const QList<int> &rowIndexes) const;
    U2Region getRowGlobalRangeByNumber(int rowNumber) const;

    U2Region getRowsGlobalRange(int startRowNumber, int count) const;
    U2Region getRowsGlobalRange(const QList<int> &rowIndexes) const;

    U2Region getRowScreenRange(int rowIndex) const;
    U2Region getRowScreenRange(int rowIndex, const QList<int> &rowIndexes, int screenYOrigin) const;
    U2Region getRowScreenRange(int rowIndex, int screenYOrigin) const;
    U2Region getRowScreenRangeByNumber(int rowNumber) const;
    U2Region getRowScreenRangeByNumber(int rowNumber, int screenYOrigin) const;

    U2Region getRowsScreenRangeByNumbers(const U2Region &rowsNumbers) const;

    int getTotalAlignmentHeight() const;
    int getSequenceHeight() const;

    int globalYPositionToRowIndex(int y) const;
    int globalYPositionToRowNumber(int y) const;
    int screenYPositionToRowIndex(int y) const;
    int screenYPositionToRowNumber(int y) const;

protected:
    MaEditorWgt *ui;
};

}   // namespace U2

#endif // _U2_ROW_HEIGHT_CONTROLLER_H_

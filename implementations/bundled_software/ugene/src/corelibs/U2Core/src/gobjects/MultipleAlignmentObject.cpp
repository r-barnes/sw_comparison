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

#include <U2Core/DbiConnection.h>
#include <U2Core/DocumentModel.h>
#include <U2Core/GHints.h>
#include <U2Core/MsaDbiUtils.h>
#include <U2Core/MSAUtils.h>
#include <U2Core/U2AlphabetUtils.h>
#include <U2Core/U2DbiUtils.h>
#include <U2Core/U2ObjectDbi.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2SafePoints.h>

#include "MultipleAlignmentObject.h"

namespace U2 {

MaSavedState::MaSavedState()
    : lastState(NULL)
{

}

MaSavedState::~MaSavedState() {
    delete lastState;
}

bool MaSavedState::hasState() const {
    return NULL != lastState;
}

const MultipleAlignment MaSavedState::takeState() {
    const MultipleAlignment state = *lastState;
    delete lastState;
    lastState = NULL;
    return state;
}

void MaSavedState::setState(const MultipleAlignment &ma) {
    if (NULL != lastState) {
        delete lastState;
    }
    lastState = new MultipleAlignment(ma->getCopy());
}

MultipleAlignmentObject::MultipleAlignmentObject(const QString &gobjectType,
                                                 const QString &name,
                                                 const U2EntityRef &maRef,
                                                 const QVariantMap &hintsMap,
                                                 const MultipleAlignment &alignment)
    : GObject(gobjectType, name, hintsMap),
      cachedMa(alignment->getCopy())
{
    entityRef = maRef;
    dataLoaded = false;

    if (!cachedMa->isEmpty()) {
        dataLoaded = true;
    }
}

MultipleAlignmentObject::~MultipleAlignmentObject(){
    emit si_invalidateAlignmentObject();
}

void MultipleAlignmentObject::setTrackMod(U2OpStatus &os, U2TrackModType trackMod) {
    // Prepare the connection
    DbiConnection con(entityRef.dbiRef, os);
    CHECK_OP(os, );

    U2ObjectDbi *objectDbi = con.dbi->getObjectDbi();
    SAFE_POINT(NULL != objectDbi, "NULL Object Dbi", );

    // Set the new status
    objectDbi->setTrackModType(entityRef.entityId, trackMod, os);
}

const MultipleAlignment& MultipleAlignmentObject::getMultipleAlignment() const {
    ensureDataLoaded();
    return cachedMa;
}

void MultipleAlignmentObject::setMultipleAlignment(const MultipleAlignment &newMa, MaModificationInfo mi, const QVariantMap &hints) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    U2OpStatus2Log os;
    updateDatabase(os, newMa);
    SAFE_POINT_OP(os, );

    mi.hints = hints;
    updateCachedMultipleAlignment(mi);
}

const MultipleAlignment MultipleAlignmentObject::getMultipleAlignmentCopy() const {
    return getMultipleAlignment()->getCopy();
}

void MultipleAlignmentObject::setGObjectName(const QString &newName) {
    ensureDataLoaded();
    CHECK(cachedMa->getName() != newName, );

    if (!isStateLocked()) {
        U2OpStatus2Log os;
        MaDbiUtils::renameMa(entityRef, newName, os);
        CHECK_OP(os, );

        updateCachedMultipleAlignment();
    } else {
        GObject::setGObjectName(newName);
        cachedMa->setName(newName);
    }
}

const DNAAlphabet* MultipleAlignmentObject::getAlphabet() const {
    return getMultipleAlignment()->getAlphabet();
}

qint64 MultipleAlignmentObject::getLength() const {
    return getMultipleAlignment()->getLength();
}

qint64 MultipleAlignmentObject::getNumRows() const {
    return getMultipleAlignment()->getNumRows();
}

const MultipleAlignmentRow MultipleAlignmentObject::getRow(int row) const {
    return getMultipleAlignment()->getRow(row);
}

int MultipleAlignmentObject::getRowPosById(qint64 rowId) const {
    return getMultipleAlignment()->getRowsIds().indexOf(rowId);
}

U2MsaMapGapModel MultipleAlignmentObject::getMapGapModel() const {
    return getMultipleAlignment()->getMapGapModel();
}

U2MsaListGapModel MultipleAlignmentObject::getGapModel() const {
    return getMultipleAlignment()->getGapModel();
}

void MultipleAlignmentObject::removeRow(int rowIdx) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    const MultipleAlignment &ma = getMultipleAlignment();
    SAFE_POINT(rowIdx >= 0 && rowIdx < ma->getNumRows(), "Invalid row index", );
    qint64 rowId = ma->getRow(rowIdx)->getRowId();

    U2OpStatus2Log os;
    removeRowPrivate(os, entityRef, rowId);
    SAFE_POINT_OP(os, );

    MaModificationInfo mi;
    mi.rowContentChanged = false;
    mi.alignmentLengthChanged = false;

    QList<qint64> removedRowIds;
    removedRowIds << rowId;

    updateCachedMultipleAlignment(mi, removedRowIds);
}

void MultipleAlignmentObject::renameRow(int rowIdx, const QString &newName) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    const MultipleAlignment &ma = getMultipleAlignment();
    SAFE_POINT(rowIdx >= 0 && rowIdx < ma->getNumRows(), "Invalid row index", );
    qint64 rowId = ma->getRow(rowIdx)->getRowId();

    U2OpStatus2Log os;
    MaDbiUtils::renameRow(entityRef, rowId, newName, os);
    SAFE_POINT_OP(os, );

    MaModificationInfo mi;
    mi.alignmentLengthChanged = false;
    updateCachedMultipleAlignment(mi);
}

bool MultipleAlignmentObject::isRegionEmpty(int startPos, int startRow, int numChars, int numRows) const {
    const MultipleAlignment ma = getMultipleAlignment();
    bool isBlockEmpty = true;
    for (int row = startRow; row < startRow + numRows && isBlockEmpty; ++row) {
        for (int pos = startPos; pos < startPos + numChars; ++pos) {
            if (!ma->isGap(row, pos)) {
                isBlockEmpty = false;
                break;
            }
        }
    }
    return isBlockEmpty;
}

void MultipleAlignmentObject::moveRowsBlock(int firstRow, int numRows, int shift) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    QList<qint64> rowIds = getMultipleAlignment()->getRowsIds();
    QList<qint64> rowsToMove;

    for (int i = 0; i < numRows; ++i) {
        rowsToMove << rowIds[firstRow + i];
    }

    U2OpStatusImpl os;
    MaDbiUtils::moveRows(entityRef, rowsToMove, shift, os);
    CHECK_OP(os, );

    updateCachedMultipleAlignment();
}

void MultipleAlignmentObject::updateRowsOrder(U2OpStatus &os, const QList<qint64> &rowIds) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    MaDbiUtils::updateRowsOrder(entityRef, rowIds, os);
    CHECK_OP(os, );

    MaModificationInfo mi;
    mi.alignmentLengthChanged = false;
    updateCachedMultipleAlignment(mi);
}

void MultipleAlignmentObject::changeLength(U2OpStatus &os, qint64 newLength) {
    const qint64 length = getLength();
    CHECK(length != newLength, );

    MaDbiUtils::updateMaLength(getEntityRef(), newLength, os);
    CHECK_OP(os, );

    bool rowContentChangeStatus = false;
    if (newLength < length) {
        const qint64 numRows = getNumRows();
        for (int i = 0; i < numRows; i++) {
            MultipleAlignmentRow row = getRow(i);
            qint64 rowLengthWithoutTrailing = row->getRowLengthWithoutTrailing();
            if (rowLengthWithoutTrailing > newLength) {
                U2OpStatus2Log os;
                row->crop(os, 0, newLength);
                rowContentChangeStatus = true;
            }
        }
    }

    MaModificationInfo modificationInfo;
    modificationInfo.rowContentChanged = rowContentChangeStatus;
    modificationInfo.rowListChanged = false;
    updateCachedMultipleAlignment(modificationInfo);
}

void MultipleAlignmentObject::updateCachedMultipleAlignment(const MaModificationInfo &mi, const QList<qint64> &removedRowIds) {
    ensureDataLoaded();
    emit si_startMaUpdating();

    MultipleAlignment maBefore = cachedMa->getCopy();
    QString oldName = maBefore->getName();

    U2OpStatus2Log os;

    if (mi.alignmentLengthChanged) {
        qint64 msaLength = MaDbiUtils::getMaLength(entityRef, os);
        SAFE_POINT_OP(os, );
        if (msaLength != cachedMa->getLength()) {
            cachedMa->setLength(msaLength);
        }
    }

    if (mi.alphabetChanged) {
        U2AlphabetId alphabet = MaDbiUtils::getMaAlphabet(entityRef, os);
        SAFE_POINT_OP(os, );
        if (alphabet.id != cachedMa->getAlphabet()->getId() && !alphabet.id.isEmpty()) {
            const DNAAlphabet *newAlphabet = U2AlphabetUtils::getById(alphabet);
            cachedMa->setAlphabet(newAlphabet);
        }
    }

     if (mi.modifiedRowIds.isEmpty() && removedRowIds.isEmpty()) { // suppose that in this case all the alignment has changed
        loadAlignment(os);
        SAFE_POINT_OP(os, );
    } else { // only specified rows were changed
        if (!removedRowIds.isEmpty()) {
            foreach (qint64 rowId, removedRowIds) {
                const int rowIndex = cachedMa->getRowIndexByRowId(rowId, os);
                SAFE_POINT_OP(os, );
                cachedMa->removeRow(rowIndex, os);
                SAFE_POINT_OP(os, );
            }
        }
        if (!mi.modifiedRowIds.isEmpty()) {
            updateCachedRows(os, mi.modifiedRowIds);
        }
    }

    setModified(true);
    if (!mi.middleState) {
        emit si_alignmentChanged(maBefore, mi);

        if (cachedMa->isEmpty() && !maBefore->isEmpty()) {
            emit si_alignmentBecomesEmpty(true);
        } else if (!cachedMa->isEmpty() && maBefore->isEmpty()) {
            emit si_alignmentBecomesEmpty(false);
        }

        const QString newName = cachedMa->getName();
        if (oldName != newName) {
            setGObjectNameNotDbi(newName);
        }
    }
    if (!removedRowIds.isEmpty()) {
        emit si_rowsRemoved(removedRowIds);
    }
    if (cachedMa->getAlphabet()->getId() != maBefore->getAlphabet()->getId()) {
        emit si_alphabetChanged(mi, maBefore->getAlphabet());
    }
}

void MultipleAlignmentObject::sortRowsByList(const QStringList &order) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    MultipleSequenceAlignment ma = getMultipleAlignment()->getCopy();
    ma->sortRowsByList(order);
    CHECK(ma->getRowsIds() != cachedMa->getRowsIds(), );

    U2OpStatusImpl os;
    MaDbiUtils::updateRowsOrder(entityRef, ma->getRowsIds(), os);
    SAFE_POINT_OP(os, );

    MaModificationInfo mi;
    mi.alignmentLengthChanged = false;
    mi.rowContentChanged = false;
    mi.rowListChanged = false;
    updateCachedMultipleAlignment(mi);
}

void MultipleAlignmentObject::insertGap(const U2Region &rows, int pos, int nGaps, bool collapseTrailingGaps) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );
    const MultipleAlignment &ma = getMultipleAlignment();
    int startSeq = rows.startPos;
    int endSeq = startSeq + rows.length;

    QList<qint64> rowIdsToInsert;
    for (int i = startSeq; i < endSeq; ++i) {
        qint64 rowId = ma->getRow(i)->getRowId();
        rowIdsToInsert.append(rowId);
    }

    U2OpStatus2Log os;
    MsaDbiUtils::insertGaps(entityRef, rowIdsToInsert, pos, nGaps, os, collapseTrailingGaps);
    SAFE_POINT_OP(os, );

    MaModificationInfo mi;
    mi.rowListChanged = false;
    mi.modifiedRowIds = rowIdsToInsert;
    updateCachedMultipleAlignment(mi);
}

namespace {

template<typename T>
inline QList<T> mergeLists(const QList<T> &first, const QList<T> &second) {
    QList<T> result = first;
    foreach (const T &item, second) {
        if (!result.contains(item)) {
            result.append(item);
        }
    }
    return result;
}

QList<qint64> getRowsAffectedByDeletion(const MultipleAlignment &ma, const QList<qint64> &removedRowIds) {
    QList<qint64> rowIdsAffectedByDeletion;
    U2OpStatus2Log os;
    const QList<qint64> maRows = ma->getRowsIds();
    int previousRemovedRowIndex = -1;
    foreach (qint64 removedRowId, removedRowIds) {
        if (-1 != previousRemovedRowIndex) {
            const int currentRemovedRowIndex = ma->getRowIndexByRowId(removedRowId, os);
            SAFE_POINT_OP(os, QList<qint64>());
            SAFE_POINT(currentRemovedRowIndex > previousRemovedRowIndex, "Rows order violation", QList<qint64>());
            const int countOfUnchangedRowsBetween = currentRemovedRowIndex - previousRemovedRowIndex - 1;
            if (0 < countOfUnchangedRowsBetween) {
                for (int middleRowIndex = previousRemovedRowIndex + 1; middleRowIndex < currentRemovedRowIndex; ++middleRowIndex) {
                    rowIdsAffectedByDeletion += maRows[middleRowIndex];
                }
            }
        }
        previousRemovedRowIndex = ma->getRowIndexByRowId(removedRowId, os);
        SAFE_POINT_OP(os, QList<qint64>());
    }
    const int lastDeletedRowIndex = ma->getRowIndexByRowId(removedRowIds.last(), os);
    SAFE_POINT_OP(os, QList<qint64>());
    if (lastDeletedRowIndex < maRows.size() - 1) { // if the last removed row was not in the bottom of the msa
        rowIdsAffectedByDeletion += maRows.mid(lastDeletedRowIndex + 1);
    }
    return rowIdsAffectedByDeletion;
}

}

void MultipleAlignmentObject::removeRegion(int startPos, int startRow, int nBases, int nRows, bool removeEmptyRows, bool track) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", );

    QList<qint64> modifiedRowIds;
    const MultipleAlignment &ma = getMultipleAlignment();
    const QList<MultipleAlignmentRow> &maRows = ma->getRows();
    SAFE_POINT(nRows > 0 && startRow >= 0 && startRow + nRows <= maRows.size() && startPos + nBases <= ma->getLength(), "Invalid parameters", );
    QList<MultipleAlignmentRow>::ConstIterator it = maRows.begin() + startRow;
    QList<MultipleAlignmentRow>::ConstIterator end = it + nRows;
    for (; it != end; it++) {
        modifiedRowIds << (*it)->getRowId();
    }

    U2OpStatus2Log os;
    removeRegionPrivate(os, entityRef, modifiedRowIds, startPos, nBases);
    SAFE_POINT_OP(os, );

    QList<qint64> removedRows;
    if (removeEmptyRows) {
        removedRows = MsaDbiUtils::removeEmptyRows(entityRef, modifiedRowIds, os);
        SAFE_POINT_OP(os, );
        if (!removedRows.isEmpty()) { // suppose that if at least one row in msa was removed then
            // all the rows below it were changed
            const QList<qint64> rowIdsAffectedByDeletion = getRowsAffectedByDeletion(ma, removedRows);
            foreach (qint64 removedRowId, removedRows) { // removed rows ain't need to be update
                modifiedRowIds.removeAll(removedRowId);
            }
            modifiedRowIds = mergeLists(modifiedRowIds, rowIdsAffectedByDeletion);
        }
    }

    if (track || !removedRows.isEmpty()) {
        MaModificationInfo mi;
        mi.modifiedRowIds = modifiedRowIds;
        updateCachedMultipleAlignment(mi, removedRows);
    }

    if (!removedRows.isEmpty()) {
        emit si_rowsRemoved(removedRows);
    }
}

int MultipleAlignmentObject::deleteGap(U2OpStatus &os, const U2Region &rows, int pos, int maxGaps) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", 0);

    const int removingGapColumnCount = getMaxWidthOfGapRegion(os, rows, pos, maxGaps);
    SAFE_POINT_OP(os, 0);
    if (0 == removingGapColumnCount) {
        return 0;
    } else if (removingGapColumnCount < maxGaps) {
        pos += maxGaps - removingGapColumnCount;
    }
    QList<qint64> modifiedRowIds;
    modifiedRowIds.reserve(rows.length);

    MultipleAlignment msa = getMultipleAlignmentCopy();
    // iterate through given rows to update each of them in DB
    for (int rowCount = rows.startPos; rowCount < rows.endPos(); ++rowCount) {
        msa->removeChars(rowCount, pos, removingGapColumnCount, os);
        CHECK_OP(os, 0);

        const MultipleAlignmentRow row = msa->getRow(rowCount);
        MaDbiUtils::updateRowGapModel(entityRef, row->getRowId(), row->getGapModel(), os);
        CHECK_OP(os, 0);
        modifiedRowIds << row->getRowId();
    }
    if (rows.startPos == 0 && rows.length == getNumRows()) {
        // delete columns
        MaDbiUtils::updateMaLength(entityRef, getLength() - removingGapColumnCount, os);
        CHECK_OP(os, 0);
    }

    MaModificationInfo mi;
    mi.rowListChanged = false;
    mi.modifiedRowIds = modifiedRowIds;
    updateCachedMultipleAlignment(mi);
    return removingGapColumnCount;
}

int MultipleAlignmentObject::shiftRegion(int startPos, int startRow, int nBases, int nRows, int shift) {
    SAFE_POINT(!isStateLocked(), "Alignment state is locked", 0);
    SAFE_POINT(!isRegionEmpty(startPos, startRow, nBases, nRows), "Region is empty", 0);
    SAFE_POINT(0 <= startPos && 0 <= startRow && 0 < nBases && 0 < nRows, "Invalid parameters of selected region encountered", 0);
    U2OpStatusImpl os;

    int n = 0;
    if (shift > 0) {
        //if last symbol selected - do not add gaps at the end
        if (!(startPos + nBases == getLength())) {
            // if some trailing gaps are selected --> save them!
            if (startPos + nBases + shift > getLength()) {
                bool increaseAlignmentLen = true;
                for (int i = startRow; i < startRow + nRows; i++) {
                    int rowLen = getRow(i)->getRowLengthWithoutTrailing();
                    if (rowLen >= startPos + nBases + shift) {
                        increaseAlignmentLen = false;
                        break;
                    }
                }
                if (increaseAlignmentLen) {
                    MaDbiUtils::updateMaLength(entityRef, startPos + nBases + shift, os);
                    SAFE_POINT_OP(os, 0);
                    updateCachedMultipleAlignment();
                }
            }
        }

        insertGap(U2Region(startRow, nRows), startPos, shift);
        n = shift;
    } else if (0 < startPos) {
        if (0 > startPos + shift) {
            shift = -startPos;
        }
        n = -deleteGap(os, U2Region(startRow, nRows), startPos + shift, -shift);
        SAFE_POINT_OP(os, 0);
    }
    return n;
}

void MultipleAlignmentObject::saveState(){
    const MultipleAlignment &ma = getMultipleAlignment();
    emit si_completeStateChanged(false);
    savedState.setState(ma);
}

void MultipleAlignmentObject::releaseState() {
    if (!isStateLocked()) {
        emit si_completeStateChanged(true);

        CHECK(savedState.hasState(), );
        MultipleAlignment maBefore = savedState.takeState();
        CHECK(*maBefore != *getMultipleAlignment(), );
        setModified(true);

        MaModificationInfo mi;
        emit si_alignmentChanged(maBefore, mi);

        if (cachedMa->isEmpty() && !maBefore->isEmpty()) {
            emit si_alignmentBecomesEmpty(true);
        } else if (!cachedMa->isEmpty() && maBefore->isEmpty()) {
            emit si_alignmentBecomesEmpty(false);
        }
    }
}

void MultipleAlignmentObject::loadDataCore(U2OpStatus &os) {
    DbiConnection con(entityRef.dbiRef, os);
    Q_UNUSED(con);
    CHECK_OP(os, );
    loadAlignment(os);
}

int MultipleAlignmentObject::getMaxWidthOfGapRegion(U2OpStatus &os, const U2Region &rows, int pos, int maxGaps) {
    const MultipleAlignment &ma = getMultipleAlignment();
    SAFE_POINT_EXT(U2Region(0, ma->getNumRows()).contains(rows) && 0 <= pos && 0 <= maxGaps && ma->getLength() > pos,
                   os.setError("Illegal parameters of the gap region"), 0);

    const int maxRemovedGaps = qBound(0, maxGaps, ma->getLength() - pos);
    // check if there is nothing to remove
    if (0 == maxRemovedGaps) {
        return 0;
    }

    int removingGapColumnCount = maxRemovedGaps;
    bool isRegionInRowTrailingGaps = true;
    // iterate through given rows to determine the width of the continuous gap region
    for (int rowCount = rows.startPos; rowCount < rows.endPos(); ++rowCount) {
        int gapCountInCurrentRow = 0;
        // iterate through current row bases to determine gap count
        while (gapCountInCurrentRow < maxRemovedGaps) {
            if (!ma->isGap(rowCount, pos + maxGaps - gapCountInCurrentRow - 1)) {
                break;
            }
            gapCountInCurrentRow++;
        }

        // determine if the given area intersects a row in the area of trailing gaps
        if (0 != gapCountInCurrentRow && isRegionInRowTrailingGaps) {
            int trailingPosition = pos + maxRemovedGaps - gapCountInCurrentRow;
            if (ma->getLength() != trailingPosition) {
                while (ma->getLength() > trailingPosition && isRegionInRowTrailingGaps) {
                    isRegionInRowTrailingGaps &= ma->isGap(rowCount, trailingPosition);
                    ++trailingPosition;
                }
            }
        } else if (isRegionInRowTrailingGaps) {
            isRegionInRowTrailingGaps = false;
        }

        if (0 == gapCountInCurrentRow) {
            // don't do anything if there is a row without gaps
            return 0;
        }
        removingGapColumnCount = qMin(removingGapColumnCount, gapCountInCurrentRow);
    }

    if (isRegionInRowTrailingGaps) {
        if (rows.length == getNumRows() && rows.startPos == 0) {
            return qMin(getLength() - pos, (qint64)maxGaps);
        } else {
            return 0;
        }
    }

    return removingGapColumnCount;
}

}   // namespace U2

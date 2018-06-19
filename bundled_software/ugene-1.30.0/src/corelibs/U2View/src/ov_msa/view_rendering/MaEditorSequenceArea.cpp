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
#include <QCursor>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QRubberBand>

#include <U2Algorithm/MsaHighlightingScheme.h>
#include <U2Algorithm/MsaColorScheme.h>

#include <U2Core/AppContext.h>
#include <U2Core/BaseDocumentFormats.h>
#include <U2Core/Counter.h>
#include <U2Core/DNAAlphabet.h>
#include <U2Core/L10n.h>
#include <U2Core/MultipleAlignmentObject.h>
#include <U2Core/Settings.h>
#include <U2Core/TextUtils.h>
#include <U2Core/U2Mod.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2SafePoints.h>

#include <U2Gui/GScrollBar.h>
#include <U2Gui/GUIUtils.h>
#include <U2Gui/OptionsPanel.h>

#include <U2View/MSAHighlightingTabFactory.h>

#include "MaEditorSequenceArea.h"
#include "MaEditorWgt.h"
#include "SequenceAreaRenderer.h"
#include "ov_msa/MaEditor.h"
#include "ov_msa/McaEditorWgt.h"
#include "ov_msa/MSACollapsibleModel.h"
#include "ov_msa/helpers/BaseWidthController.h"
#include "ov_msa/helpers/DrawHelper.h"
#include "ov_msa/helpers/RowHeightController.h"
#include "ov_msa/helpers/ScrollController.h"
#include "ov_msa/Highlighting/MsaSchemesMenuBuilder.h"

namespace U2 {

const QChar MaEditorSequenceArea::emDash = QChar(0x2015);

MaEditorSequenceArea::MaEditorSequenceArea(MaEditorWgt *ui, GScrollBar *hb, GScrollBar *vb)
    : editor(ui->getEditor()),
      ui(ui),
      colorScheme(NULL),
      highlightingScheme(NULL),
      shBar(hb),
      svBar(vb),
      editModeAnimationTimer(this),
      prevPressedButton(Qt::NoButton),
      maVersionBeforeShifting(-1),
      useDotsAction(NULL),
      replaceCharacterAction(NULL),
      changeTracker(editor->getMaObject()->getEntityRef())
{
    rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
    maMode = ViewMode;

    setSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
    setMinimumSize(100, 100);
    highlightSelection = false;
    selecting = false;
    shifting = false;
    editingEnabled = false;
    movableBorder = SelectionModificationHelper::NoMovableBorder;
    isCtrlPressed = false;
    lengthOnMousePress = editor->getMaObject()->getLength();

    cachedView = new QPixmap();
    completeRedraw = true;

    useDotsAction = new QAction(QString(tr("Use dots")), this);
    useDotsAction->setCheckable(true);
    useDotsAction->setChecked(false);
    connect(useDotsAction, SIGNAL(triggered()), SLOT(sl_useDots()));

    replaceCharacterAction = new QAction(tr("Replace selected character"), this);
    replaceCharacterAction->setObjectName("replace_selected_character");
    replaceCharacterAction->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_R));
    replaceCharacterAction->setShortcutContext(Qt::WidgetShortcut);
    addAction(replaceCharacterAction);
    connect(replaceCharacterAction, SIGNAL(triggered()), SLOT(sl_replaceSelectedCharacter()));

    fillWithGapsinsSymAction = new QAction(tr("Fill selection with gaps"), this);
    fillWithGapsinsSymAction->setObjectName("fill_selection_with_gaps");
    connect(fillWithGapsinsSymAction, SIGNAL(triggered()), SLOT(sl_fillCurrentSelectionWithGaps()));
    addAction(fillWithGapsinsSymAction);

    QAction* undoAction = ui->getUndoAction();
    QAction* redoAction = ui->getRedoAction();
    addAction(undoAction);
    addAction(redoAction);

    connect(editor, SIGNAL(si_completeUpdate()), SLOT(sl_completeUpdate()));
    connect(editor, SIGNAL(si_zoomOperationPerformed(bool)), SLOT(sl_completeUpdate()));
    connect(editor, SIGNAL(si_updateActions()), SLOT(sl_updateActions()));
    connect(ui, SIGNAL(si_completeRedraw()), SLOT(sl_completeRedraw()));
    connect(hb, SIGNAL(actionTriggered(int)), SLOT(sl_hScrollBarActionPerfermed()));


    // SANGER_TODO: why is it commented?
//    connect(editor, SIGNAL(si_fontChanged(QFont)), SLOT(sl_fontChanged(QFont)));

    connect(&editModeAnimationTimer, SIGNAL(timeout()), SLOT(sl_changeSelectionColor()));

    connect(editor->getMaObject(), SIGNAL(si_alignmentChanged(const MultipleAlignment&, const MaModificationInfo&)),
        SLOT(sl_alignmentChanged(const MultipleAlignment&, const MaModificationInfo&)));
}

MaEditorSequenceArea::~MaEditorSequenceArea() {
    exitFromEditCharacterMode();
    delete cachedView;
    deleteOldCustomSchemes();
    delete highlightingScheme;
}

MaEditor *MaEditorSequenceArea::getEditor() const {
    return editor;
}

QSize MaEditorSequenceArea::getCanvasSize(const QList<int> &seqIdx, const U2Region &region) const {
    return QSize(ui->getBaseWidthController()->getBasesWidth(region), ui->getRowHeightController()->getRowsHeight(seqIdx));
}

int MaEditorSequenceArea::getFirstVisibleBase() const {
    return ui->getScrollController()->getFirstVisibleBase();
}

int MaEditorSequenceArea::getLastVisibleBase(bool countClipped) const {
    return getEditor()->getUI()->getScrollController()->getLastVisibleBase(width(), countClipped);
}

int MaEditorSequenceArea::getNumVisibleBases() const {
    return ui->getDrawHelper()->getVisibleBasesCount(width());
}

int MaEditorSequenceArea::getNumDisplayableSequences() const {
    CHECK(!isAlignmentEmpty(), 0);
    MSACollapsibleItemModel *model = ui->getCollapseModel();
    SAFE_POINT(NULL != model, tr("Invalid collapsible item model!"), -1);
    return model->getDisplayableRowsCount();
}

int MaEditorSequenceArea::getRowIndex(const int num) const {
    CHECK(!isAlignmentEmpty(), -1);
    MSACollapsibleItemModel *model = ui->getCollapseModel();
    SAFE_POINT(NULL != model, tr("Invalid collapsible item model!"), -1);
    return model->mapToRow(num);
}

bool MaEditorSequenceArea::isAlignmentEmpty() const {
    return editor->isAlignmentEmpty();
}

bool MaEditorSequenceArea::isPosInRange(int position) const {
    return position >= 0 && position < editor->getAlignmentLen();
}

bool MaEditorSequenceArea::isSeqInRange(int rowNumber) const {
    return rowNumber >= 0 && rowNumber < getNumDisplayableSequences();
}

bool MaEditorSequenceArea::isInRange(const QPoint &point) const {
    return isPosInRange(point.x()) && isSeqInRange(point.y());
}

QPoint MaEditorSequenceArea::boundWithVisibleRange(const QPoint &point) const {
    return QPoint(qBound(0, point.x(), editor->getAlignmentLen() - 1), qBound(0, point.y(), ui->getCollapseModel()->getDisplayableRowsCount() - 1));
}

bool MaEditorSequenceArea::isVisible(const QPoint& p, bool countClipped) const {
    return isPositionVisible(p.x(), countClipped) && isRowVisible(p.y(), countClipped);
}

bool MaEditorSequenceArea::isPositionVisible(int position, bool countClipped) const {
    return ui->getDrawHelper()->getVisibleBases(width(), countClipped, countClipped).contains(position);
}

bool MaEditorSequenceArea::isRowVisible(int rowNumber, bool countClipped) const {
    const int rowIndex = ui->getCollapseModel()->mapToRow(rowNumber);
    return ui->getDrawHelper()->getVisibleRowsIndexes(height(), countClipped, countClipped).contains(rowIndex);
}

const MaEditorSelection & MaEditorSequenceArea::getSelection() const {
    return selection;
}

void MaEditorSequenceArea::updateSelection(const QPoint& newPos) {
    const int width = qAbs(newPos.x() - cursorPos.x()) + 1;
    const int height = qAbs(newPos.y() - cursorPos.y()) + 1;
    const int left = qMin(newPos.x(), cursorPos.x());
    const int top = qMin(newPos.y(), cursorPos.y());
    const QPoint topLeft = boundWithVisibleRange(QPoint(left, top));
    const QPoint bottomRight = boundWithVisibleRange(QPoint(left + width - 1, top + height - 1));

    MaEditorSelection s(topLeft, bottomRight);
    if (newPos.x() != -1 && newPos.y() != -1) {
        ui->getScrollController()->scrollToPoint(newPos, size());
        setSelection(s);
    }
    bool selectionExists = !selection.isNull();
    ui->getCopySelectionAction()->setEnabled(selectionExists);
    ui->getCopyFormattedSelectionAction()->setEnabled(selectionExists);
    emit si_copyFormattedChanging(selectionExists);
}

void MaEditorSequenceArea::updateSelection() {
    CHECK(!baseSelection.isNull(), );

    if (!ui->isCollapsibleMode()) {
        setSelection(baseSelection);
        return;
    }
    MSACollapsibleItemModel* m = ui->getCollapseModel();
    CHECK_EXT(NULL != m, sl_cancelSelection(), );

    int startPos = baseSelection.y();
    int endPos = startPos + baseSelection.height();

    // convert selected rows indexes to indexes of selected collapsible items
    int newStart = m->rowToMap(startPos);
    int newEnd = m->rowToMap(endPos);

    SAFE_POINT_EXT(newStart >= 0 && newEnd >= 0, sl_cancelSelection(), );

    int selectionHeight = newEnd - newStart;
    // accounting of collapsing children items
    int itemIndex = m->itemForRow(newEnd);
    if (selectionHeight <= 1 && itemIndex >= 0) {
        const MSACollapsableItem& collapsibleItem = m->getItem(itemIndex);
        if(newEnd == collapsibleItem.row && !collapsibleItem.isCollapsed) {
            selectionHeight = qMax(selectionHeight, endPos - newStart + collapsibleItem.numRows);
        }
    }
    if(selectionHeight > 0 && newStart + selectionHeight <= m->getDisplayableRowsCount()) {
        MaEditorSelection s(selection.topLeft().x(), newStart, selection.width(), selectionHeight);
        setSelection(s);
    } else {
        sl_cancelSelection();
    }
}

void MaEditorSequenceArea::setSelection(const MaEditorSelection& s, bool newHighlightSelection) {
    CHECK(!isAlignmentEmpty() || s.isEmpty(), );
    // TODO: assert(isInRange(s));
    exitFromEditCharacterMode();
    if (highlightSelection != newHighlightSelection) {
        highlightSelection = newHighlightSelection;
        update();
    }

    MaEditorSelection prevSelection = selection;
    selection = s;

    if (!selection.isEmpty()) {
        Q_ASSERT(isInRange(selection.topLeft()));
        Q_ASSERT(isInRange(selection.bottomRight()));
        selection = MaEditorSelection(MaEditorSequenceArea::boundWithVisibleRange(selection.topLeft()),
                                      MaEditorSequenceArea::boundWithVisibleRange(selection.bottomRight()));
    }

    int selEndPos = s.x() + s.width() - 1;
    int ofRange = selEndPos - editor->getAlignmentLen();
    if (ofRange >= 0) {
        selection = MaEditorSelection(s.topLeft(), s.width() - ofRange - 1, s.height());
    }

    bool selectionExists = !selection.isNull();
    ui->getCopySelectionAction()->setEnabled(selectionExists);
    ui->getCopyFormattedSelectionAction()->setEnabled(selectionExists);
    emit si_copyFormattedChanging(selectionExists);

    U2Region selectedRowsRegion = getSelectedRows();
    baseSelection = MaEditorSelection(selection.topLeft().x(), getSelectedRows().startPos, selection.width(), selectedRowsRegion.length);

    QStringList selectedRowNames;
    for (int x = selectedRowsRegion.startPos; x < selectedRowsRegion.endPos(); x++) {
        selectedRowNames.append(editor->getMaObject()->getRow(x)->getName());
    }
    emit si_selectionChanged(selectedRowNames);
    emit si_selectionChanged(selection, prevSelection);
    update();
    sl_updateActions();

    CHECK(!selection.isNull(), );
}

void MaEditorSequenceArea::moveSelection(int dx, int dy, bool allowSelectionResize) {
    int leftX = selection.x();
    int topY = selection.y();
    int bottomY = selection.y() + selection.height() - 1;
    int rightX = selection.x() + selection.width() - 1;
    QPoint baseTopLeft(leftX, topY);
    QPoint baseBottomRight(rightX,bottomY);

    QPoint newTopLeft = baseTopLeft + QPoint(dx,dy);
    QPoint newBottomRight = baseBottomRight + QPoint(dx,dy);

    if ((!isInRange(newTopLeft)) || (!isInRange(newBottomRight))) {
        if (!allowSelectionResize) {
            return;
        } else {
            MaEditorSelection newSelection(selection.topLeft(),
                                            qMin(selection.width(), editor->getAlignmentLen() - newTopLeft.x()),
                                            qMin(selection.height(), editor->getNumSequences() - newTopLeft.y()));
            setSelection(newSelection);
        }
    }

    MaEditorSelection newSelection(newTopLeft, selection.width(), selection.height());
    setSelection(newSelection);
    ui->getScrollController()->scrollToMovedSelection(dx, dy);
}

U2Region MaEditorSequenceArea::getSelectedRows() const {
    return ui->getCollapseModel()->mapSelectionRegionToRows(U2Region(selection.y(), selection.height()));
}

QString MaEditorSequenceArea::getCopyFormatedAlgorithmId() const{
    return AppContext::getSettings()->getValue(SETTINGS_ROOT + SETTINGS_COPY_FORMATTED, BaseDocumentFormats::CLUSTAL_ALN).toString();
}

void MaEditorSequenceArea::setCopyFormatedAlgorithmId(const QString& algoId){
    AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_COPY_FORMATTED, algoId);
}


void MaEditorSequenceArea::deleteCurrentSelection() {
    CHECK(getEditor() != NULL, );
    CHECK(!selection.isNull(), );

    assert(isInRange(selection.topLeft()));
    assert(isInRange(QPoint(selection.x() + selection.width() - 1, selection.y() + selection.height() - 1)));
    MultipleAlignmentObject* maObj = getEditor()->getMaObject();
    if (maObj == NULL || maObj->isStateLocked()) {
        return;
    }

    const QRect areaBeforeSelection(0, 0, selection.x(), selection.height());
    const QRect areaAfterSelection(selection.x() + selection.width(), selection.y(),
        maObj->getLength() - selection.x() - selection.width(), selection.height());
    if (maObj->isRegionEmpty(areaBeforeSelection.x(), areaBeforeSelection.y(), areaBeforeSelection.width(), areaBeforeSelection.height())
        && maObj->isRegionEmpty(areaAfterSelection.x(), areaAfterSelection.y(), areaAfterSelection.width(), areaAfterSelection.height())
        && selection.height() == maObj->getNumRows())
    {
        return;
    }

    // if this method was invoked during a region shifting
    // then shifting should be canceled
    cancelShiftTracking();

    U2OpStatusImpl os;
    U2UseCommonUserModStep userModStep(maObj->getEntityRef(), os);
    Q_UNUSED(userModStep);
    SAFE_POINT_OP(os, );

    const U2Region& sel = getSelectedRows();
    const bool isGap = maObj->getRow(selection.topLeft().y())->isGap(selection.topLeft().x());
    maObj->removeRegion(selection.x(), sel.startPos, selection.width(), sel.length, true);
    GRUNTIME_NAMED_COUNTER(cvar, tvar, "Delete current selection", editor->getFactoryId());

    if (selection.height() == 1 && selection.width() == 1) {
        GRUNTIME_NAMED_CONDITION_COUNTER(cvar2, tvar2, isGap, "Remove gap", editor->getFactoryId());
        GRUNTIME_NAMED_CONDITION_COUNTER(cvar3, tvar3, !isGap, "Remove character", editor->getFactoryId());

        if (isInRange(selection.topLeft())) {
            return;
        }
    }
    sl_cancelSelection();
}

bool MaEditorSequenceArea::shiftSelectedRegion(int shift) {
    CHECK(shift != 0, true);

    // shifting of selection
    MultipleAlignmentObject *maObj = editor->getMaObject();
    if (!maObj->isStateLocked()) {
        const U2Region rows = getSelectedRows();
        const int x = selection.x();
        const int y = rows.startPos;
        const int selectionWidth = selection.width();
        const int height = rows.length;
        if (maObj->isRegionEmpty(x, y, selectionWidth, height)) {
            return true;
        }
        // backup current selection for the case when selection might disappear
        const MaEditorSelection selectionBackup = selection;

        const int resultShift = shiftRegion(shift);
        if (0 != resultShift) {
            U2OpStatus2Log os;
            adjustReferenceLength(os);

            int newCursorPosX = (cursorPos.x() + resultShift >= 0) ? cursorPos.x() + resultShift : 0;
            setCursorPos(newCursorPosX);

            const MaEditorSelection newSelection(selectionBackup.x() + resultShift, selectionBackup.y(),
                                                 selectionBackup.width(), selectionBackup.height());
            setSelection(newSelection);
            if (resultShift > 0) {
                ui->getScrollController()->scrollToBase(static_cast<int>(newSelection.getXRegion().endPos() - 1), width());
            } else {
                ui->getScrollController()->scrollToBase(newSelection.x(), width());
            }

            return true;
        } else {
            return false;
        }
    }
    return false;
}

int MaEditorSequenceArea::shiftRegion(int shift) {
    int resultShift = 0;

    MultipleAlignmentObject *maObj = editor->getMaObject();
    const U2Region rows = getSelectedRows();
    const int selectionWidth = selection.width();
    const int height = rows.length;
    const int y = rows.startPos;
    int x = selection.x();
    if (isCtrlPressed) {
        if (shift > 0) {
            QList<U2MsaGap> gapModelToRemove = findRemovableGapColumns(shift);
            if (!gapModelToRemove.isEmpty()) {
                foreach(U2MsaGap gap, gapModelToRemove) {
                    x = selection.x();
                    U2OpStatus2Log os;
                    const int length = maObj->getLength();
                    if (length != gap.offset) {
                        maObj->deleteGap(os, rows, gap.offset, gap.gap);
                    }
                    CHECK_OP(os, resultShift);
                    resultShift += maObj->shiftRegion(x, y, selectionWidth, height, gap.gap);
                    MaEditorSelection newSel(QPoint(gap.gap + x, selection.y()), selectionWidth, height);
                    setSelection(newSel);
                }
            }
        } else if (shift < 0 && !ctrlModeGapModel.isEmpty()) {
            QList<U2MsaGap> gapModelToRestore = findRestorableGapColumns(shift);
            if (!gapModelToRestore.isEmpty()) {
                resultShift = maObj->shiftRegion(x, y, selectionWidth, height, shift);
                foreach(U2MsaGap gap, gapModelToRestore) {
                    if (gap.endPos() < lengthOnMousePress) {
                        maObj->insertGap(rows, gap.offset, gap.gap);
                    } else if (gap.offset >= lengthOnMousePress) {
                        U2OpStatus2Log os;
                        U2Region allRows(0, maObj->getNumRows());
                        maObj->deleteGap(os, allRows, maObj->getLength() - gap.gap, gap.gap);
                        CHECK_OP(os, resultShift);
                    }
                }
            }
        }
    } else {
        resultShift = maObj->shiftRegion(x, y, selectionWidth, height, shift);
    }

    return resultShift;
}

QList<U2MsaGap> MaEditorSequenceArea::findRemovableGapColumns(int& shift) {
    CHECK(shift > 0, QList<U2MsaGap>());

    int numOfRemovableColumns = 0;
    U2MsaRowGapModel commonGapColumns = findCommonGapColumns(numOfRemovableColumns);
    if (numOfRemovableColumns < shift){
        int count = shift - numOfRemovableColumns;
        commonGapColumns << addTrailingGapColumns(count);
    }

    QList<U2MsaGap> gapColumnsToRemove;
    int count = shift;
    foreach(U2MsaGap gap, commonGapColumns) {
        if (count >= gap.gap) {
            gapColumnsToRemove.append(gap);
            count -= gap.gap;
            if (count == 0) {
                break;
            }
        } else {
            gapColumnsToRemove.append(U2MsaGap(gap.offset, count));
            break;
        }
    }

    ctrlModeGapModel << gapColumnsToRemove;

    if (count < shift) {
        shift -= count;
    }
    return gapColumnsToRemove;
}

QList<U2MsaGap> MaEditorSequenceArea::findCommonGapColumns(int& numOfColumns) {
    const U2Region rows = getSelectedRows();
    const int x = selection.x();
    const int wight = selection.width();
    const U2MsaListGapModel listGapModel = editor->getMaObject()->getGapModel();

    U2MsaRowGapModel gapModelToUpdate;
    foreach(U2MsaGap gap, listGapModel[rows.startPos]) {
        if (gap.offset + gap.gap <= x + wight) {
            continue;
        } else if (gap.offset < x + wight && gap.offset + gap.gap > x + wight) {
            int startPos = x + wight;
            U2MsaGap g(startPos, gap.offset + gap.gap - startPos);
            gapModelToUpdate << g;
        } else {
            gapModelToUpdate << gap;
        }
    }

    numOfColumns = 0;
    for (int i = rows.startPos + 1; i < rows.endPos(); i++) {
        U2MsaRowGapModel currentGapModelToRemove;
        int currentNumOfColumns = 0;
        foreach(U2MsaGap gap, listGapModel[i]) {
            foreach(U2MsaGap gapToRemove, gapModelToUpdate) {
                U2MsaGap intersectedGap = gap.intersect(gapToRemove);
                if (intersectedGap.gap == 0) {
                    continue;
                }
                currentNumOfColumns += intersectedGap.gap;
                currentGapModelToRemove << intersectedGap;
            }
        }
        gapModelToUpdate = currentGapModelToRemove;
        numOfColumns = currentNumOfColumns;
    }

    return gapModelToUpdate;
}

U2MsaGap MaEditorSequenceArea::addTrailingGapColumns(int count) {
    MultipleAlignmentObject *maObj = editor->getMaObject();
    qint64 length = maObj->getLength();
    return U2MsaGap(length, count);
}

QList<U2MsaGap> MaEditorSequenceArea::findRestorableGapColumns(const int shift) {
    CHECK(shift < 0, QList<U2MsaGap>());
    CHECK(!ctrlModeGapModel.isEmpty(), QList<U2MsaGap>());

    QList<U2MsaGap> gapColumnsToRestore;
    int absShift = qAbs(shift);
    const int size = ctrlModeGapModel.size();
    for (int i = size - 1; i >= 0; i--) {
        if (ctrlModeGapModel[i].gap >= absShift) {
            const int offset = ctrlModeGapModel[i].gap - absShift;
            U2MsaGap gapToRestore(ctrlModeGapModel[i].offset + offset, absShift);
            gapColumnsToRestore.push_front(gapToRestore);
            ctrlModeGapModel[i].gap -= absShift;
            if (ctrlModeGapModel[i].gap == 0) {
                ctrlModeGapModel.removeOne(ctrlModeGapModel[i]);
            }
            break;
        } else {
            gapColumnsToRestore.push_front(ctrlModeGapModel[i]);
            absShift -= ctrlModeGapModel[i].gap;
            ctrlModeGapModel.removeOne(ctrlModeGapModel[i]);
        }
    }

    return gapColumnsToRestore;
}

void MaEditorSequenceArea::centerPos(const QPoint &point) {
    SAFE_POINT(isInRange(point), QString("Point (%1, %2) is out of range").arg(point.x()).arg(point.y()), );
    ui->getScrollController()->centerPoint(point, size());
    update();
}

void MaEditorSequenceArea::centerPos(int position) {
    SAFE_POINT(isPosInRange(position), QString("Base %1 is out of range").arg(position), );
    ui->getScrollController()->centerBase(position, width());
    update();
}

void MaEditorSequenceArea::onVisibleRangeChanged() {
    exitFromEditCharacterMode();
    CHECK(!isAlignmentEmpty(), );

    const QStringList rowsNames = editor->getMaObject()->getMultipleAlignment()->getRowNames();
    QStringList visibleRowsNames;

    const QList<int> visibleRows =  ui->getDrawHelper()->getVisibleRowsIndexes(height());
    foreach (const int rowIndex, visibleRows) {
        SAFE_POINT(rowIndex < rowsNames.size(), QString("Row index is out of rowsNames boudaries: index is %1, size is %2").arg(rowIndex).arg(rowsNames.size()), );
        visibleRowsNames << rowsNames[rowIndex];
    }

    const int rowsHeight = ui->getRowHeightController()->getRowsHeight(visibleRows);

    emit si_visibleRangeChanged(visibleRowsNames, rowsHeight);
}

bool MaEditorSequenceArea::isAlignmentLocked() const {
    MultipleAlignmentObject* obj = editor->getMaObject();
    SAFE_POINT(NULL != obj, tr("Alignment object is not available"), true);
    return obj->isStateLocked();
}

void MaEditorSequenceArea::drawVisibleContent(QPainter& painter) {
    const U2Region basesToDraw = ui->getDrawHelper()->getVisibleBases(width());
    const QList<int> seqIdx = ui->getDrawHelper()->getVisibleRowsIndexes(height());
    CHECK(!basesToDraw.isEmpty(), );
    CHECK(!seqIdx.isEmpty(), );
    const int xStart = ui->getBaseWidthController()->getBaseScreenRange(basesToDraw.startPos).startPos;
    const int yStart = ui->getRowHeightController()->getRowScreenRange(seqIdx.first()).startPos;
    drawContent(painter, basesToDraw, seqIdx, xStart, yStart);
}

bool MaEditorSequenceArea::drawContent(QPainter &painter, const QRect &area) {
    const int xStart = ui->getBaseWidthController()->getFirstVisibleBaseGlobalOffset(true);
    const int yStart = ui->getRowHeightController()->getFirstVisibleRowGlobalOffset(true);
    return drawContent(painter, area, xStart, yStart);
}

bool MaEditorSequenceArea::drawContent(QPainter &painter, const QRect &area, int xStart, int yStart) {
    QList<int> seqIdx;
    for (int rowNumber = 0; rowNumber < area.height(); rowNumber++) {
        seqIdx << ui->getCollapseModel()->mapToRow(rowNumber);
    }
    bool ok = renderer->drawContent(painter, U2Region(area.x(), area.width()), seqIdx, xStart, yStart);
    emit si_visibleRangeChanged();
    return ok;
}

bool MaEditorSequenceArea::drawContent(QPainter &painter, const U2Region &region, const QList<int> &seqIdx) {
    const int xStart = ui->getBaseWidthController()->getFirstVisibleBaseScreenOffset(true);
    const int yStart = ui->getRowHeightController()->getFirstVisibleRowScreenOffset(true);
    return drawContent(painter, region, seqIdx, xStart, yStart);
}

bool MaEditorSequenceArea::drawContent(QPainter &painter, const U2Region &region, const QList<int> &seqIdx, int xStart, int yStart) {
    // SANGER_TODO: optimize
    return renderer->drawContent(painter, region, seqIdx, xStart, yStart);
}

bool MaEditorSequenceArea::drawContent(QPainter &painter) {
    const QRect areaToDraw = QRect(0, 0, editor->getAlignmentLen(), ui->getCollapseModel()->getDisplayableRowsCount());
    return drawContent(painter, areaToDraw);
}

bool MaEditorSequenceArea::drawContent(QPixmap &pixmap) {
    const int totalAlignmentWidth = ui->getBaseWidthController()->getTotalAlignmentWidth();
    const int totalAlignmentHeight = ui->getRowHeightController()->getTotalAlignmentHeight();
    CHECK(totalAlignmentWidth < 32768 && totalAlignmentHeight < 32768, false);

    pixmap = QPixmap(totalAlignmentWidth, totalAlignmentHeight);
    QPainter p(&pixmap);

    const QRect areaToDraw = QRect(0, 0, editor->getAlignmentLen(), ui->getCollapseModel()->getDisplayableRowsCount());
    return drawContent(p, areaToDraw, 0, 0);
}

bool MaEditorSequenceArea::drawContent(QPixmap &pixmap,
                                       const U2Region &region,
                                       const QList<int> &seqIdx) {
    CHECK(!region.isEmpty(), false);
    CHECK(!seqIdx.isEmpty(), false);

    const int canvasWidth = ui->getBaseWidthController()->getBasesWidth(region);
    const int canvasHeight = ui->getRowHeightController()->getRowsHeight(seqIdx);

    CHECK(canvasWidth < 32768 &&
          canvasHeight < 32768, false);
    pixmap = QPixmap(canvasWidth, canvasHeight);
    QPainter p(&pixmap);
    return drawContent(p, region, seqIdx, 0, 0);
}

void MaEditorSequenceArea::highlightCurrentSelection()  {
    highlightSelection = true;
    update();
}

QString MaEditorSequenceArea::exportHighlighting(int startPos, int endPos, int startingIndex, bool keepGaps, bool dots, bool transpose) {
    CHECK(getEditor() != NULL, QString());
    CHECK(qobject_cast<MSAEditor*>(editor) != NULL, QString());
    SAFE_POINT(editor->getReferenceRowId() != U2MsaRow::INVALID_ROW_ID, "Export highlighting is not supported without a reference", QString());
    QStringList result;

    MultipleAlignmentObject* maObj = editor->getMaObject();
    assert(maObj!=NULL);

    const MultipleAlignment msa = maObj->getMultipleAlignment();

    U2OpStatusImpl os;
    const int refSeq = getEditor()->getMaObject()->getMultipleAlignment()->getRowIndexByRowId(editor->getReferenceRowId(), os);
    SAFE_POINT_OP(os, QString());
    MultipleAlignmentRow row = msa->getRow(refSeq);

    QString header;
    header.append("Position\t");
    QString refSeqName = editor->getReferenceRowName();
    header.append(refSeqName);
    header.append("\t");
    foreach(QString name, maObj->getMultipleAlignment()->getRowNames()){
        if(name != refSeqName){
            header.append(name);
            header.append("\t");
        }
    }
    header.remove(header.length()-1,1);
    result.append(header);

    int posInResult = startingIndex;

    for (int pos = startPos-1; pos < endPos; pos++) {
        QString rowStr;
        rowStr.append(QString("%1").arg(posInResult));
        rowStr.append(QString("\t") + QString(msa->charAt(refSeq, pos)) + QString("\t"));
        bool informative = false;
        for (int seq = 0; seq < msa->getNumRows(); seq++) {  //FIXME possible problems when sequences have moved in view
            if (seq == refSeq) continue;
            char c = msa->charAt(seq, pos);

            const char refChar = row->charAt(pos);
            if (refChar == '-' && !keepGaps) {
                continue;
            }

            QColor unused;
            bool highlight = false;
            highlightingScheme->setUseDots(useDotsAction->isChecked());
            highlightingScheme->process(refChar, c, unused, highlight, pos, seq);

            if (highlight) {
                rowStr.append(c);
                informative = true;
            } else {
                if (dots) {
                    rowStr.append(".");
                } else {
                    rowStr.append(" ");
                }
            }
            rowStr.append("\t");
        }
        if(informative){
            header.remove(rowStr.length() - 1, 1);
            result.append(rowStr);
        }
        posInResult++;
    }

    if (!transpose){
        QStringList transposedRows = TextUtils::transposeCSVRows(result, "\t");
        return transposedRows.join("\n");
    }

    return result.join("\n");
}

MsaColorScheme * MaEditorSequenceArea::getCurrentColorScheme() const {
    return colorScheme;
}

MsaHighlightingScheme * MaEditorSequenceArea::getCurrentHighlightingScheme() const {
    return highlightingScheme;
}

bool MaEditorSequenceArea::getUseDotsCheckedState() const {
    return useDotsAction->isChecked();
}

QAction *MaEditorSequenceArea::getReplaceCharacterAction() const {
    return replaceCharacterAction;
}

void MaEditorSequenceArea::sl_changeColorSchemeOutside(const QString &id) {
    QAction* a = GUIUtils::findActionByData(QList<QAction*>() << colorSchemeMenuActions << customColorSchemeMenuActions << highlightingSchemeMenuActions, id);
    if (a != NULL) {
        a->trigger();
    }
}

void MaEditorSequenceArea::sl_changeCopyFormat(const QString& alg){
    setCopyFormatedAlgorithmId(alg);
}

void MaEditorSequenceArea::sl_changeColorScheme() {
    QAction *action = qobject_cast<QAction *>(sender());
    if (NULL == action) {
        action = GUIUtils::getCheckedAction(customColorSchemeMenuActions);
    }
    CHECK(NULL != action, );

    applyColorScheme(action->data().toString());
}

void MaEditorSequenceArea::sl_delCurrentSelection() {
    emit si_startMaChanging();
    deleteCurrentSelection();
    emit si_stopMaChanging(true);
}

void MaEditorSequenceArea::sl_cancelSelection() {
    if (maMode != ViewMode) {
        exitFromEditCharacterMode();
        return;
    }
    GRUNTIME_NAMED_CONDITION_COUNTER(cvat, tvar, qobject_cast<McaEditorWgt*>(sender()) != NULL, "Clear selection", editor->getFactoryId());
    MaEditorSelection emptySelection;
    setSelection(emptySelection);
}

void MaEditorSequenceArea::sl_fillCurrentSelectionWithGaps() {
    GRUNTIME_NAMED_COUNTER(cvat, tvar, "Fill selection with gaps", editor->getFactoryId());
    if(!isAlignmentLocked()) {
        emit si_startMaChanging();
        insertGapsBeforeSelection();
        emit si_stopMaChanging(true);
    }
}

void MaEditorSequenceArea::sl_alignmentChanged(const MultipleAlignment &, const MaModificationInfo &modInfo) {
    exitFromEditCharacterMode();
    int nSeq = editor->getNumSequences();
    int aliLen = editor->getAlignmentLen();

    if (ui->isCollapsibleMode()) {
        nSeq = getNumDisplayableSequences();
        updateCollapsedGroups(modInfo);
    }

    editor->updateReference();

    if ((selection.x() > aliLen - 1) || (selection.y() > nSeq - 1)) {
        sl_cancelSelection();
    } else {
        const QPoint selTopLeft(qMin(selection.x(), aliLen - 1),
            qMin(selection.y(), nSeq - 1));
        const QPoint selBottomRight(qMin(selection.x() + selection.width() - 1, aliLen - 1),
            qMin(selection.y() + selection.height() - 1, nSeq -1));

        MaEditorSelection newSelection(selTopLeft, selBottomRight);
        // we don't emit "selection changed" signal to avoid redrawing
        setSelection(newSelection);
    }

    ui->getScrollController()->sl_updateScrollBars();

    completeRedraw = true;
    sl_updateActions();
    update();
}

void MaEditorSequenceArea::sl_completeUpdate(){
    completeRedraw = true;
    sl_updateActions();
    update();
    onVisibleRangeChanged();
}

void MaEditorSequenceArea::sl_completeRedraw() {
    completeRedraw = true;
    update();
}

void MaEditorSequenceArea::sl_triggerUseDots() {
    useDotsAction->trigger();
}

void MaEditorSequenceArea::sl_useDots(){
    completeRedraw = true;
    update();
    emit si_highlightingChanged();
}

void MaEditorSequenceArea::sl_registerCustomColorSchemes() {
    deleteOldCustomSchemes();

    MsaSchemesMenuBuilder::createAndFillColorSchemeMenuActions(customColorSchemeMenuActions,
                                                               MsaSchemesMenuBuilder::Custom, getEditor()->getMaObject()->getAlphabet()->getType(),
                                                               this);
}

void MaEditorSequenceArea::sl_colorSchemeFactoryUpdated() {
    applyColorScheme(colorScheme->getFactory()->getId());
}

void MaEditorSequenceArea::sl_setDefaultColorScheme() {
    MsaColorSchemeFactory *defaultFactory = getDefaultColorSchemeFactory();
    SAFE_POINT(NULL != defaultFactory, L10N::nullPointerError("default color scheme factory"), );
    applyColorScheme(defaultFactory->getId());
}

void MaEditorSequenceArea::sl_changeHighlightScheme(){
    QAction* a = qobject_cast<QAction*>(sender());
    if (NULL == a) {
        a = GUIUtils::getCheckedAction(customColorSchemeMenuActions);
    }
    CHECK(NULL != a, );

    editor->saveHighlightingSettings(highlightingScheme->getFactory()->getId(), highlightingScheme->getSettings());

    QString id = a->data().toString();
    MsaHighlightingSchemeFactory* factory = AppContext::getMsaHighlightingSchemeRegistry()->getSchemeFactoryById(id);
    SAFE_POINT(NULL != factory, L10N::nullPointerError("highlighting scheme"), );
    if (ui->getEditor()->getMaObject() == NULL) {
        return;
    }

    delete highlightingScheme;
    highlightingScheme = factory->create(this, ui->getEditor()->getMaObject());
    highlightingScheme->applySettings(editor->getHighlightingSettings(id));

    const MultipleAlignment ma = ui->getEditor()->getMaObject()->getMultipleAlignment();

    U2OpStatusImpl os;
    const int refSeq = ma->getRowIndexByRowId(editor->getReferenceRowId(), os);

    MSAHighlightingFactory msaHighlightingFactory;
    QString msaHighlightingId = msaHighlightingFactory.getOPGroupParameters().getGroupId();

    CHECK(ui->getEditor(), );
    CHECK(ui->getEditor()->getOptionsPanel(), );

    if(!factory->isRefFree() && refSeq == -1 && ui->getEditor()->getOptionsPanel()->getActiveGroupId() != msaHighlightingId) {
        QMessageBox::warning(ui, tr("No reference sequence selected"),
            tr("Reference sequence for current highlighting scheme is not selected. Use context menu or Highlighting tab on Options panel to select it"));
    }

    foreach(QAction* action, highlightingSchemeMenuActions) {
        action->setChecked(action == a);
    }
    if (factory->isAlphabetTypeSupported(DNAAlphabet_RAW)) {
        AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_RAW, id);
    }
    if (factory->isAlphabetTypeSupported(DNAAlphabet_NUCL)) {
        AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_NUCL, id);
    }
    if (factory->isAlphabetTypeSupported(DNAAlphabet_AMINO)) {
        AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_AMINO, id);
    }
    if (factory->isAlphabetTypeSupported(DNAAlphabet_UNDEFINED)) {
        FAIL(tr("Unknown alphabet"), );
    }

    completeRedraw = true;
    update();
    emit si_highlightingChanged();
}

void MaEditorSequenceArea::sl_replaceSelectedCharacter() {
    maMode = ReplaceCharMode;
    editModeAnimationTimer.start(500);
    highlightCurrentSelection();
    sl_updateActions();
}

void MaEditorSequenceArea::sl_changeSelectionColor() {
    QColor black(Qt::black);
    selectionColor = (black == selectionColor) ? Qt::darkGray : Qt::black;
    update();
}

void MaEditorSequenceArea::sl_modelChanged() {
    updateSelection();
    ui->getScrollController()->updateVerticalScrollBar();
    sl_completeRedraw();
}

void MaEditorSequenceArea::sl_hScrollBarActionPerfermed() {
    CHECK((shifting && editingEnabled) || selecting, );
    const QAbstractSlider::SliderAction action = shBar->getRepeatAction();
    CHECK(QAbstractSlider::SliderSingleStepAdd == action || QAbstractSlider::SliderSingleStepSub == action, );

    const QPoint localPoint = mapFromGlobal(QCursor::pos());
    const QPoint newCurPos = ui->getScrollController()->getMaPointByScreenPoint(localPoint);

    if (shifting && editingEnabled) {
        shiftSelectedRegion(newCurPos.x() - cursorPos.x());
    } else if (selecting) {
        // There the correct geometry can be set
//        rubberBand->setGeometry(QRect(rubberBandOrigin, localPoint).normalized());
    }
}

void MaEditorSequenceArea::setCursorPos(const QPoint& p) {
    CHECK(!isAlignmentEmpty(), )
    SAFE_POINT(isInRange(p), tr("Cursor position is out of range"), );
    CHECK(p != cursorPos, );

    cursorPos = p;

    highlightSelection = false;
    sl_updateActions();
}

void MaEditorSequenceArea::setCursorPos(int x, int y) {
    setCursorPos(QPoint(x, y));
}

void MaEditorSequenceArea::setCursorPos(int pos) {
    setCursorPos(QPoint(pos, cursorPos.y()));
}

void MaEditorSequenceArea::resizeEvent(QResizeEvent *e) {
    completeRedraw = true;
    ui->getScrollController()->sl_updateScrollBars();
    emit si_visibleRangeChanged();
    QWidget::resizeEvent(e);
}

void MaEditorSequenceArea::paintEvent(QPaintEvent *e) {
    drawAll();
    QWidget::paintEvent(e);
}

void MaEditorSequenceArea::wheelEvent(QWheelEvent *we) {
    bool toMin = we->delta() > 0;
    if (we->modifiers() == 0) {
        shBar->triggerAction(toMin ? QAbstractSlider::SliderSingleStepSub : QAbstractSlider::SliderSingleStepAdd);
    }  else if (we->modifiers() & Qt::SHIFT) {
        svBar->triggerAction(toMin ? QAbstractSlider::SliderSingleStepSub : QAbstractSlider::SliderSingleStepAdd);
    }
    QWidget::wheelEvent(we);
}

void MaEditorSequenceArea::mousePressEvent(QMouseEvent *e) {
    prevPressedButton = e->button();

    if (!hasFocus()) {
        setFocus();
    }

    if ((e->button() == Qt::LeftButton)) {
        if (Qt::ShiftModifier == e->modifiers()) {
            QWidget::mousePressEvent(e);
            return;
        }

        Qt::KeyboardModifiers km = QApplication::keyboardModifiers();
        isCtrlPressed = km.testFlag(Qt::ControlModifier);
        lengthOnMousePress = editor->getMaObject()->getLength();

        rubberBandOrigin = e->pos();
        const QPoint p = ui->getScrollController()->getMaPointByScreenPoint(e->pos());
        setCursorPos(boundWithVisibleRange(p));
        if (isInRange(p)) {
            const MaEditorSelection &s = getSelection();
            if (s.getRect().contains(cursorPos) && !isAlignmentLocked() && editingEnabled) {
                shifting = true;
                maVersionBeforeShifting = editor->getMaObject()->getModificationVersion();
                U2OpStatus2Log os;
                changeTracker.startTracking(os);
                CHECK_OP(os, );
                editor->getMaObject()->saveState();
                emit si_startMaChanging();
            }
        }

        Qt::CursorShape shape = cursor().shape();
        if (shape != Qt::ArrowCursor) {
            QPoint pos = e->pos();
            changeTracker.finishTracking();
            shifting = false;
            QPoint globalMousePosition = ui->getScrollController()->getGlobalMousePosition(pos);
            const double baseWidth = ui->getBaseWidthController()->getBaseWidth();
            const double baseHeight = ui->getRowHeightController()->getSequenceHeight();
            movableBorder = SelectionModificationHelper::getMovableSide(shape, globalMousePosition, selection.getRect(), QSize(baseWidth, baseHeight));
            moveBorder(pos);
        } else if (!shifting) {
            selecting = true;
            rubberBandOrigin = e->pos();
            rubberBand->setGeometry(QRect(rubberBandOrigin, QSize()));
            const bool isMSAEditor = (qobject_cast<MSAEditor*>(getEditor()) != NULL);
            if (isMSAEditor) {
                rubberBand->show();
            }
            sl_cancelSelection();
        }
    }

    QWidget::mousePressEvent(e);
}

void MaEditorSequenceArea::mouseReleaseEvent(QMouseEvent *e) {
    rubberBand->hide();
    if (shifting) {
        changeTracker.finishTracking();
        editor->getMaObject()->releaseState();
    }

    QPoint newCurPos = ui->getScrollController()->getMaPointByScreenPoint(e->pos());

    if (shifting) {
        emit si_stopMaChanging(maVersionBeforeShifting != editor->getMaObject()->getModificationVersion());
    } else if (Qt::LeftButton == e->button() && Qt::LeftButton == prevPressedButton && movableBorder == SelectionModificationHelper::NoMovableBorder) {
        updateSelection(newCurPos);
    }
    shifting = false;
    selecting = false;
    maVersionBeforeShifting = -1;
    movableBorder = SelectionModificationHelper::NoMovableBorder;

    if (ctrlModeGapModel.isEmpty() && isCtrlPressed) {
        MultipleAlignmentObject* maObj = editor->getMaObject();
        maObj->si_completeStateChanged(true);
        MaModificationInfo mi;
        mi.alignmentLengthChanged = false;
        maObj->si_alignmentChanged(maObj->getMultipleAlignment(), mi);
    }
    ctrlModeGapModel.clear();

    ui->getScrollController()->stopSmoothScrolling();

    QWidget::mouseReleaseEvent(e);
}

void MaEditorSequenceArea::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        const QPoint p = event->pos();
        const QPoint newCurPos = ui->getScrollController()->getMaPointByScreenPoint(p);
        if (isInRange(newCurPos)) {
            if (isVisible(newCurPos, false)) {
                ui->getScrollController()->stopSmoothScrolling();
            } else {
                ScrollController::Directions direction = ScrollController::None;
                if (newCurPos.x() < ui->getScrollController()->getFirstVisibleBase(false)) {
                    direction |= ScrollController::Left;
                } else if (newCurPos.x() > ui->getScrollController()->getLastVisibleBase(width(), false)) {
                    direction |= ScrollController::Right;
                }

                if (newCurPos.y() < ui->getScrollController()->getFirstVisibleRowNumber(false)) {
                    direction |= ScrollController::Up;
                } else if (newCurPos.y() > ui->getScrollController()->getLastVisibleRowNumber(height(), false)) {
                    direction |= ScrollController::Down;
                }
                ui->getScrollController()->scrollSmoothly(direction);
            }
        }

        Qt::CursorShape shape = cursor().shape();
        if (shape != Qt::ArrowCursor) {
            moveBorder(p);
        } else if (shifting && editingEnabled) {
            shiftSelectedRegion(newCurPos.x() - cursorPos.x());
        } else if (selecting) {
            rubberBand->setGeometry(QRect(rubberBandOrigin, p).normalized());
        }
    } else {
        setBorderCursor(event->pos());
    }

    QWidget::mouseMoveEvent(event);
}

void MaEditorSequenceArea::setBorderCursor(const QPoint& p) {
    const QPoint globalMousePos = ui->getScrollController()->getGlobalMousePosition(p);
    setCursor(SelectionModificationHelper::getCursorShape(globalMousePos, selection.getRect(), ui->getBaseWidthController()->getBaseWidth(), ui->getRowHeightController()->getSequenceHeight()));
}

void MaEditorSequenceArea::moveBorder(const QPoint& screenMousePos) {
    CHECK(movableBorder != SelectionModificationHelper::NoMovableBorder, );

    QPoint globalMousePos = ui->getScrollController()->getGlobalMousePosition(screenMousePos);
    globalMousePos = QPoint(qMax(0, globalMousePos.x()), qMax(0, globalMousePos.y()));
    const qreal baseWidth = ui->getBaseWidthController()->getBaseWidth();
    const qreal baseHeight = ui->getRowHeightController()->getSequenceHeight();

    QRect newSelection = SelectionModificationHelper::getNewSelection(movableBorder, globalMousePos, QSizeF(baseWidth, baseHeight), selection.getRect());

    setCursor(SelectionModificationHelper::getCursorShape(movableBorder, cursor().shape()));

    CHECK(!newSelection.isEmpty(), );
    if (!isPosInRange(newSelection.right())) {
        newSelection.setRight(selection.getRect().right());
    }
    if (!isSeqInRange(newSelection.bottom())) {
        newSelection.setBottom(selection.bottom());
    }

    CHECK(isInRange(newSelection.bottomRight()), );
    CHECK(isInRange(newSelection.topLeft()), );
    setSelection(MaEditorSelection(newSelection.topLeft(), newSelection.bottomRight()));
}

void MaEditorSequenceArea::keyPressEvent(QKeyEvent *e) {
    if (!hasFocus()) {
        setFocus();
    }

    int key = e->key();
    if (maMode != ViewMode) {
        processCharacterInEditMode(e);
        return;
    }

    bool enlargeSelection = qobject_cast<MSAEditor*>(getEditor()) != NULL;

    bool shift = e->modifiers().testFlag(Qt::ShiftModifier);
    const bool ctrl = e->modifiers().testFlag(Qt::ControlModifier);
#ifdef Q_OS_MAC
    // In one case it is better to use a Command key as modifier,
    // in another - a Control key. genuineCtrl - Control key on Mac OS X.
    const bool genuineCtrl = e->modifiers().testFlag(Qt::MetaModifier);
#else
    const bool genuineCtrl = ctrl;
#endif
    static QPoint selectionStart(0, 0);
    static QPoint selectionEnd(0, 0);

    if (ctrl && (key == Qt::Key_Left || key == Qt::Key_Right || key == Qt::Key_Up || key == Qt::Key_Down)) {
        //remap to page_up/page_down
        shift = key == Qt::Key_Up || key == Qt::Key_Down;
        key =  (key == Qt::Key_Up || key == Qt::Key_Left) ? Qt::Key_PageUp : Qt::Key_PageDown;
    }
    //part of these keys are assigned to actions -> so them never passed to keyPressEvent (action handling has higher priority)
    int endX, endY;
    switch(key) {
        case Qt::Key_Escape:
             sl_cancelSelection();
             break;
        case Qt::Key_Left:
            if(!shift || !enlargeSelection) {
                moveSelection(-1,0);
                break;
            }
            if (selectionEnd.x() < 1) {
                break;
            }
            selectionEnd.setX(selectionEnd.x() - 1);
            endX = selectionEnd.x();
            if (isPosInRange(endX)) {
                if (endX != -1) {
                    int firstColumn = qMin(selectionStart.x(),endX);
                    int selectionWidth = qAbs(endX - selectionStart.x()) + 1;
                    int startSeq = selection.y();
                    int height = selection.height();
                    if (selection.isNull()) {
                        startSeq = cursorPos.y();
                        height = 1;
                    }
                    MaEditorSelection _selection(firstColumn, startSeq, selectionWidth, height);
                    setSelection(_selection);
                    ui->getScrollController()->scrollToBase(endX, width());
                }
            }
            break;
        case Qt::Key_Right:
            if(!shift || !enlargeSelection) {
                moveSelection(1,0);
                break;
            }
            if (selectionEnd.x() >= (editor->getAlignmentLen() - 1)) {
                break;
            }

            selectionEnd.setX(selectionEnd.x() +  1);
            endX = selectionEnd.x();
            if (isPosInRange(endX)) {
                if (endX != -1) {
                    int firstColumn = qMin(selectionStart.x(),endX);
                    int selectionWidth = qAbs(endX - selectionStart.x()) + 1;
                    int startSeq = selection.y();
                    int height = selection.height();
                    if (selection.isNull()) {
                        startSeq = cursorPos.y();
                        height = 1;
                    }
                    MaEditorSelection _selection(firstColumn, startSeq, selectionWidth, height);
                    setSelection(_selection);
                    ui->getScrollController()->scrollToBase(endX, width());
                }
            }
            break;
        case Qt::Key_Up:
            if(!shift || !enlargeSelection) {
                moveSelection(0,-1);
                break;
            }
            if(selectionEnd.y() < 1) {
                break;
            }
            selectionEnd.setY(selectionEnd.y() - 1);
            endY = selectionEnd.y();
            if (isSeqInRange(endY)) {
                if (endY != -1) {
                    int startSeq = qMin(selectionStart.y(),endY);
                    int height = qAbs(endY - selectionStart.y()) + 1;
                    int firstColumn = selection.x();
                    int width = selection.width();
                    if (selection.isNull()) {
                        firstColumn = cursorPos.x();
                        width = 1;
                    }
                    MaEditorSelection _selection(firstColumn, startSeq, width, height);
                    setSelection(_selection);
                    ui->getScrollController()->scrollToRowByNumber(endY, this->height());
                }
            }
            break;
        case Qt::Key_Down:
            if(!shift || !enlargeSelection) {
                moveSelection(0, 1);
                break;
            }
            if (selectionEnd.y() >= (ui->getCollapseModel()->getDisplayableRowsCount() - 1)) {
                break;
            }
            selectionEnd.setY(selectionEnd.y() + 1);
            endY = selectionEnd.y();
            if (isSeqInRange(endY)) {
                if (endY != -1) {
                    int startSeq = qMin(selectionStart.y(),endY);
                    int height = qAbs(endY - selectionStart.y()) + 1;
                    int firstColumn = selection.x();
                    int width = selection.width();
                    if (selection.isNull()) {
                        firstColumn = cursorPos.x();
                        width = 1;
                    }
                    MaEditorSelection _selection(firstColumn, startSeq, width, height);
                    setSelection(_selection);
                    ui->getScrollController()->scrollToRowByNumber(endY, this->height());
                }
            }
            break;
        case Qt::Key_Delete:
            if (!isAlignmentLocked() && !shift) {
                emit si_startMaChanging();
                deleteCurrentSelection();
            }
            break;
        case Qt::Key_Home:
            if (shift) {
                // vertical scrolling
                ui->getScrollController()->scrollToEnd(ScrollController::Up);
                setCursorPos(QPoint(cursorPos.x(), 0));
            } else {
                // horizontal scrolling
                ui->getScrollController()->scrollToEnd(ScrollController::Left);
                setCursorPos(QPoint(0, cursorPos.y()));
            }
            break;
        case Qt::Key_End:
            if (shift) {
                // vertical scrolling
                ui->getScrollController()->scrollToEnd(ScrollController::Down);
                setCursorPos(QPoint(cursorPos.x(), getNumDisplayableSequences() - 1));
            } else {
                // horizontal scrolling
                ui->getScrollController()->scrollToEnd(ScrollController::Right);
                setCursorPos(QPoint(editor->getAlignmentLen() - 1, cursorPos.y()));
            }
            break;
        case Qt::Key_PageUp:
            if (shift) {
                // vertical scrolling
                ui->getScrollController()->scrollPage(ScrollController::Up);
            } else {
                // horizontal scrolling
                ui->getScrollController()->scrollPage(ScrollController::Left);
            }
            break;
        case Qt::Key_PageDown:
            if (shift) {
                // vertical scrolling
                ui->getScrollController()->scrollPage(ScrollController::Down);
            } else {
                // horizontal scrolling
                ui->getScrollController()->scrollPage(ScrollController::Right);
            }
            break;
        case Qt::Key_Backspace:
            removeGapsPrecedingSelection(genuineCtrl ? 1 : -1);
            break;
        case Qt::Key_Insert:
        case Qt::Key_Space:
            // We can't use Command+Space on Mac OS X - it is reserved
            if(!isAlignmentLocked()) {
                emit si_startMaChanging();
                insertGapsBeforeSelection(genuineCtrl ? 1 : -1);
            }
            break;
        case Qt::Key_Shift:
            if (!selection.isNull()) {
                selectionStart = selection.topLeft();
                selectionEnd = selection.getRect().bottomRight();
            } else {
                selectionStart = cursorPos;
                selectionEnd = cursorPos;
            }
            break;
    }
    QWidget::keyPressEvent(e);
}

void MaEditorSequenceArea::keyReleaseEvent(QKeyEvent *ke) {
    if ((ke->key() == Qt::Key_Space || ke->key() == Qt::Key_Delete) && !isAlignmentLocked() && !ke->isAutoRepeat()) {
        emit si_stopMaChanging(true);
    }

    QWidget::keyReleaseEvent(ke);
}

void MaEditorSequenceArea::drawBackground(QPainter &) {

}

void MaEditorSequenceArea::insertGapsBeforeSelection(int countOfGaps) {
    CHECK(getEditor() != NULL, );
    if (selection.isNull() || 0 == countOfGaps || -1 > countOfGaps) {
        return;
    }
    SAFE_POINT(isInRange(selection.topLeft()), tr("Top left corner of the selection has incorrect coords"), );
    SAFE_POINT(isInRange(QPoint(selection.x() + selection.width() - 1, selection.y() + selection.height() - 1)),
        tr("Bottom right corner of the selection has incorrect coords"), );

    // if this method was invoked during a region shifting
    // then shifting should be canceled
    cancelShiftTracking();

    MultipleAlignmentObject *maObj = editor->getMaObject();
    if (NULL == maObj || maObj->isStateLocked()) {
        return;
    }
    U2OpStatus2Log os;
    U2UseCommonUserModStep userModStep(maObj->getEntityRef(), os);
    Q_UNUSED(userModStep);
    SAFE_POINT_OP(os,);

    const MultipleAlignment ma = maObj->getMultipleAlignment();
    if (selection.width() == ma->getLength() && selection.height() == ma->getNumRows()) {
        return;
    }

    const int removedRegionWidth = (-1 == countOfGaps) ? selection.width() : countOfGaps;
    const U2Region& sequences = getSelectedRows();
    maObj->insertGap(sequences, selection.x(), removedRegionWidth);
    adjustReferenceLength(os);
    CHECK_OP(os,);
    moveSelection(removedRegionWidth, 0, true);
    if (!getSelection().isEmpty()) {
        ui->getScrollController()->scrollToMovedSelection(ScrollController::Right);
    }
}

void MaEditorSequenceArea::removeGapsPrecedingSelection(int countOfGaps) {
    const MaEditorSelection selectionBackup = selection;
    // check if selection exists
    if (selectionBackup.isNull()) {
        return;
    }

    const QPoint selectionTopLeftCorner(selectionBackup.topLeft());
    // don't perform the deletion if the selection is at the alignment start
    if (0 == selectionTopLeftCorner.x() || -1 > countOfGaps || 0 == countOfGaps) {
        return;
    }

    int removedRegionWidth = (-1 == countOfGaps) ? selectionBackup.width() : countOfGaps;
    QPoint topLeftCornerOfRemovedRegion(selectionTopLeftCorner.x() - removedRegionWidth,
        selectionTopLeftCorner.y());
    if (0 > topLeftCornerOfRemovedRegion.x()) {
        removedRegionWidth -= qAbs(topLeftCornerOfRemovedRegion.x());
        topLeftCornerOfRemovedRegion.setX(0);
    }

    MultipleAlignmentObject *maObj = editor->getMaObject();
    if (NULL == maObj || maObj->isStateLocked()) {
        return;
    }

    // if this method was invoked during a region shifting
    // then shifting should be canceled
    cancelShiftTracking();

    const U2Region rowsContainingRemovedGaps(getSelectedRows());
    U2OpStatus2Log os;
    U2UseCommonUserModStep userModStep(maObj->getEntityRef(), os);
    Q_UNUSED(userModStep);

    const int countOfDeletedSymbols = maObj->deleteGap(os, rowsContainingRemovedGaps,
        topLeftCornerOfRemovedRegion.x(), removedRegionWidth);

    // if some symbols were actually removed and the selection is not located
    // at the alignment end, then it's needed to move the selection
    // to the place of the removed symbols
    if (0 < countOfDeletedSymbols) {
        const MaEditorSelection newSelection(selectionBackup.x() - countOfDeletedSymbols,
            topLeftCornerOfRemovedRegion.y(), selectionBackup.width(),
            selectionBackup.height());
        setSelection(newSelection);
    }
}

void MaEditorSequenceArea::cancelShiftTracking() {
    shifting = false;
    selecting = false;
    changeTracker.finishTracking();
    editor->getMaObject()->releaseState();
}

void MaEditorSequenceArea::drawAll() {
    QSize s = size() * devicePixelRatio();
    if (cachedView->size() != s) {
        delete cachedView;
        cachedView = new QPixmap(s);
        cachedView->setDevicePixelRatio(devicePixelRatio());
        completeRedraw = true;
    }
    if (completeRedraw) {
        cachedView->fill(Qt::transparent);
        QPainter pCached(cachedView);
        drawVisibleContent(pCached);
        completeRedraw = false;
    }

    QPainter painter(this);
    painter.fillRect(QRect(QPoint(0, 0), s), Qt::white);
    drawBackground(painter);

    painter.drawPixmap(0, 0, *cachedView);
    renderer->drawSelection(painter);
    renderer->drawFocus(painter);
}

void MaEditorSequenceArea::updateColorAndHighlightSchemes() {
    Settings* s = AppContext::getSettings();
    if (!s || !editor){
        return;
    }
    MultipleAlignmentObject* maObj = editor->getMaObject();
    if (!maObj){
        return;
    }

    const DNAAlphabet* al = maObj->getAlphabet();
    if (!al){
        return;
    }

    MsaColorSchemeRegistry* csr = AppContext::getMsaColorSchemeRegistry();
    MsaHighlightingSchemeRegistry* hsr = AppContext::getMsaHighlightingSchemeRegistry();

    QString csid;
    QString hsid;
    getColorAndHighlightingIds(csid, hsid);
    MsaColorSchemeFactory* csf = csr->getSchemeFactoryById(csid);
    MsaHighlightingSchemeFactory* hsf = hsr->getSchemeFactoryById(hsid);
    initColorSchemes(csf);
    initHighlightSchemes(hsf);
}

void MaEditorSequenceArea::initColorSchemes(MsaColorSchemeFactory *defaultColorSchemeFactory) {
    MsaColorSchemeRegistry *msaColorSchemeRegistry = AppContext::getMsaColorSchemeRegistry();
    connect(msaColorSchemeRegistry, SIGNAL(si_customSettingsChanged()), SLOT(sl_registerCustomColorSchemes()));

    registerCommonColorSchemes();
    sl_registerCustomColorSchemes();

    applyColorScheme(defaultColorSchemeFactory->getId());
}

void MaEditorSequenceArea::registerCommonColorSchemes() {
    qDeleteAll(colorSchemeMenuActions);
    colorSchemeMenuActions.clear();

    MsaSchemesMenuBuilder::createAndFillColorSchemeMenuActions(colorSchemeMenuActions, MsaSchemesMenuBuilder::Common, getEditor()->getMaObject()->getAlphabet()->getType(), this);
}

void MaEditorSequenceArea::initHighlightSchemes(MsaHighlightingSchemeFactory* hsf) {
    qDeleteAll(highlightingSchemeMenuActions);
    highlightingSchemeMenuActions.clear();
    SAFE_POINT(hsf != NULL, "Highlight scheme factory is NULL", );

    MultipleAlignmentObject* maObj = editor->getMaObject();
    QVariantMap settings = highlightingScheme != NULL
            ? highlightingScheme->getSettings()
            : QVariantMap();
    delete highlightingScheme;

    highlightingScheme = hsf->create(this, maObj);
    highlightingScheme->applySettings(settings);

    MsaSchemesMenuBuilder::createAndFillHighlightingMenuActions(highlightingSchemeMenuActions, getEditor()->getMaObject()->getAlphabet()->getType(), this);
    QList<QAction *> tmpActions = QList<QAction *>() << highlightingSchemeMenuActions;
    foreach(QAction *action, tmpActions) {
        action->setChecked(action->data() == hsf->getId());
    }
}

MsaColorSchemeFactory * MaEditorSequenceArea::getDefaultColorSchemeFactory() const {
    MsaColorSchemeRegistry *msaColorSchemeRegistry = AppContext::getMsaColorSchemeRegistry();

    switch (editor->getMaObject()->getAlphabet()->getType()) {
    case DNAAlphabet_RAW:
        return msaColorSchemeRegistry->getSchemeFactoryById(MsaColorScheme::EMPTY);
    case DNAAlphabet_NUCL:
        return msaColorSchemeRegistry->getSchemeFactoryById(MsaColorScheme::UGENE_NUCL);
    case DNAAlphabet_AMINO:
        return msaColorSchemeRegistry->getSchemeFactoryById(MsaColorScheme::UGENE_AMINO);
    default:
        FAIL(tr("Unknown alphabet"), NULL);
    }
    return NULL;
}

MsaHighlightingSchemeFactory* MaEditorSequenceArea::getDefaultHighlightingSchemeFactory() const {
    MsaHighlightingSchemeRegistry *hsr = AppContext::getMsaHighlightingSchemeRegistry();
    MsaHighlightingSchemeFactory *hsf = hsr->getSchemeFactoryById(MsaHighlightingScheme::EMPTY);
    return hsf;
}

void MaEditorSequenceArea::getColorAndHighlightingIds(QString &csid, QString &hsid) {
    DNAAlphabetType atype = getEditor()->getMaObject()->getAlphabet()->getType();
    Settings* s = AppContext::getSettings();
    switch (atype) {
    case DNAAlphabet_RAW:
        csid = s->getValue(SETTINGS_ROOT + SETTINGS_COLOR_RAW, MsaColorScheme::EMPTY).toString();
        hsid = s->getValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_RAW, MsaHighlightingScheme::EMPTY).toString();
        break;
    case DNAAlphabet_NUCL:
        csid = s->getValue(SETTINGS_ROOT + SETTINGS_COLOR_NUCL, MsaColorScheme::UGENE_NUCL).toString();
        hsid = s->getValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_NUCL, MsaHighlightingScheme::EMPTY).toString();
        break;
    case DNAAlphabet_AMINO:
        csid = s->getValue(SETTINGS_ROOT + SETTINGS_COLOR_AMINO, MsaColorScheme::UGENE_AMINO).toString();
        hsid = s->getValue(SETTINGS_ROOT + SETTINGS_HIGHLIGHT_AMINO, MsaHighlightingScheme::EMPTY).toString();
        break;
    default:
        csid = "";
        hsid = "";
        break;
    }

    MsaColorSchemeRegistry* csr = AppContext::getMsaColorSchemeRegistry();
    MsaHighlightingSchemeRegistry* hsr = AppContext::getMsaHighlightingSchemeRegistry();

    MsaColorSchemeFactory* csf = csr->getSchemeFactoryById(csid);
    if (csf == NULL) {
        csid = getDefaultColorSchemeFactory()->getId();
    }
    MsaHighlightingSchemeFactory* hsf = hsr->getSchemeFactoryById(hsid);
    if (hsf == NULL) {
        hsid = getDefaultHighlightingSchemeFactory()->getId();
    }


    if (colorScheme != NULL && colorScheme->getFactory()->isAlphabetTypeSupported(atype)) {
        csid = colorScheme->getFactory()->getId();
    }
    if (highlightingScheme != NULL && highlightingScheme->getFactory()->isAlphabetTypeSupported(atype)) {
        hsid = highlightingScheme->getFactory()->getId();
    }
}

void MaEditorSequenceArea::applyColorScheme(const QString &id) {
    CHECK(NULL != ui->getEditor()->getMaObject(), );

    MsaColorSchemeFactory *factory = AppContext::getMsaColorSchemeRegistry()->getSchemeFactoryById(id);
    delete colorScheme;
    colorScheme = factory->create(this, ui->getEditor()->getMaObject());

    connect(factory, SIGNAL(si_factoryChanged()), SLOT(sl_colorSchemeFactoryUpdated()), Qt::UniqueConnection);
    connect(factory, SIGNAL(destroyed(QObject *)), SLOT(sl_setDefaultColorScheme()), Qt::UniqueConnection);

    QList<QAction *> tmpActions = QList<QAction *>() << colorSchemeMenuActions << customColorSchemeMenuActions;
    foreach (QAction *action, tmpActions) {
        action->setChecked(action->data() == id);
    }

    if (qobject_cast<MSAEditor*>(getEditor()) != NULL) { // to avoid setting of sanger scheme
        switch (ui->getEditor()->getMaObject()->getAlphabet()->getType()) {
        case DNAAlphabet_RAW:
            AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_COLOR_RAW, id);
            break;
        case DNAAlphabet_NUCL:
            AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_COLOR_NUCL, id);
            break;
        case DNAAlphabet_AMINO:
            AppContext::getSettings()->setValue(SETTINGS_ROOT + SETTINGS_COLOR_AMINO, id);
            break;
        default:
            FAIL(tr("Unknown alphabet"), );
            break;
        }
    }

    completeRedraw = true;
    update();
    emit si_highlightingChanged();
}

void MaEditorSequenceArea::processCharacterInEditMode(QKeyEvent *e) {
    if (e->key() == Qt::Key_Escape) {
        exitFromEditCharacterMode();
        return;
    }

    QString text = e->text().toUpper();
    if (1 == text.length()) {
        if (isCharacterAcceptable(text)) {
            QChar newChar = text.at(0);
            newChar = (newChar == '-' || newChar == emDash || newChar == ' ') ? U2Msa::GAP_CHAR : newChar;
            processCharacterInEditMode(newChar.toLatin1());
        } else {
            MainWindow *mainWindow = AppContext::getMainWindow();
            mainWindow->addNotification(getInacceptableCharacterErrorMessage(), Error_Not);
            exitFromEditCharacterMode();
        }
    }
}

void MaEditorSequenceArea::processCharacterInEditMode(char newCharacter) {
    switch (maMode) {
    case ReplaceCharMode:
        replaceChar(newCharacter);
        break;
    case InsertCharMode:
        insertChar(newCharacter);
    case ViewMode:
    default:
        // do nothing
        ;
    }
}

void MaEditorSequenceArea::replaceChar(char newCharacter) {
    CHECK(maMode == ReplaceCharMode, );
    CHECK(getEditor() != NULL, );
    if (selection.isNull()) {
        return;
    }
    SAFE_POINT(isInRange(selection.topLeft()), "Incorrect selection is detected!", );
    MultipleAlignmentObject* maObj = editor->getMaObject();
    if (maObj == NULL || maObj->isStateLocked()) {
        return;
    }
    if (maObj->getNumRows() == 1 && maObj->getRow(selection.y())->getCoreLength() == 1 && newCharacter == U2Msa::GAP_CHAR) {
        exitFromEditCharacterMode();
        return;
    }

    const bool isGap = maObj->getRow(selection.y())->isGap(selection.x());
    GRUNTIME_NAMED_CONDITION_COUNTER(cvar, tvar, isGap, "Replace gap", editor->getFactoryId());
    GRUNTIME_NAMED_CONDITION_COUNTER(ccvar, ttvar, !isGap, "Replace character", editor->getFactoryId());

    U2OpStatusImpl os;
    U2UseCommonUserModStep userModStep(maObj->getEntityRef(), os);
    Q_UNUSED(userModStep);
    SAFE_POINT_OP(os, );

    // replacement is valid only for one symbol
    const U2Region& sel = getSelectedRows();
    for (qint64 rowIndex = sel.startPos; rowIndex < sel.endPos(); rowIndex++) {
        maObj->replaceCharacter(selection.x(), rowIndex, newCharacter);
    }

    exitFromEditCharacterMode();
}

void MaEditorSequenceArea::exitFromEditCharacterMode() {
    if (maMode != ViewMode) {
        editModeAnimationTimer.stop();
        highlightSelection = false;
        selectionColor = Qt::black;
        maMode = ViewMode;
        sl_updateActions();
        update();
    }
}

bool MaEditorSequenceArea::isCharacterAcceptable(const QString &text) const {
    static const QRegExp latinCharacterOrGap(QString("([A-Z]| |-|%1)").arg(emDash));
    return latinCharacterOrGap.exactMatch(text);
}

const QString &MaEditorSequenceArea::getInacceptableCharacterErrorMessage() const {
    static const QString message = tr("It is not possible to insert the character into the alignment. "
                                      "Please use a character from set A-Z (upper-case or lower-case) or the gap character ('Space', '-' or '%1').").arg(emDash);
    return message;
}

void MaEditorSequenceArea::deleteOldCustomSchemes() {
    qDeleteAll(customColorSchemeMenuActions);
    customColorSchemeMenuActions.clear();
}

void MaEditorSequenceArea::updateCollapsedGroups(const MaModificationInfo&) {

}

MaEditorSequenceArea::MaMode MaEditorSequenceArea::getModInfo() {
    return maMode;
}

} // namespace

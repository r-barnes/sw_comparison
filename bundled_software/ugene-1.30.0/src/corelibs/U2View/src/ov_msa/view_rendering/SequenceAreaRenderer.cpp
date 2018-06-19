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

#include <QPainter>

#include <U2Algorithm/MsaHighlightingScheme.h>
#include <U2Algorithm/MsaColorScheme.h>

#include <U2Core/U2OpStatusUtils.h>

#include "SequenceAreaRenderer.h"
#include "ov_msa/helpers/BaseWidthController.h"
#include "ov_msa/helpers/DrawHelper.h"
#include "ov_msa/helpers/RowHeightController.h"
#include "ov_msa/helpers/ScrollController.h"

namespace U2 {

SequenceAreaRenderer::SequenceAreaRenderer(MaEditorWgt *ui, MaEditorSequenceArea *seqAreaWgt)
    : QObject(seqAreaWgt),
      ui(ui),
      seqAreaWgt(seqAreaWgt),
      drawLeadingAndTrailingGaps(true) {
}

bool SequenceAreaRenderer::drawContent(QPainter &painter, const U2Region &region, const QList<int> &seqIdx, int xStart, int yStart)  const {
    CHECK(!region.isEmpty(), false);
    CHECK(!seqIdx.isEmpty(), false);

    MsaHighlightingScheme* highlightingScheme = seqAreaWgt->getCurrentHighlightingScheme();
    MaEditor* editor = seqAreaWgt->getEditor();

    painter.setPen(Qt::black);
    painter.setFont(editor->getFont());

    MultipleAlignmentObject* maObj = editor->getMaObject();
    SAFE_POINT(maObj != NULL, tr("Alignment object is NULL"), false);
    const MultipleAlignment ma = maObj->getMultipleAlignment();

    //Use dots to draw regions, which are similar to reference sequence
    highlightingScheme->setUseDots(seqAreaWgt->getUseDotsCheckedState());

    foreach (const int rowIndex, seqIdx) {
        drawRow(painter, ma, rowIndex, region, xStart, yStart);
        yStart += ui->getRowHeightController()->getRowHeight(rowIndex);
    }

    return true;
}

void SequenceAreaRenderer::drawSelection(QPainter &painter) const {
    MaEditorSelection selection = seqAreaWgt->getSelection();

    const QRect selectionRect = ui->getDrawHelper()->getSelectionScreenRect(selection);

    QPen pen(seqAreaWgt->highlightSelection || seqAreaWgt->hasFocus()
             ? seqAreaWgt->selectionColor
             : Qt::gray);
    if (seqAreaWgt->maMode == MaEditorSequenceArea::ViewMode) {
        pen.setStyle(Qt::DashLine);
    }
    pen.setWidth(seqAreaWgt->highlightSelection ? 2 : 1);
    painter.setPen(pen);

    switch (seqAreaWgt->maMode) {
    case MaEditorSequenceArea::ViewMode:
    case MaEditorSequenceArea::ReplaceCharMode:
        painter.drawRect(selectionRect);
        break;
    case MaEditorSequenceArea::InsertCharMode:
        painter.drawLine(selectionRect.left(), selectionRect.top(), selectionRect.left(), selectionRect.bottom());
        break;
    }
}

void SequenceAreaRenderer::drawFocus(QPainter &painter) const {
    if (seqAreaWgt->hasFocus()) {
        painter.setPen(QPen(Qt::black, 1, Qt::DotLine));
        painter.drawRect(0, 0, seqAreaWgt->width() - 1, seqAreaWgt->height() - 1);
    }
}

int SequenceAreaRenderer::drawRow(QPainter &painter, const MultipleAlignment &ma, int rowIndex, const U2Region &region, int xStart, int yStart) const {
    // SANGER_TODO: deal with frequent handlign of editor or h/color schemes through the editor etc.
    // move to class parameter
    MsaHighlightingScheme* highlightingScheme = seqAreaWgt->getCurrentHighlightingScheme();
    highlightingScheme->setUseDots(seqAreaWgt->getUseDotsCheckedState());

    MaEditor* editor = seqAreaWgt->getEditor();
    QString schemeName = highlightingScheme->metaObject()->className();
    bool isGapsScheme = schemeName == "U2::MSAHighlightingSchemeGaps";
    bool isResizeMode = editor->getResizeMode() == MSAEditor::ResizeMode_FontAndContent;

    U2OpStatusImpl os;
    const int refSeq = ma->getRowIndexByRowId(editor->getReferenceRowId(), os);
    QString refSeqName = editor->getReferenceRowName();

    qint64 regionEnd = region.endPos() - (int)(region.endPos() == editor->getAlignmentLen());
    MultipleAlignmentRow row = ma->getRow(rowIndex);
    const int rowHeight = ui->getRowHeightController()->getSequenceHeight();
    const int baseWidth = ui->getBaseWidthController()->getBaseWidth();
    for (int pos = region.startPos; pos <= regionEnd; pos++) {
        if (!drawLeadingAndTrailingGaps
                && (pos < row->getCoreStart() || pos > row->getCoreStart() + row->getCoreLength() - 1)) {
            xStart += baseWidth;
            continue;
        }

        const QRect charRect(xStart, yStart, baseWidth, rowHeight);
        char c = ma->charAt(rowIndex, pos);

        bool highlight = false;
        QColor color = seqAreaWgt->getCurrentColorScheme()->getColor(rowIndex, pos, c); //! SANGER_TODO: add NULL checks or do smt with the infrastructure
        if (isGapsScheme || highlightingScheme->getFactory()->isRefFree()) { //schemes which applied without reference
            const char refChar = '\n';
            highlightingScheme->process(refChar, c, color, highlight, pos, rowIndex);
        } else if (rowIndex == refSeq || refSeqName.isEmpty()) {
            highlight = true;
        } else {
            const char refChar = editor->getReferenceCharAt(pos);
            highlightingScheme->process(refChar, c, color, highlight, pos, rowIndex);
        }

        if (color.isValid() && highlight) {
            painter.fillRect(charRect, color);
        }
        if (isResizeMode) {
            painter.drawText(charRect, Qt::AlignCenter, QString(c));
        }

        xStart += baseWidth;
    }
    return rowHeight;
}

} // namespace

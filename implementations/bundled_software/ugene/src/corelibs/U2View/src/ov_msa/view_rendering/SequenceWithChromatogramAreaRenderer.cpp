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

#include <U2Core/AppContext.h>
#include <U2Core/DNASequenceSelection.h>
#include <U2Core/Settings.h>

#include <U2View/SequenceObjectContext.h>

#include "SequenceWithChromatogramAreaRenderer.h"
#include "ov_msa/McaEditorNameList.h"
#include "ov_msa/McaEditorSequenceArea.h"
#include "ov_msa/helpers/BaseWidthController.h"
#include "ov_msa/helpers/RowHeightController.h"
#include "ov_msa/helpers/ScrollController.h"

namespace U2 {

const int SequenceWithChromatogramAreaRenderer::INDENT_BETWEEN_ROWS = 15;
const int SequenceWithChromatogramAreaRenderer::CHROMATOGRAM_MAX_HEIGHT = 100;
const qreal SequenceWithChromatogramAreaRenderer::TRACE_OR_BC_LINES_DIVIDER = 2;

SequenceWithChromatogramAreaRenderer::SequenceWithChromatogramAreaRenderer(MaEditorWgt *ui, McaEditorSequenceArea *seqAreaWgt)
    : SequenceAreaRenderer(ui, seqAreaWgt),
      linePen(Qt::gray, 1, Qt::DotLine)
{
    drawLeadingAndTrailingGaps = false;

    QFontMetricsF fm(seqAreaWgt->getEditor()->getFont());
    charWidth = fm.width('W');
    charHeight = fm.ascent();

    heightBC = seqAreaWgt->getEditor()->getSequenceRowHeight();
    heightPD = getChromatogramHeight() - INDENT_BETWEEN_ROWS;
    heightQuality = charHeight;

    Settings* s = AppContext::getSettings();
    SAFE_POINT(s != NULL, "AppContext::settings is NULL", );
    SAFE_POINT(ui->getEditor() != NULL, "MaEditor is NULL", );
    maxTraceHeight = s->getValue(ui->getEditor()->getSettingsRoot() + MCAE_SETTINGS_PEAK_HEIGHT, heightPD - heightBC).toInt();
}

void SequenceWithChromatogramAreaRenderer::drawReferenceSelection(QPainter &painter) const {
    McaEditor* editor = getSeqArea()->getEditor();
    SAFE_POINT(editor != NULL, "McaEditor is NULL", );
    DNASequenceSelection* selection = editor->getReferenceContext()->getSequenceSelection();
    SAFE_POINT(selection != NULL, "DNASequenceSelection is NULL", );
    SAFE_POINT(selection->regions.size() <= 1, "Unexpected multiselection",);
    CHECK(!selection->regions.isEmpty(), );

    U2Region region = selection->regions.first();
    const U2Region xRange = ui->getBaseWidthController()->getBasesScreenRange(region);

    painter.save();
    // SANGER_TODO: color can be const -- for consensus and here
    QColor color(Qt::lightGray);
    color = color.lighter(115);
    color.setAlpha(127);
    painter.fillRect(xRange.startPos, 0,
               xRange.length, seqAreaWgt->height(),
               color);
    painter.restore();
}

void SequenceWithChromatogramAreaRenderer::drawNameListSelection(QPainter &painter) const {
    McaEditor* editor = getSeqArea()->getEditor();
    SAFE_POINT(editor != NULL, "McaEditor is NULL", );
    SAFE_POINT(editor->getUI() != NULL, "McaEditor UI is NULL", );

    MaEditorNameList* nameList = editor->getUI()->getEditorNameList();
    SAFE_POINT(nameList != NULL, "MaEditorNameList is NULL", );
    U2Region selection = nameList->getSelection();
    CHECK(!selection.isEmpty(), );
    U2Region selectionPxl = ui->getRowHeightController()->getRowsScreenRangeByNumbers(selection);
    painter.save();
    // SANGER_TODO: color can be const -- for consensus and here
    QColor color(Qt::lightGray);
    color = color.lighter(115);
    color.setAlpha(127);
    painter.fillRect(0, selectionPxl.startPos, seqAreaWgt->width(), selectionPxl.length, color);
    painter.restore();
}

void SequenceWithChromatogramAreaRenderer::setAreaHeight(int h) {
    maxTraceHeight = h;

    Settings* s = AppContext::getSettings();
    SAFE_POINT(s != NULL, "AppContext::settings is NULL", );
    SAFE_POINT(ui->getEditor() != NULL, "MaEditor is NULL", );
    s->setValue(ui->getEditor()->getSettingsRoot() + MCAE_SETTINGS_PEAK_HEIGHT, maxTraceHeight);
}

int SequenceWithChromatogramAreaRenderer::getAreaHeight() const {
    return maxTraceHeight;
}

int SequenceWithChromatogramAreaRenderer::getScaleBarValue() const {
    return maxTraceHeight;
}

int SequenceWithChromatogramAreaRenderer::drawRow(QPainter &painter, const MultipleAlignment &mca, int rowIndex, const U2Region &region, int xStart, int yStart) const {
    McaEditor* editor = getSeqArea()->getEditor();
    if (editor->isChromVisible(rowIndex)) {
        painter.translate(0, INDENT_BETWEEN_ROWS / 2);
    }
    bool ok = SequenceAreaRenderer::drawRow(painter, mca, rowIndex, region, xStart, yStart);
    CHECK(ok, -1);

    SAFE_POINT(getSeqArea() != NULL, "seqAreaWgt is NULL", -1);
    const int width = getSeqArea()->width();
    const int seqRowHeight = editor->getUI()->getRowHeightController()->getSequenceHeight();
    if (editor->isChromVisible(rowIndex)) {
        painter.save();
        painter.translate(0, yStart + seqRowHeight);
        painter.setPen(QPen(Qt::gray, 1, Qt::DashLine));
        painter.drawLine(0, - INDENT_BETWEEN_ROWS / 2 - seqRowHeight, width, - INDENT_BETWEEN_ROWS / 2 - seqRowHeight);

        const MultipleChromatogramAlignmentRow& row = editor->getMaObject()->getMcaRow(rowIndex);
        drawChromatogram(painter, row, region, xStart);
        painter.setPen(QPen(Qt::gray, 1, Qt::DashLine));
        painter.restore();
        painter.translate(0, - INDENT_BETWEEN_ROWS / 2);
    }
    return seqRowHeight;
}

void SequenceWithChromatogramAreaRenderer::drawChromatogram(QPainter &painter, const MultipleChromatogramAlignmentRow &row, const U2Region &visibleRegion, int xStart) const {
    const DNAChromatogram chroma = row->getGappedChromatogram();

    // SANGER_TODO: should not be here
    chromaMax = 0;
    for (int i = 0; i < chroma.traceLength; i++)
    {
        if (chromaMax < chroma.A[i]) chromaMax = chroma.A[i];
        if (chromaMax < chroma.C[i]) chromaMax = chroma.C[i];
        if (chromaMax < chroma.G[i]) chromaMax = chroma.G[i];
        if (chromaMax < chroma.T[i]) chromaMax = chroma.T[i];
    }

    U2Region regionToDraw = visibleRegion.intersect(row->getCoreRegion());
    CHECK(!regionToDraw.isEmpty(), );

    if (regionToDraw.startPos > visibleRegion.startPos) {
        MaEditor* editor = seqAreaWgt->getEditor();
        SAFE_POINT(editor != NULL, "MaEditor is NULL", );
        const int emptySpaceWidth = ui->getBaseWidthController()->getBasesWidth(regionToDraw.startPos - visibleRegion.startPos);
        painter.translate(emptySpaceWidth, 0);
    }
    regionToDraw.startPos -= row->getCoreStart();

    painter.translate(xStart, 0);

    const int regionWidth = ui->getBaseWidthController()->getBasesWidth(regionToDraw);
    const QByteArray seq = row->getCore(); // SANGER_TODO: tmp, get only required region

    // SANGER_TODO:
//    GSLV_UpdateFlags uf = view->getUpdateFlags();
    const bool completeRedraw = true; //uf.testFlag(GSLV_UF_NeedCompleteRedraw) || uf.testFlag(GSLV_UF_ViewResized) || uf.testFlag(GSLV_UF_VisibleRangeChanged);
    bool drawQuality = chroma.hasQV && getSeqArea()->getShowQA();
    const bool baseCallsLinesVisible = seqAreaWgt->getEditor()->getResizeMode() == MSAEditor::ResizeMode_FontAndContent;

    if (completeRedraw) {
        painter.setRenderHint(QPainter::Antialiasing, true);
        painter.setPen(Qt::black);
        if (baseCallsLinesVisible) {
            // quality and base calls can be visible
            if (drawQuality) {
                drawQualityValues(chroma, regionWidth, heightQuality,
                                  painter, regionToDraw, seq);
                painter.translate(0, heightQuality);
            }
            drawOriginalBaseCalls(drawQuality * heightQuality, painter, regionToDraw, seq);
        } else {
            drawQuality = false; // to avoid shifting in case the base calls and quality was not visible
        }

        if (regionWidth / charWidth > regionToDraw.length / TRACE_OR_BC_LINES_DIVIDER) {
            // draw continious trace
            drawChromatogramTrace(chroma, 0, heightBC, heightPD - heightBC - drawQuality * heightQuality,
                                  painter, regionToDraw);
        } else {
            // draw only "columns" of peaks
            drawChromatogramBaseCallsLines(chroma, heightPD, painter, regionToDraw, seq);
        }
    }

    painter.translate(-xStart, 0);
}

QColor SequenceWithChromatogramAreaRenderer::getBaseColor( char base ) const {
    switch(base) {
        case 'A':
            return Qt::darkGreen;
        case 'C':
            return Qt::blue;
        case 'G':
            return Qt::black;
        case 'T':
            return Qt::red;
        default:
            return Qt::black;
    }
}

namespace {

static int getPreviousBaseCallEndPosition(const QVector<ushort>& baseCalls, int startPos) {
    int res = 0;
    SAFE_POINT(startPos > 0 && startPos < baseCalls.size(), "Out of array boundary", 0);
    int prevStep = baseCalls[startPos] - baseCalls[startPos - 1];
    //When many gaps was insered to the single place, the difference between current and previous baceCalls element may be very little.
    //Because of it, left correct point to draw may be out of the left edge of visible area
    //If it happends, we need to go to the left while we will find a correct point
    if (prevStep <= 1) {
        int pos = startPos - 1;
        while (prevStep == 0 && pos > 0) {
            prevStep = baseCalls[pos] - baseCalls[pos - 1];
            pos--;
        }
        SAFE_POINT(pos >= 0, "Pos is the ordinal number of baseCalls element, it must be >= 0", 0);
        res = baseCalls[startPos] - prevStep;
    } else {
        res = baseCalls[startPos] - prevStep / 2;
    }
    return res;
}

static int getCorrectPointsCountVariable(const QVector<ushort>& baseCalls, int pointsCount, int endPos, int currentNumBer) {
    //The same situation as with "getPreviousBaseCallEndPosition" except in this case we look for correct point for right edge
    if (currentNumBer != endPos - 1) {
        return pointsCount;
    }

    int res = pointsCount;
    int pos = endPos;
    while (res == 0 && pos < baseCalls.size()) {
        res = baseCalls[pos] - baseCalls[pos - 1];
        pos++;
    }
    SAFE_POINT(pos <= baseCalls.size(), "Pos is the ordinal number of baseCalls element, it must be < baseCalls.size() ", 0);
    return res;
}

}

void SequenceWithChromatogramAreaRenderer::drawChromatogramTrace(const DNAChromatogram& chroma,
                                                                 qreal x, qreal y, qreal h, QPainter& p,
                                                                 const U2Region& visible) const
{
    if (chromaMax == 0) {
        //nothing to draw
        return;
    }
    //founding problems

    p.setRenderHint(QPainter::Antialiasing, true);
    p.translate(x, h + y);

    QPolygonF polylineA;
    QPolygonF polylineC;
    QPolygonF polylineG;
    QPolygonF polylineT;
    int areaHeight = (heightPD - heightBC) * this->maxTraceHeight / 100;
    qreal columnWidth = getSeqArea()->getEditor()->getColumnWidth();

    int startPos = visible.startPos;
    int prev = 0;
    if (startPos != 0) {
        prev = getPreviousBaseCallEndPosition(chroma.baseCalls, startPos);
    }

    qint64 endPos = visible.endPos();
    for (int i = startPos; i < endPos; i++) {
        SAFE_POINT(i < chroma.baseCalls.length(), "Base calls array is too short: visible range index is out range", );
        int currentBaseCalls = chroma.baseCalls[i];
        int pointsCount = currentBaseCalls - prev;

        pointsCount = getCorrectPointsCountVariable(chroma.baseCalls, pointsCount, endPos, i);

        qreal pxPerPoint = columnWidth / pointsCount;
        for (int j = 0; j < pointsCount; j++) {
            double x = columnWidth * (i - startPos) + columnWidth / 2 - (pointsCount - j) * pxPerPoint;
            qreal yA = -qMin(static_cast<qreal>(chroma.A[prev + j]) * areaHeight / chromaMax, h);
            qreal yC = -qMin(static_cast<qreal>(chroma.C[prev + j]) * areaHeight / chromaMax, h);
            qreal yG = -qMin(static_cast<qreal>(chroma.G[prev + j]) * areaHeight / chromaMax, h);
            qreal yT = -qMin(static_cast<qreal>(chroma.T[prev + j]) * areaHeight / chromaMax, h);
            polylineA.append(QPointF(x, yA));
            polylineC.append(QPointF(x, yC));
            polylineG.append(QPointF(x, yG));
            polylineT.append(QPointF(x, yT));
        }
        prev = chroma.baseCalls[i];
    }

    completePolygonsWithLastBaseCallTrace(polylineA, polylineC, polylineG, polylineT, chroma, columnWidth, visible, h);

    if (getSettings().drawTraceA) {
        p.setPen(getBaseColor('A'));
        p.drawPolyline(polylineA);
    }
    if (getSettings().drawTraceC) {
        p.setPen(getBaseColor('C'));
        p.drawPolyline(polylineC);
    }
    if (getSettings().drawTraceG) {
        p.setPen(getBaseColor('G'));
        p.drawPolyline(polylineG);
    }
    if (getSettings().drawTraceT) {
        p.setPen(getBaseColor('T'));
        p.drawPolyline(polylineT);
    }
    p.translate(- x, - h - y);
}

void SequenceWithChromatogramAreaRenderer::completePolygonsWithLastBaseCallTrace(QPolygonF& polylineA, QPolygonF& polylineC, QPolygonF& polylineG, QPolygonF& polylineT,
                                                               const DNAChromatogram& chroma, qreal columnWidth, const U2Region& visible, qreal h) const {
    //The last character may not to be included in visible area, so the trace for this symbol may be necessary to draw separately.
    int areaHeight = (heightPD - heightBC) * this->maxTraceHeight / 100;
    int startPos = visible.startPos;
    int endPos = visible.endPos();
    int prev = 0;
    int pointsCount = 0;
    if (endPos == chroma.baseCalls.size()) {
        prev = chroma.baseCalls.back();
        pointsCount = 2;
    } else {
        prev = chroma.baseCalls[endPos - 1];
        pointsCount = chroma.baseCalls[endPos] - prev;
        pointsCount = getCorrectPointsCountVariable(chroma.baseCalls, pointsCount, endPos, endPos - 1);
        pointsCount = pointsCount == 1 ? 2 : pointsCount;
    }
    qreal pxPerPoint = columnWidth / pointsCount;
    for (int i = 0; i < pointsCount; i++) {
        double x = columnWidth * (endPos - startPos) + columnWidth / 2 - (pointsCount - i) * pxPerPoint;
        if (chroma.A.size() == prev + i) {
            prev -= i;
        }
        qreal yA = -qMin(static_cast<qreal>(chroma.A[prev + i]) * areaHeight / chromaMax, h);
        qreal yC = -qMin(static_cast<qreal>(chroma.C[prev + i]) * areaHeight / chromaMax, h);
        qreal yG = -qMin(static_cast<qreal>(chroma.G[prev + i]) * areaHeight / chromaMax, h);
        qreal yT = -qMin(static_cast<qreal>(chroma.T[prev + i]) * areaHeight / chromaMax, h);

        polylineA.append(QPointF(x, yA));
        polylineC.append(QPointF(x, yC));
        polylineG.append(QPointF(x, yG));
        polylineT.append(QPointF(x, yT));
    }
}

void SequenceWithChromatogramAreaRenderer::drawOriginalBaseCalls(qreal h, QPainter& p, const U2Region& visible, const QByteArray& ba) const {
    p.setPen(Qt::black);
    p.translate( 0, h);

    int colWidth = getSeqArea()->getEditor()->getColumnWidth();
    for (int i = visible.startPos; i < visible.endPos(); i++) {
        QColor color = getBaseColor(ba[i]);
        p.setPen(color);

        int xP = colWidth * (i - visible.startPos) + colWidth / 2;

        p.setPen(linePen);
        p.setRenderHint(QPainter::Antialiasing, false);
        p.drawLine(xP, 0, xP, heightPD - h);
    }
    p.translate( 0, - h);
}

void SequenceWithChromatogramAreaRenderer::drawQualityValues(const DNAChromatogram& chroma, qreal w, qreal h,
                                                             QPainter& p, const U2Region& visible, const QByteArray& ba) const
{
    p.translate(0, h);

    //draw grid
    p.setPen(linePen);
    p.setRenderHint(QPainter::Antialiasing, false);
    for (int i = 0; i < 5; ++i) {
        p.drawLine(0, -h * i/4, w, -h * i/4);
    }

    QLinearGradient gradient(10, 0, 10, -h);
    gradient.setColorAt(0, Qt::green);
    gradient.setColorAt(0.33, Qt::yellow);
    gradient.setColorAt(0.66, Qt::red);
    QBrush brush(gradient);

    p.setBrush(brush);
    p.setPen(Qt::black);
    p.setRenderHint(QPainter::Antialiasing, true);

    int colWidth = getSeqArea()->getEditor()->getColumnWidth();
    QRectF rectangle;
    for (int i = visible.startPos; i < visible.endPos(); i++) {
        int xP = colWidth * (i - visible.startPos);
        switch (ba[i])  {
        case 'A':
            rectangle.setCoords(xP, 0, xP + charWidth, - h / 100 * chroma.prob_A[i]);
            break;
        case 'C':
            rectangle.setCoords(xP, 0, xP + charWidth, - h / 100 * chroma.prob_C[i]);
            break;
        case 'G':
            rectangle.setCoords(xP, 0, xP + charWidth, - h / 100 * chroma.prob_G[i]);
            break;
        case 'T':
            rectangle.setCoords(xP, 0, xP + charWidth, - h / 100 * chroma.prob_T[i]);
            break;
        }
        if (qAbs( rectangle.height() ) > h / 100) {
            p.drawRoundedRect(rectangle, 1.0, 1.0);
        }
    }

    p.translate( 0, - h);
}


void SequenceWithChromatogramAreaRenderer::drawChromatogramBaseCallsLines(const DNAChromatogram& chroma, qreal h, QPainter& p,
                                                                          const U2Region& visible, const QByteArray& ba) const
{
    p.setRenderHint(QPainter::Antialiasing, false);
    p.translate(0, h);

    double yRes = 0;
    int areaHeight = (heightPD - heightBC) * this->maxTraceHeight / 100;
    int colWidth = getSeqArea()->getEditor()->getColumnWidth();
    for (int i = visible.startPos; i < visible.startPos + visible.length; i++) {
        SAFE_POINT(i < chroma.baseCalls.length(), "Base calls array is too short: visible range index is out range", );
        int temp = chroma.baseCalls[i];
        SAFE_POINT(temp <= chroma.traceLength, "Broken chromatogram data", );

        double x = colWidth * (i - visible.startPos) + colWidth / 2;
        bool drawBase = true;
        p.setPen(getBaseColor(ba[i]));
        switch (ba[i])  {
            case 'A':
                yRes = -qMin(static_cast<qreal>(chroma.A[temp])*areaHeight/chromaMax, h);
                drawBase = getSettings().drawTraceA;
                break;
            case 'C':
                yRes = -qMin(static_cast<qreal>(chroma.C[temp]) * areaHeight / chromaMax, h);
                drawBase = getSettings().drawTraceC;
                break;
            case 'G':
                yRes = -qMin(static_cast<qreal>(chroma.G[temp]) * areaHeight / chromaMax, h);
                drawBase = getSettings().drawTraceG;
                break;
            case 'T':
                yRes = -qMin(static_cast<qreal>(chroma.T[temp]) * areaHeight / chromaMax, h);
                drawBase = getSettings().drawTraceT;
                break;
            case 'N':
                continue;
        };
        if (drawBase) {
            p.drawLine(x, 0, x, yRes);
        }
    }
    p.translate( 0, - h);
}

McaEditorSequenceArea* SequenceWithChromatogramAreaRenderer::getSeqArea() const {
    return qobject_cast<McaEditorSequenceArea*>(seqAreaWgt);
}

const ChromatogramViewSettings& SequenceWithChromatogramAreaRenderer::getSettings() const {
    return getSeqArea()->getSettings();
}

int SequenceWithChromatogramAreaRenderer::getChromatogramHeight() {
    return CHROMATOGRAM_MAX_HEIGHT * MaEditor::zoomMult;
}

} // namespace

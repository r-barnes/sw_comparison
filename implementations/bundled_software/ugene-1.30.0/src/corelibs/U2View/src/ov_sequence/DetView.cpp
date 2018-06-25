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

#include "DetView.h"
#include "DetViewSequenceEditor.h"

#include "ADVSequenceObjectContext.h"
#include "view_rendering/DetViewSingleLineRenderer.h"
#include "view_rendering/DetViewMultiLineRenderer.h"

#include <U2Core/AnnotationSettings.h>
#include <U2Core/AppContext.h>
#include <U2Core/Counter.h>
#include <U2Core/DNAAlphabet.h>
#include <U2Core/DNASequenceObject.h>
#include <U2Core/DNASequenceSelection.h>
#include <U2Core/DNATranslation.h>
#include <U2Core/DNATranslationImpl.h>
#include <U2Core/SignalBlocker.h>
#include <U2Core/TextUtils.h>
#include <U2Core/U1AnnotationUtils.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2SafePoints.h>

#include <U2Gui/GraphUtils.h>
#include <U2Gui/GScrollBar.h>
#include <U2Gui/SelectionModificationHelper.h>

#include <QApplication>
#include <QFontMetrics>
#include <QLayout>
#include <QMenu>
#include <QMessageBox>
#include <QPainter>
#include <QTextEdit>


namespace U2 {


/************************************************************************/
/* DetView */
/************************************************************************/
DetView::DetView(QWidget* p, SequenceObjectContext* ctx)
    : GSequenceLineViewAnnotated(p, ctx)
{
    editor = new DetViewSequenceEditor(this);

    showComplementAction = new QAction(tr("Show complementary strand"), this);
    showComplementAction->setIcon(QIcon(":core/images/show_compl.png"));
    showComplementAction->setObjectName("complement_action");
    connect(showComplementAction, SIGNAL(triggered(bool)), SLOT(sl_showComplementToggle(bool)));

    showTranslationAction = new QAction(tr("Show/hide translations"), this);
    showTranslationAction->setObjectName("translation_action");
    connect(showTranslationAction, SIGNAL(triggered(bool)), SLOT(sl_showTranslationToggle(bool)));

    doNotTranslateAction = new QAction(tr("Do not translate"), this);
    doNotTranslateAction->setObjectName("do_not_translate_radiobutton");
    doNotTranslateAction->setData(SequenceObjectContext::TS_DoNotTranslate);
    connect(doNotTranslateAction, SIGNAL(triggered(bool)), SLOT(sl_doNotTranslate()));
    doNotTranslateAction->setCheckable(true);
    doNotTranslateAction->setChecked(true);

    translateAnnotationsOrSelectionAction = new QAction(tr("Translate selection"), this);
    translateAnnotationsOrSelectionAction->setData(SequenceObjectContext::TS_AnnotationsOrSelection);
    connect(translateAnnotationsOrSelectionAction, SIGNAL(triggered(bool)), SLOT(sl_translateAnnotationsOrSelection()));
    translateAnnotationsOrSelectionAction->setCheckable(true);

    setUpFramesManuallyAction = new QAction(tr("Set up frames manually"), this);
    setUpFramesManuallyAction->setObjectName("set_up_frames_manuallt_radiobutton");
    setUpFramesManuallyAction->setData(SequenceObjectContext::TS_SetUpFramesManually);
    connect(setUpFramesManuallyAction, SIGNAL(triggered(bool)), SLOT(sl_setUpFramesManually()));
    setUpFramesManuallyAction->setCheckable(true);

    showAllFramesAction = new QAction(tr("Show all frames"), this);
    showAllFramesAction->setObjectName("show_all_frames_radiobutton");
    showAllFramesAction->setData(SequenceObjectContext::TS_ShowAllFrames);
    connect(showAllFramesAction, SIGNAL(triggered(bool)), SLOT(sl_showAllFrames()));
    showAllFramesAction->setCheckable(true);

    wrapSequenceAction = new QAction(tr("Wrap sequence"), this);
    wrapSequenceAction->setIcon(QIcon(":core/images/wrap_sequence.png"));
    wrapSequenceAction->setObjectName("wrap_sequence_action");
    connect(wrapSequenceAction, SIGNAL(triggered(bool)), SLOT(sl_wrapSequenceToggle(bool)));

    showComplementAction->setCheckable(true);
    showTranslationAction->setCheckable(true);
    wrapSequenceAction->setCheckable(true);
    wrapSequenceAction->setChecked(true);

    bool hasComplement = ctx->getComplementTT() != NULL;
    showComplementAction->setChecked(hasComplement);

    bool hasAmino = ctx->getAminoTT() != NULL;
    showTranslationAction->setChecked(hasAmino);

    assert(ctx->getSequenceObject()!=NULL);
    featureFlags&=!GSLV_FF_SupportsCustomRange;
    renderArea = new DetViewRenderArea(this);
    renderArea->setObjectName("render_area_" + ctx->getSequenceObject()->getSequenceName());

    connect(ctx, SIGNAL(si_aminoTranslationChanged()), SLOT(sl_onAminoTTChanged()));
    connect(ctx, SIGNAL(si_translationRowsChanged()), SLOT(sl_translationRowsChanged()));

    addActionToLocalToolbar(wrapSequenceAction);
    if (hasComplement) {
        addActionToLocalToolbar(showComplementAction);
    }
    if (hasAmino) {
        setupTranslationsMenu();
        setupGeneticCodeMenu();
    }
    addActionToLocalToolbar(editor->getEditAction());

    verticalScrollBar = new GScrollBar(Qt::Vertical, this);
    verticalScrollBar->setObjectName("multiline_scrollbar");
    scrollBar->setObjectName("singleline_scrollbar");
    currentShiftsCounter = 0;
    numShiftsInOneLine = 1;

    verticalScrollBar->setHidden(!wrapSequenceAction->isChecked());
    scrollBar->setHidden(wrapSequenceAction->isChecked());

    pack();

    updateActions();

    // TODO_SVEDIT: check its required
    connect(ctx->getSequenceObject(), SIGNAL(si_sequenceChanged()), SLOT(sl_sequenceChanged()));

    setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
}
DetView::~DetView() {
    removeEventFilter(editor);
}

DetViewRenderArea* DetView::getDetViewRenderArea() const {
    return static_cast<DetViewRenderArea*>(renderArea);
}

bool DetView::hasTranslations() const {
    return getAminoTT() != NULL;
}

bool DetView::hasComplementaryStrand() const {
    return getComplementTT() != NULL;
}

bool DetView::isWrapMode() const {
    return wrapSequenceAction->isChecked();
}

bool DetView::isEditMode() const {
    return editor->isEditMode();
}

void DetView::setStartPos(qint64 newPos) {
    if (newPos + visibleRange.length > seqLen && !isWrapMode()) {
        newPos = seqLen - visibleRange.length;
    }
    if (newPos < 0) {
        newPos = 0;
    }

    if (visibleRange.startPos != newPos) {
        visibleRange.startPos = newPos;
        updateVisibleRange();
    }
}

void DetView::setCenterPos(qint64 pos) {
    if (!isWrapMode()) {
        GSequenceLineView::setCenterPos(pos);
        return;
    }

    DetViewRenderArea* detArea = getDetViewRenderArea();
    qint64 line = pos / detArea->getSymbolsPerLine();

    qint64 newPos = (line - detArea->getLinesCount()/ 2) * detArea->getSymbolsPerLine();
    currentShiftsCounter = 0;
    setStartPos(newPos);
}

DNATranslation* DetView::getComplementTT() const {
    return showComplementAction->isChecked() ? ctx->getComplementTT() : NULL;
}

DNATranslation* DetView::getAminoTT() const {
    return !doNotTranslateAction->isChecked() ? ctx->getAminoTT() : NULL;
}

int DetView::getSymbolsPerLine() const {
    return getDetViewRenderArea()->getSymbolsPerLine();
}

void DetView::setShowComplement(bool t) {
    showComplementAction->disconnect(this);
    showComplementAction->setChecked(t);
    ctx->showComplementActions(t);
    connect(showComplementAction, SIGNAL(triggered(bool)), SLOT(sl_showComplementToggle(bool)));

    updateSize();
    updateVisibleRange();
}

void DetView::setShowTranslation(bool t) {
    showTranslationAction->disconnect(this);
    showTranslationAction->setChecked(t);
    getSequenceContext()->setTranslationsVisible(t);
    connect(showTranslationAction, SIGNAL(triggered(bool)), SLOT(sl_showTranslationToggle(bool)));

    updateSize();
    updateVisibleRange();
}

void DetView::setDisabledDetViewActions(bool t){
    showTranslationAction->setDisabled(t);
    showComplementAction->setDisabled(t);
    wrapSequenceAction->setDisabled(t);
}

int DetView::getShift() const {
    return isWrapMode() ? currentShiftsCounter * getDetViewRenderArea()->getShiftHeight() : 0;
}

void DetView::ensurePositionVisible(int pos) {
    CHECK(pos >= 0 && pos <= getSequenceLength(), );

    DetViewRenderArea* detViewRenderArea = getDetViewRenderArea();
    if (isWrapMode()) {
        if (!visibleRange.contains(pos)) {
            if (pos < visibleRange.startPos) {
                // scroll up till the line is visible
                int line = pos / getSymbolsPerLine();
                int listStartPos = line * getSymbolsPerLine();
                visibleRange.startPos = listStartPos;
                currentShiftsCounter = detViewRenderArea->getDirectLine() + 1;
            } else {
                // scroll down till the line becomes visible
                const int line = pos / getSymbolsPerLine();
                const int visibleShiftsCount = detViewRenderArea->height() / detViewRenderArea->getShiftHeight();
                const int shiftToEnsureIsVisible = line * numShiftsInOneLine + detViewRenderArea->getDirectLine() + 1;
                const int lastVisibleShift = verticalScrollBar->value() + visibleShiftsCount;
                const int shiftsToScrollDown = shiftToEnsureIsVisible - lastVisibleShift + 1;

                currentShiftsCounter += shiftsToScrollDown;
                visibleRange.startPos += getSymbolsPerLine() * (currentShiftsCounter / numShiftsInOneLine);
                currentShiftsCounter %= numShiftsInOneLine;
            }
        } else {
            // ensure the direct strand is visible
            if (currentShiftsCounter > detViewRenderArea->getDirectLine() &&
                    U2Region(visibleRange.startPos, getSymbolsPerLine()).contains(pos)) {
                // this is the first line, just remove the shift
                currentShiftsCounter = detViewRenderArea->getDirectLine() + 1;
            }

            if (U2Region(visibleRange.endPos() - getSymbolsPerLine(), getSymbolsPerLine()).contains(pos)) {
                // the last visible line, scroll a little
                // TODO_SVEDIT: check the tail
                int availableSpace = detViewRenderArea->height() + currentShiftsCounter * detViewRenderArea->getShiftHeight();
                int lastLinePart = availableSpace % (numShiftsInOneLine * detViewRenderArea->getShiftHeight());
                int lastVisibleShifts = lastLinePart / detViewRenderArea->getShiftHeight();

                if (lastVisibleShifts <= detViewRenderArea->getDirectLine()) {
                    currentShiftsCounter += detViewRenderArea->getDirectLine() - lastVisibleShifts + 2;
                }
                if (currentShiftsCounter > numShiftsInOneLine) {
                    currentShiftsCounter %= numShiftsInOneLine;
                    visibleRange.startPos += getSymbolsPerLine();
                }
            }
            // do nothing otherwise -- the cursor is visible
        }
    } else {
        // ensure the direct strand is visible
        const int visibleShiftsCount = detViewRenderArea->height() / detViewRenderArea->getShiftHeight();
        const int shiftToEnsureIsVisible = detViewRenderArea->getDirectLine() + 1;
        const int firstVisibleShift = verticalScrollBar->value();
        const int lastVisibleShift = verticalScrollBar->value() + visibleShiftsCount;

        if (shiftToEnsureIsVisible < firstVisibleShift) {
            // the direct line is not visible, scroll up till the line becomes visible
            const int shiftsToScrollUp = firstVisibleShift - shiftToEnsureIsVisible + 1;
            verticalScrollBar->setValue(verticalScrollBar->value() - shiftsToScrollUp);
        } else if (shiftToEnsureIsVisible > lastVisibleShift) {
            // the direct line is not visible, scroll down till the line becomes visible
            const int shiftsToScrollDown = shiftToEnsureIsVisible - lastVisibleShift + 1;
            verticalScrollBar->setValue(verticalScrollBar->value() + shiftsToScrollDown - 1);
        }

        // ensure horizontal position is visible
        CHECK(!visibleRange.contains(pos), );
        if (pos < visibleRange.startPos) {
            visibleRange.startPos = pos;
        } else {
            visibleRange.startPos = pos - visibleRange.length;
        }
    }

    updateVisibleRange();
}

void DetView::sl_sequenceChanged() {
    seqLen = ctx->getSequenceLength();
    updateVerticalScrollBar();
    updateScrollBar();
    updateVisibleRange();
    GSequenceLineView::sl_sequenceChanged();
}

void DetView::sl_onDNASelectionChanged(LRegionsSelection* sel, const QVector<U2Region>& added, const QVector<U2Region>& removed) {
    GSequenceLineViewAnnotated::sl_onDNASelectionChanged(sel, added, removed);
    setSelectedTranslations();
}

void DetView::sl_onAminoTTChanged() {
    lastUpdateFlags |= GSLV_UF_NeedCompleteRedraw;
    update();
}

void DetView::sl_translationRowsChanged() {
    QVector<bool> visibleRows = getSequenceContext()->getTranslationRowsVisibleStatus();
    bool anyFrame = false;
    foreach(bool b, visibleRows){
        anyFrame = anyFrame || b;
    }
    if (!anyFrame){
        if(showTranslationAction->isChecked()) {
            sl_showTranslationToggle(false);
        }
        return;
    }
    if (!showTranslationAction->isChecked()) {
        if(!getSequenceContext()->isRowChoosed()){
            sl_showTranslationToggle(true);
        }
        else{
            showTranslationAction->setChecked(true);
        }
    }

    updateScrollBar();
    updateVerticalScrollBar();
    updateSize();
    completeUpdate();
}

void DetView::sl_showComplementToggle(bool v) {
    GCOUNTER( cvar, tvar, "SequenceView::DetView::ShowComplement" );
    setShowComplement(v);
}

void DetView::sl_showTranslationToggle(bool v) {
    GCOUNTER( cvar, tvar, "SequenceView::DetView::ShowTranslations" );
    setShowTranslation(v);
}

void DetView::sl_doNotTranslate() {
    updateSelectedTranslations(SequenceObjectContext::TS_DoNotTranslate);
}

void DetView::sl_translateAnnotationsOrSelection() {
    updateSelectedTranslations(SequenceObjectContext::TS_AnnotationsOrSelection);
}

void DetView::sl_setUpFramesManually() {
    updateSelectedTranslations(SequenceObjectContext::TS_SetUpFramesManually);
}

void DetView::sl_showAllFrames() {
    updateSelectedTranslations(SequenceObjectContext::TS_ShowAllFrames);
}

void DetView::updateSelectedTranslations(const SequenceObjectContext::TranslationState& state) {
    ctx->setTranslationState(state);
    setSelectedTranslations();
}

void DetView::sl_wrapSequenceToggle(bool v) {
    GCOUNTER( cvar, tvar, "SequenceView::DetView::WrapSequence" );
    // turn off/on multiline mode
    scrollBar->setHidden(v);
    verticalScrollBar->setVisible(v);

    currentShiftsCounter = 0;

    DetViewRenderArea* detArea = getDetViewRenderArea();
    detArea->setWrapSequence(v);
    updateVerticalScrollBar();
    updateScrollBar();

    updateVisibleRange();
    updateSize();

    addUpdateFlags(GSLV_UF_NeedCompleteRedraw);
    completeUpdate();
    if (!v) {
        verticalScrollBar->setSliderPosition(0);
    }
}

void DetView::sl_verticalSrcollBarMoved(int pos) {
    if (isWrapMode()) {
        currentShiftsCounter = pos % numShiftsInOneLine;
        DetViewRenderArea* detArea = getDetViewRenderArea();
        if (pos / numShiftsInOneLine == visibleRange.startPos / detArea->getSymbolsPerLine()) {
            updateVisibleRange();
            completeUpdate();
            return;
        }
        setStartPos((pos / numShiftsInOneLine) * detArea->getSymbolsPerLine());
    } else {
        updateVisibleRange();
    }
}

void DetView::pack() {
    QGridLayout* layout = new QGridLayout();
    layout->setMargin(0);
    layout->setSpacing(0);
    layout->addWidget(renderArea, 0, 0);
    layout->addWidget(scrollBar, 1, 0);
    layout->addWidget(verticalScrollBar, 0, 1);

    setContentLayout(layout);
    setMinimumHeight(scrollBar->height() + renderArea->minimumHeight());
}

void DetView::showEvent(QShowEvent * e) {
    updateVerticalScrollBar();
    updateVisibleRange();
    updateScrollBar();
    updateActions();
    GSequenceLineViewAnnotated::showEvent(e);
}

void DetView::hideEvent(QHideEvent * e) {
    updateActions();
    GSequenceLineViewAnnotated::hideEvent(e);
}

void DetView::uncheckAllTranslations() {
    for (int i = 0; i < 6; i++) {
        ctx->showTranslationFrame(i, false);
    }
}

void DetView::setSelectedTranslations() {
    if ((ctx->getTranslationState() == SequenceObjectContext::TS_AnnotationsOrSelection)) {
        uncheckAllTranslations();
        updateTranslationsState();
    }

    getDetViewRenderArea()->getRenderer()->update();
    updateVisibleRange();
    updateVerticalScrollBar();
    completeUpdate();
}

void DetView::updateTranslationsState() {
    updateTranslationsState(U2Strand::Direct);
    updateTranslationsState(U2Strand::Complementary);
}

void DetView::updateTranslationsState(const U2Strand::Direction direction) {
    QVector<U2Region> selectedRegions = ctx->getSequenceSelection()->getSelectedRegions();
    QList<bool> lineState = QList<bool>() << false << false << false;
    foreach(const U2Region& reg, selectedRegions) {
        int mod = direction == U2Strand::Direct ? reg.startPos % 3 : ((ctx->getSequenceLength() - reg.endPos()) % 3);
        lineState[mod] = true;
    }
    const int start = direction == U2Strand::Direct ? 0 : 3;
    const int end = direction == U2Strand::Direct ? 3 : 6;
    const int indent = direction == U2Strand::Direct ? 0 : 3;
    for (int i = start; i < end; i++) {
        const bool state = lineState[i - indent];
        if (!state) {
            continue;
        }
        ctx->showTranslationFrame(i, state);
    }
}

void DetView::mouseMoveEvent(QMouseEvent *me) {
    if (!me->buttons()) {
        setBorderCursor(me->pos());
    }

    if (isSelectionResizing) {
        if (me->buttons() & Qt::LeftButton) {
            Qt::CursorShape shape = cursor().shape();
            if (shape != Qt::ArrowCursor) {
                moveBorder(me->pos());
                QWidget::mouseMoveEvent(me);
                return;
            }
        }

        if (lastPressPos == -1) {
            QWidget::mouseMoveEvent(me);
            return;
        }
        if (me->buttons() & Qt::LeftButton) {
            moveBorder(me->pos());
        }
    }
    setSelectedTranslations();

    QWidget::mouseMoveEvent(me);
}

void DetView::mouseReleaseEvent(QMouseEvent* me) {
    //click with 'alt' shift selects single base in GSingleSeqWidget;
    //here we adjust this behavior -> if click was done in translation line -> select 3 bases
    Qt::KeyboardModifiers km = QApplication::keyboardModifiers();
    bool singleBaseSelectionMode = km.testFlag(Qt::AltModifier);
    if (me->button() == Qt::LeftButton && singleBaseSelectionMode) {
        QPoint areaPoint = toRenderAreaPoint(me->pos());
        DetViewRenderArea* detArea = getDetViewRenderArea();
        if (detArea->isOnTranslationsLine(areaPoint)) {
            qint64 pos = detArea->coordToPos(areaPoint);
            if (pos == lastPressPos) {
                U2Region rgn(pos - 1, 3);
                if (rgn.startPos >= 0 && rgn.endPos() <= seqLen) {
                    setSelection(rgn);
                    lastPressPos = -1;
                }
            }
        }
    }
    setSelectedTranslations();
    verticalScrollBar->setupRepeatAction(QAbstractSlider::SliderNoAction);
    GSequenceLineViewAnnotated::mouseReleaseEvent(me);
}

void DetView::wheelEvent(QWheelEvent *we) {
    bool renderAreaWheel = QRect(renderArea->x(), renderArea->y(), renderArea->width(), renderArea->height()).contains(we->pos());
    if (!renderAreaWheel) {
        QWidget::wheelEvent(we);
        return;
    }
    setFocus();

    bool toMin = we->delta() > 0;
    if (we->modifiers() == 0) {
        // clear wheel event
        GScrollBar* sBar = wrapSequenceAction->isChecked() ? verticalScrollBar : scrollBar;
        sBar->triggerAction(toMin ? QAbstractSlider::SliderSingleStepSub : QAbstractSlider::SliderSingleStepAdd);
    }
    setSelectedTranslations();
}

void DetView::resizeEvent(QResizeEvent *e) {
    updateVerticalScrollBar();
    updateVisibleRange();

    addUpdateFlags(GSLV_UF_ViewResized);
    GSequenceLineView::resizeEvent(e);
}

void DetView::keyPressEvent(QKeyEvent *e) {
    int key = e->key();
    bool accepted = false;
    GScrollBar* sBar = isWrapMode() ? verticalScrollBar : scrollBar;
    switch(key) {
    case Qt::Key_Left:
    case Qt::Key_Up:
        if (isWrapMode()) {
            verticalScrollBar->triggerAction(QAbstractSlider::SliderSingleStepSub);
        } else {
            setStartPos(qMax(qint64(0), visibleRange.startPos - 1));
        }
        accepted = true;
        break;
    case Qt::Key_Right:
    case Qt::Key_Down:
        if (isWrapMode()) {
            verticalScrollBar->triggerAction(QAbstractSlider::SliderSingleStepAdd);
        } else {
            setStartPos(qMin(seqLen - 1, visibleRange.startPos + 1));
        }
        accepted = true;
        break;
    case Qt::Key_Home:
        setStartPos(0);
        currentShiftsCounter = 0;
        accepted = true;
        break;
    case Qt::Key_End:
        setStartPos(seqLen - 1);
        accepted = true;
        break;
    case Qt::Key_PageUp:
        sBar->triggerAction(QAbstractSlider::SliderPageStepSub);
        accepted = true;
        break;
    case Qt::Key_PageDown:
        sBar->triggerAction(QAbstractSlider::SliderPageStepAdd);
        accepted = true;
        break;
    }
    if (accepted) {
        e->accept();
    } else {
        QWidget::keyPressEvent(e);
    }
}

void DetView::updateVisibleRange() {
    DetViewRenderArea* detArea = getDetViewRenderArea();
    if (isWrapMode()) {
        if (visibleRange.startPos % detArea->getSymbolsPerLine() != 0) {
            // shift to the nearest line break
            visibleRange.startPos = detArea->getSymbolsPerLine() * (int)(visibleRange.startPos / detArea->getSymbolsPerLine());
        }

        int visibleLinesCount = detArea->getLinesCount() + (getShift() != 0 ? 1 : 0);
        if (detArea->height() + getShift() - detArea->getShiftsCount() * detArea->getShiftHeight() * visibleLinesCount > 0) {
            visibleLinesCount ++;
        }

        int visibleRangeLen = visibleLinesCount * detArea->getSymbolsPerLine();

        int lastLine = seqLen / detArea->getSymbolsPerLine() + (seqLen % detArea->getSymbolsPerLine() == 0 ? 0 : 1) - detArea->getLinesCount();
        if (detArea->height() - detArea->getLinesCount() * detArea->getShiftsCount() * detArea->getShiftHeight() > 0 && lastLine > 0) {
            lastLine --;
        }
        int lastStartPos = lastLine * detArea->getSymbolsPerLine();

        bool posAtTheEnd = visibleRange.startPos > lastStartPos;
        visibleRange.length = qMin((int)(seqLen - visibleRange.startPos), qMin(visibleRangeLen, (int)seqLen));
        bool emptyLineDetected = (visibleRangeLen - visibleRange.length) > detArea->getSymbolsPerLine();

        if (posAtTheEnd || (emptyLineDetected && visibleRange.startPos + visibleRangeLen >= seqLen)) {
            // scroll to the end
            visibleRange.startPos = qMax(0, (verticalScrollBar->maximum() / numShiftsInOneLine) * detArea->getSymbolsPerLine());
            visibleRange.length = qMin((int)(seqLen - visibleRange.startPos), qMin(visibleRangeLen, (int)seqLen));
            currentShiftsCounter = qMax(0, verticalScrollBar->maximum() % numShiftsInOneLine);
        }
    } else {
        visibleRange.length = qMin((qint64)detArea->getVisibleSymbolsCount(), seqLen);
        visibleRange.startPos = qMin(visibleRange.startPos, seqLen - visibleRange.length);
    }

    SAFE_POINT(visibleRange.startPos >= 0 && visibleRange.endPos() <= seqLen, "Visible range is out of sequence range", );

    updateVerticalScrollBarPosition();
    onVisibleRangeChanged();
}

void DetView::updateActions() {
    bool hasComplement = ctx->getComplementTT() != NULL;
    showComplementAction->setEnabled(hasComplement);

    bool hasAmino = ctx->getAminoTT() != NULL;
    showTranslationAction->setEnabled(hasAmino);
}

void DetView::updateSize() {
    addUpdateFlags(GSLV_UF_ViewResized);

    DetViewRenderArea* detArea = getDetViewRenderArea();
    detArea->updateSize();

    updateVerticalScrollBar();
}

void DetView::updateVerticalScrollBar() {
    verticalScrollBar->disconnect(this);

    DetViewRenderArea* renderer = getDetViewRenderArea();
    bool wasVisible = verticalScrollBar->isVisible();

    const int shiftsCount = renderer->getShiftsCount();
    const int shiftsHeight = renderer->getShiftHeight();
    const int height = renderer->height();
    bool isSingleViewLineNotVisible = !isWrapMode() && (height < shiftsCount * shiftsHeight);
    verticalScrollBar->setVisible(isSingleViewLineNotVisible || isWrapMode());
    if (wasVisible && verticalScrollBar->isHidden()) {
        verticalScrollBar->setSliderPosition(0);
    }

    int maximum = 0;
    if (isWrapMode()) {
        DetViewRenderArea* detArea = getDetViewRenderArea();
        int linesCount = seqLen / detArea->getSymbolsPerLine();
        if (seqLen % detArea->getSymbolsPerLine() != 0) {
            linesCount++;
        }

        numShiftsInOneLine = getDetViewRenderArea()->getShiftsCount();
        int shiftsOnWidget = renderArea->height() / getDetViewRenderArea()->getShiftHeight();
        maximum = numShiftsInOneLine * linesCount - shiftsOnWidget;
    } else if (isSingleViewLineNotVisible) {
        numShiftsInOneLine = 1;
        const int shiftsVisible = height / shiftsHeight;
        maximum = shiftsCount - shiftsVisible;
        const int sliderPosition = verticalScrollBar->sliderPosition();
        const int currentMaximum = verticalScrollBar->maximum();
        if (sliderPosition == currentMaximum && currentMaximum > maximum) {
            const int newSliderPos = sliderPosition - (currentMaximum - maximum);
            verticalScrollBar->setSliderPosition(newSliderPos);
        }
    } else {
        maximum = 0;
        numShiftsInOneLine = 0;
    }

    verticalScrollBar->setMinimum(0);
    verticalScrollBar->setMaximum(maximum);
    verticalScrollBar->setPageStep(numShiftsInOneLine);
    updateVerticalScrollBarPosition();
    connect(verticalScrollBar, SIGNAL(valueChanged(int)), SLOT(sl_verticalSrcollBarMoved(int)));
}

void DetView::updateVerticalScrollBarPosition() {
    if (isWrapMode()) {
        DetViewRenderArea* detArea = getDetViewRenderArea();
        SignalBlocker blocker(verticalScrollBar);
        Q_UNUSED(blocker);
        verticalScrollBar->setSliderPosition(qMin(verticalScrollBar->maximum(),
            currentShiftsCounter + numShiftsInOneLine * int(visibleRange.startPos / detArea->getSymbolsPerLine())));
    }
}

void DetView::setupTranslationsMenu() {
    QMenu *translationsMenu = ctx->createTranslationFramesMenu(QList<QAction*>() << doNotTranslateAction << translateAnnotationsOrSelectionAction << setUpFramesManuallyAction << showAllFramesAction);
    CHECK(NULL != translationsMenu, );
    QToolButton *button = addActionToLocalToolbar(translationsMenu->menuAction());
    button->setPopupMode(QToolButton::InstantPopup);
    button->setObjectName("translationsMenuToolbarButton");
}

void DetView::setupGeneticCodeMenu() {
    QMenu *ttMenu = ctx->createGeneticCodeMenu();
    CHECK(NULL != ttMenu, );
    QToolButton *button = addActionToLocalToolbar(ttMenu->menuAction());
    SAFE_POINT(button, QString("ToolButton for %1 is NULL").arg(ttMenu->menuAction()->objectName()), );
    button->setPopupMode(QToolButton::InstantPopup);
    button->setObjectName("AminoToolbarButton");
}

int DetView::getVerticalScrollBarPosition() {
    if (!isWrapMode()) {
        return verticalScrollBar->sliderPosition();
    }
    return 0;
}

QPoint DetView::getRenderAreaPointAfterAutoScroll(const QPoint& pos) {
    QPoint areaPoint = toRenderAreaPoint(pos);
    if (isWrapMode()) {
        if (areaPoint.y() > height()) {
            verticalScrollBar->setupRepeatAction(QAbstractSlider::SliderSingleStepAdd);
        } else if (areaPoint.y() <= 0) {
            verticalScrollBar->setupRepeatAction(QAbstractSlider::SliderSingleStepSub);
        } else {
            verticalScrollBar->setupRepeatAction(QAbstractSlider::SliderNoAction);
        }
    } else {
        if (areaPoint.x() > width()) {
            scrollBar->setupRepeatAction(QAbstractSlider::SliderSingleStepAdd);
        } else if (areaPoint.x() <= 0) {
            scrollBar->setupRepeatAction(QAbstractSlider::SliderSingleStepSub);
        } else {
            scrollBar->setupRepeatAction(QAbstractSlider::SliderNoAction);
        }
    }

    if (isWrapMode() && (areaPoint.x() > width() || areaPoint.x() <= 0)) {
        areaPoint = QPoint(areaPoint.x() > width() ? width() : 0, areaPoint.y());
    }

    return areaPoint;
}

void DetView::moveBorder(const QPoint& p) {
    QPoint areaPoint = getRenderAreaPointAfterAutoScroll(p);
    resizeSelection(areaPoint);
}

void DetView::setBorderCursor(const QPoint& p) {
    if (!isWrapMode()) {
        GSequenceLineView::setBorderCursor(p);
    } else {
        const int sliderPos = verticalScrollBar->sliderPosition();
        const int shiftHeight = getDetViewRenderArea()->getShiftHeight();
        const int lineHeight = numShiftsInOneLine * shiftHeight;
        const int globalY = p.y() + sliderPos * shiftHeight;
        const int numLines = qRound(double(globalY) / lineHeight + 0.5);

        const double scale = renderArea->getCurrentScale();
        const int width = getRenderArea()->width();
        const int symPerLine = width / scale;
        const int correctWidth = symPerLine * scale;

        const int offset = correctWidth * (numLines - 1);
        CHECK(offset >= 0, );

        QPoint pos(p.x() + offset, p.y());
        GSequenceLineView::setBorderCursor(pos);
    }
}

/************************************************************************/
/* DetViewRenderArea */
/************************************************************************/
DetViewRenderArea::DetViewRenderArea(DetView* v)
    : GSequenceLineViewAnnotatedRenderArea(v, true) {
    renderer = DetViewRendererFactory::createRenderer(getDetView(), view->getSequenceContext(), v->isWrapMode());
    setMouseTracking(true);
    updateSize();
}

DetViewRenderArea::~DetViewRenderArea() {
    delete renderer;
}

void DetViewRenderArea::setWrapSequence(bool v) {
    delete renderer;
    renderer = DetViewRendererFactory::createRenderer(getDetView(), view->getSequenceContext(), v);

    updateSize();
}

U2Region DetViewRenderArea::getAnnotationYRange(Annotation* a, int r, const AnnotationSettings* as) const {
    U2Region absoluteRegion = renderer->getAnnotationYRange(a, r, as, size(), view->getVisibleRange());
    absoluteRegion.startPos += renderer->getContentIndentY(size(), view->getVisibleRange());
    return  absoluteRegion;
}

bool DetViewRenderArea::isOnTranslationsLine(const QPoint& p) const {
    return renderer->isOnTranslationsLine(p, size(), view->getVisibleRange());
}

bool DetViewRenderArea::isPosOnAnnotationYRange(const QPoint &p, Annotation *a, int region, const AnnotationSettings *as) const {
    int scrollShift = getDetView()->getShift();
    QPoint pShifted(p.x(), p.y() + scrollShift);

    QSize expandedSize = size();
    expandedSize.setHeight(expandedSize.height() + scrollShift);
    return renderer->isOnAnnotationLine(pShifted, a, region, as, expandedSize, view->getVisibleRange());
}

void DetViewRenderArea::drawAll(QPaintDevice* pd) {
    GSLV_UpdateFlags uf = view->getUpdateFlags();
    bool completeRedraw = uf.testFlag(GSLV_UF_NeedCompleteRedraw)  || uf.testFlag(GSLV_UF_ViewResized)  ||
                          uf.testFlag(GSLV_UF_VisibleRangeChanged) || uf.testFlag(GSLV_UF_AnnotationsChanged);

    int scrollShift = getDetView()->getShift();
    QSize canvasSize(pd->width(), pd->height() + scrollShift);

    if (completeRedraw) {
        QPainter pCached(cachedView);
        pCached.translate(0, - scrollShift);
        renderer->drawAll(pCached, canvasSize, view->getVisibleRange());
        pCached.end();
    }

    QPainter p(pd);

    p.drawPixmap(0, 0, *cachedView);
    p.translate(0, - scrollShift);
    renderer->drawSelection(p, canvasSize, view->getVisibleRange());
    renderer->drawCursor(p, canvasSize, view->getVisibleRange());
    p.translate(0, scrollShift);

    if (view->hasFocus()) {
        drawFocus(p);
    }
}

qint64 DetViewRenderArea::coordToPos(const QPoint& p) const {
    QPoint pShifted(p.x(), p.y() + getDetView()->getShift());
    return renderer->coordToPos(pShifted, QSize(width(), height()), view->getVisibleRange());
}

double DetViewRenderArea::getCurrentScale() const {
    return renderer->getCurrentScale();
}

void DetViewRenderArea::updateSize()  {
    renderer->update();
    setMinimumHeight(renderer->getMinimumHeight());
    repaint();
}

DetView* DetViewRenderArea::getDetView() const {
    return static_cast<DetView*>(view);
}

int DetViewRenderArea::getSymbolsPerLine() const {
    return width() / charWidth;
}

int DetViewRenderArea::getLinesCount() const {
    return renderer->getLinesCount(size());
}

int DetViewRenderArea::getVisibleSymbolsCount() const {
    return getLinesCount() * getSymbolsPerLine();
}

int DetViewRenderArea::getDirectLine() const {
    return renderer->getDirectLine();
}

int DetViewRenderArea::getShiftsCount() const {
    return renderer->getRowsInLineCount();
}

int DetViewRenderArea::getShiftHeight() const {
    return renderer->getRowLineHeight();
}

} // namespace U2

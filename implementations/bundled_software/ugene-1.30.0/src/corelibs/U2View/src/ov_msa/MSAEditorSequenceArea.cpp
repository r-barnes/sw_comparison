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
#include <QClipboard>
#include <QDialog>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QTextStream>
#include <QWidgetAction>

#include <U2Algorithm/CreateSubalignmentTask.h>
#include <U2Algorithm/MsaColorScheme.h>
#include <U2Algorithm/MsaHighlightingScheme.h>

#include <U2Core/AddSequencesToAlignmentTask.h>
#include <U2Core/AppContext.h>
#include <U2Core/ClipboardController.h>
#include <U2Core/DNAAlphabet.h>
#include <U2Core/DNASequenceObject.h>
#include <U2Core/DNATranslation.h>
#include <U2Core/DocumentModel.h>
#include <U2Core/GUrlUtils.h>
#include <U2Core/IOAdapter.h>
#include <U2Core/IOAdapterUtils.h>
#include <U2Core/L10n.h>
#include <U2Core/MultipleSequenceAlignment.h>
#include <U2Core/MultipleSequenceAlignmentObject.h>
#include <U2Core/MSAUtils.h>
#include <U2Core/MsaDbiUtils.h>
#include <U2Core/MultiTask.h>
#include <U2Core/ProjectModel.h>
#include <U2Core/QObjectScopedPointer.h>
#include <U2Core/SaveDocumentTask.h>
#include <U2Core/Settings.h>
#include <U2Core/Settings.h>
#include <U2Core/Task.h>
#include <U2Core/TaskSignalMapper.h>
#include <U2Core/TaskWatchdog.h>
#include <U2Core/U2AlphabetUtils.h>
#include <U2Core/U2ObjectDbi.h>
#include <U2Core/U2OpStatusUtils.h>
#include <U2Core/U2SafePoints.h>
#include <U2Core/U2SequenceUtils.h>

#include <U2Formats/DocumentFormatUtils.h>

#include <U2Gui/AppSettingsGUI.h>
#include <U2Gui/DialogUtils.h>
#include <U2Gui/GUIUtils.h>
#include <U2Gui/LastUsedDirHelper.h>
#include <U2Gui/MainWindow.h>
#include <U2Gui/Notification.h>
#include <U2Gui/OPWidgetFactory.h>
#include <U2Gui/OptionsPanel.h>
#include <U2Gui/PositionSelector.h>
#include <U2Gui/ProjectTreeController.h>
#include <U2Gui/ProjectTreeItemSelectorDialog.h>

#include "ColorSchemaSettingsController.h"
#include "CreateSubalignmentDialogController.h"
#include "ExportSequencesTask.h"
#include "MaEditorNameList.h"
#include "MSAEditor.h"
#include "MSAEditorSequenceArea.h"
#include "Highlighting/MsaSchemesMenuBuilder.h"

#include "AlignSequencesToAlignment/AlignSequencesToAlignmentTask.h"
#include "Clipboard/SubalignmentToClipboardTask.h"
#include "helpers/ScrollController.h"
#include "Highlighting/MSAHighlightingTabFactory.h"
#include "view_rendering/SequenceAreaRenderer.h"

namespace U2 {

MSAEditorSequenceArea::MSAEditorSequenceArea(MaEditorWgt* _ui, GScrollBar* hb, GScrollBar* vb)
    : MaEditorSequenceArea(_ui, hb, vb)
{
    setObjectName("msa_editor_sequence_area");
    setFocusPolicy(Qt::WheelFocus);

    initRenderer();

    connect(editor, SIGNAL(si_buildPopupMenu(GObjectView *, QMenu *)), SLOT(sl_buildContextMenu(GObjectView *, QMenu *)));
    connect(editor, SIGNAL(si_buildStaticMenu(GObjectView *, QMenu *)), SLOT(sl_buildStaticMenu(GObjectView *, QMenu *)));
    connect(editor, SIGNAL(si_buildStaticToolbar(GObjectView *, QToolBar *)), SLOT(sl_buildStaticToolbar(GObjectView *, QToolBar *)));

    selectionColor = Qt::black;
    editingEnabled = true;

    connect(ui->getCopySelectionAction(), SIGNAL(triggered()), SLOT(sl_copyCurrentSelection()));
    addAction(ui->getCopySelectionAction());

    connect(ui->getCopyFormattedSelectionAction(), SIGNAL(triggered()), SLOT(sl_copyFormattedSelection()));
    addAction(ui->getCopyFormattedSelectionAction());

    connect(ui->getPasteAction(), SIGNAL(triggered()), SLOT(sl_paste()));
    addAction(ui->getPasteAction());

    delColAction = new QAction(QIcon(":core/images/msaed_remove_columns_with_gaps.png"), tr("Remove columns of gaps..."), this);
    delColAction->setObjectName("remove_columns_of_gaps");
    delColAction->setShortcut(QKeySequence(Qt::SHIFT| Qt::Key_Delete));
    delColAction->setShortcutContext(Qt::WidgetShortcut);
    addAction(delColAction);
    connect(delColAction, SIGNAL(triggered()), SLOT(sl_delCol()));

    createSubaligniment = new QAction(tr("Save subalignment..."), this);
    createSubaligniment->setObjectName("Save subalignment");
    connect(createSubaligniment, SIGNAL(triggered()), SLOT(sl_createSubaligniment()));

    saveSequence = new QAction(tr("Export selected sequence(s)..."), this);
    saveSequence->setObjectName("Save sequence");
    connect(saveSequence, SIGNAL(triggered()), SLOT(sl_saveSequence()));

    gotoAction = new QAction(QIcon(":core/images/goto.png"), tr("Go to position..."), this);
    gotoAction->setObjectName("action_go_to_position");
    gotoAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_G));
    gotoAction->setShortcutContext(Qt::WindowShortcut);
    gotoAction->setToolTip(QString("%1 (%2)").arg(gotoAction->text()).arg(gotoAction->shortcut().toString()));
    connect(gotoAction, SIGNAL(triggered()), SLOT(sl_goto()));

    removeAllGapsAction = new QAction(QIcon(":core/images/msaed_remove_all_gaps.png"), tr("Remove all gaps"), this);
    removeAllGapsAction->setObjectName("Remove all gaps");
    connect(removeAllGapsAction, SIGNAL(triggered()), SLOT(sl_removeAllGaps()));

    addSeqFromFileAction = new QAction(tr("Sequence from file..."), this);
    addSeqFromFileAction->setObjectName("Sequence from file");
    connect(addSeqFromFileAction, SIGNAL(triggered()), SLOT(sl_addSeqFromFile()));

    addSeqFromProjectAction = new QAction(tr("Sequence from current project..."), this);
    addSeqFromProjectAction->setObjectName("Sequence from current project");
    connect(addSeqFromProjectAction, SIGNAL(triggered()), SLOT(sl_addSeqFromProject()));

    sortByNameAction = new QAction(tr("Sort sequences by name"), this);
    sortByNameAction->setObjectName("action_sort_by_name");
    connect(sortByNameAction, SIGNAL(triggered()), SLOT(sl_sortByName()));

    collapseModeSwitchAction = new QAction(QIcon(":core/images/collapse.png"), tr("Switch on/off collapsing"), this);
    collapseModeSwitchAction->setObjectName("Enable collapsing");
    collapseModeSwitchAction->setCheckable(true);
    connect(collapseModeSwitchAction, SIGNAL(toggled(bool)), SLOT(sl_setCollapsingMode(bool)));

    collapseModeUpdateAction = new QAction(QIcon(":core/images/collapse_update.png"), tr("Update collapsed groups"), this);
    collapseModeUpdateAction->setObjectName("Update collapsed groups");
    collapseModeUpdateAction->setEnabled(false);
    connect(collapseModeUpdateAction, SIGNAL(triggered()), SLOT(sl_updateCollapsingMode()));

    reverseComplementAction = new QAction(tr("Replace selected rows with reverse-complement"), this);
    reverseComplementAction->setObjectName("replace_selected_rows_with_reverse-complement");
    connect(reverseComplementAction, SIGNAL(triggered()), SLOT(sl_reverseComplementCurrentSelection()));

    reverseAction = new QAction(tr("Replace selected rows with reverse"), this);
    reverseAction->setObjectName("replace_selected_rows_with_reverse");
    connect(reverseAction, SIGNAL(triggered()), SLOT(sl_reverseCurrentSelection()));

    complementAction = new QAction(tr("Replace selected rows with complement"), this);
    complementAction->setObjectName("replace_selected_rows_with_complement");
    connect(complementAction, SIGNAL(triggered()), SLOT(sl_complementCurrentSelection()));

    connect(editor->getMaObject(), SIGNAL(si_alignmentChanged(const MultipleAlignment&, const MaModificationInfo&)),
        SLOT(sl_alignmentChanged(const MultipleAlignment&, const MaModificationInfo&)));
    connect(editor->getMaObject(), SIGNAL(si_lockedStateChanged()), SLOT(sl_lockedStateChanged()));
    connect(editor->getMaObject(), SIGNAL(si_rowsRemoved(const QList<qint64> &)), SLOT(sl_updateCollapsingMode()));

    connect(this,   SIGNAL(si_startMaChanging()),
            ui,     SIGNAL(si_startMaChanging()));
    connect(this,   SIGNAL(si_stopMaChanging(bool)),
            ui,     SIGNAL(si_stopMaChanging(bool)));

    connect(ui->getCollapseModel(), SIGNAL(si_toggled()), SLOT(sl_modelChanged()));
    connect(editor, SIGNAL(si_fontChanged(QFont)), SLOT(sl_fontChanged(QFont)));
    connect(editor, SIGNAL(si_referenceSeqChanged(qint64)), SLOT(sl_completeUpdate()));

    connect(ui->getUndoAction(), SIGNAL(triggered()), SLOT(sl_resetCollapsibleModel()));
    connect(ui->getRedoAction(), SIGNAL(triggered()), SLOT(sl_resetCollapsibleModel()));

    connect(editor->getMaObject(), SIGNAL(si_alphabetChanged(const MaModificationInfo &, const DNAAlphabet*)),
        SLOT(sl_alphabetChanged(const MaModificationInfo &, const DNAAlphabet*)));

    setMouseTracking(true);

    updateColorAndHighlightSchemes();
    sl_updateActions();
}

MSAEditor *MSAEditorSequenceArea::getEditor() const {
    return qobject_cast<MSAEditor*>(editor);
}

QStringList MSAEditorSequenceArea::getAvailableHighlightingSchemes() const{
    QStringList allSchemas;
    foreach(QAction *a, highlightingSchemeMenuActions){
        allSchemas.append(a->text());
    }
    return allSchemas;
}

bool MSAEditorSequenceArea::hasAminoAlphabet() {
    MultipleAlignmentObject* maObj = editor->getMaObject();
    SAFE_POINT(NULL != maObj, tr("MultipleAlignmentObject is null in MSAEditorSequenceArea::hasAminoAlphabet()"), false);
    const DNAAlphabet* alphabet = maObj->getAlphabet();
    SAFE_POINT(NULL != maObj, tr("DNAAlphabet is null in MSAEditorSequenceArea::hasAminoAlphabet()"), false);
    return DNAAlphabet_AMINO == alphabet->getType();
}

void MSAEditorSequenceArea::focusInEvent(QFocusEvent* fe) {
    QWidget::focusInEvent(fe);
    update();
}

void MSAEditorSequenceArea::focusOutEvent(QFocusEvent* fe) {
    QWidget::focusOutEvent(fe);
    exitFromEditCharacterMode();
    update();
}

void MSAEditorSequenceArea::moveCursor(int dx, int dy) {
    const QPoint newCursorPos = cursorPos + QPoint(dx, dy);
    CHECK(isInRange(newCursorPos), );

    // Move only one cell selection?
    // TODO: consider selection movement
    const int selectionSize = selection.width() * selection.height();
    CHECK(selectionSize == 1, );

    ui->getScrollController()->scrollToPoint(newCursorPos, size());
    setCursorPos(newCursorPos);

    // SANGER_TODO: why is it under comment
    //setSelection(MSAEditorSelection(p, 1,1));
}

void MSAEditorSequenceArea::updateCollapsedGroups(const MaModificationInfo& modInfo) {
    U2OpStatus2Log os;
    if(modInfo.rowContentChanged) {
        QList<qint64> updatedRows;
        bool isModelChanged = false;
        QMap<qint64, QList<U2MsaGap> > curGapModel = getEditor()->getMaObject()->getMapGapModel();
        QList<U2Region> updatedRegions;
        foreach (qint64 modifiedSeqId, modInfo.modifiedRowIds) {
            int modifiedRowPos = editor->getMaObject()->getRowPosById(modifiedSeqId);
            const MultipleSequenceAlignmentRow &modifiedRowRef = editor->getMaObject()->getRow(modifiedRowPos);
            modifiedRowPos = ui->getCollapseModel()->rowToMap(modifiedRowPos);
            const U2Region rowsCollapsibleGroup = ui->getCollapseModel()->mapSelectionRegionToRows(U2Region(modifiedRowPos, 1));
            if (updatedRegions.contains(rowsCollapsibleGroup)) {
                continue;
            }
            for(int i = rowsCollapsibleGroup.startPos; i < rowsCollapsibleGroup.endPos(); i++) {
                qint64 identicalRowId = editor->getMaObject()->getRow(i)->getRowId();
                if(!updatedRows.contains(identicalRowId) && !modInfo.modifiedRowIds.contains(identicalRowId)) {
                    isModelChanged = isModelChanged || modifiedRowRef->getGapModel() != curGapModel[identicalRowId];
                    curGapModel[identicalRowId] = modifiedRowRef->getGapModel();
                    updatedRows.append(identicalRowId);
                }
            }
            updatedRegions.append(rowsCollapsibleGroup);
        }
        if(isModelChanged) {
            getEditor()->getMaObject()->updateGapModel(os, curGapModel);
            return;
        }
    }
}

void MSAEditorSequenceArea::sl_buildStaticToolbar(GObjectView* v, QToolBar* t) {
    t->addAction(ui->getUndoAction());
    t->addAction(ui->getRedoAction());
    t->addAction(gotoAction);
    t->addAction(removeAllGapsAction);
    t->addSeparator();

    t->addAction(collapseModeSwitchAction);
    t->addAction(collapseModeUpdateAction);
    t->addSeparator();
}

void MSAEditorSequenceArea::sl_buildStaticMenu(GObjectView*, QMenu* m) {
    buildMenu(m);
}

void MSAEditorSequenceArea::sl_buildContextMenu(GObjectView*, QMenu* m) {
    buildMenu(m);

    QMenu* editMenu = GUIUtils::findSubMenu(m, MSAE_MENU_EDIT);
    SAFE_POINT(editMenu != NULL, "editMenu", );

    QList<QAction*> actions;
    actions << fillWithGapsinsSymAction << replaceCharacterAction << reverseComplementAction
            << reverseAction << complementAction << delColAction << removeAllGapsAction;

    QMenu* copyMenu = GUIUtils::findSubMenu(m, MSAE_MENU_COPY);
    SAFE_POINT(copyMenu != NULL, "copyMenu", );
    editMenu->insertAction(editMenu->actions().first(), ui->getDelSelectionAction());
    if (rect().contains(mapFromGlobal(QCursor::pos()))) {
        editMenu->addActions(actions);
        copyMenu->addAction(ui->getCopySelectionAction());
        copyMenu->addAction(ui->getCopyFormattedSelectionAction());
    }

    m->setObjectName("msa sequence area context menu");
}

void MSAEditorSequenceArea::sl_showCustomSettings(){
    AppContext::getAppSettingsGUI()->showSettingsDialog(ColorSchemaSettingsPageId);
}

void MSAEditorSequenceArea::initRenderer() {
    renderer = new SequenceAreaRenderer(ui, this);
}

void MSAEditorSequenceArea::buildMenu(QMenu* m) {
    QAction* copyMenuAction = GUIUtils::findAction(m->actions(), MSAE_MENU_LOAD);
    m->insertAction(copyMenuAction, gotoAction);

    QMenu* loadSeqMenu = GUIUtils::findSubMenu(m, MSAE_MENU_LOAD);
    SAFE_POINT(loadSeqMenu != NULL, "loadSeqMenu", );
    loadSeqMenu->addAction(addSeqFromProjectAction);
    loadSeqMenu->addAction(addSeqFromFileAction);

    QMenu* editMenu = GUIUtils::findSubMenu(m, MSAE_MENU_EDIT);
    SAFE_POINT(editMenu != NULL, "editMenu", );
    QList<QAction*> actions;

    MsaEditorWgt* msaWgt = getEditor()->getUI();
    QAction* editSequenceNameAction = msaWgt->getEditorNameList()->getEditSequenceNameAction();
    if (getSelection().height() != 1) {
        editSequenceNameAction->setDisabled(true);
    }
    actions << editSequenceNameAction << fillWithGapsinsSymAction << replaceCharacterAction << reverseComplementAction << reverseAction << complementAction << delColAction << removeAllGapsAction;
    editMenu->insertActions(editMenu->isEmpty() ? NULL : editMenu->actions().first(), actions);
    editMenu->insertAction(editMenu->actions().first(), ui->getDelSelectionAction());

    QMenu * exportMenu = GUIUtils::findSubMenu(m, MSAE_MENU_EXPORT);
    SAFE_POINT(exportMenu != NULL, "exportMenu", );
    exportMenu->addAction(createSubaligniment);
    exportMenu->addAction(saveSequence);

    QMenu* copyMenu = GUIUtils::findSubMenu(m, MSAE_MENU_COPY);
    SAFE_POINT(copyMenu != NULL, "copyMenu", );
    ui->getCopySelectionAction()->setDisabled(selection.isNull());
    emit si_copyFormattedChanging(!selection.isNull());
    copyMenu->addAction(ui->getCopySelectionAction());
    ui->getCopyFormattedSelectionAction()->setDisabled(selection.isNull());
    copyMenu->addAction(ui->getCopyFormattedSelectionAction());
    copyMenu->addAction(ui->getPasteAction());

    QMenu* viewMenu = GUIUtils::findSubMenu(m, MSAE_MENU_VIEW);
    SAFE_POINT(viewMenu != NULL, "viewMenu", );
    viewMenu->addAction(sortByNameAction);

    QMenu* colorsSchemeMenu = new QMenu(tr("Colors"), NULL);
    colorsSchemeMenu->menuAction()->setObjectName("Colors");
    colorsSchemeMenu->setIcon(QIcon(":core/images/color_wheel.png"));
    foreach(QAction* a, colorSchemeMenuActions) {
        MsaSchemesMenuBuilder::addActionOrTextSeparatorToMenu(a, colorsSchemeMenu);
    }
    colorsSchemeMenu->addSeparator();

    QMenu* customColorSchemaMenu = new QMenu(tr("Custom schemes"), colorsSchemeMenu);
    customColorSchemaMenu->menuAction()->setObjectName("Custom schemes");

    foreach(QAction* a, customColorSchemeMenuActions) {
        MsaSchemesMenuBuilder::addActionOrTextSeparatorToMenu(a, customColorSchemaMenu);
    }

    if (!customColorSchemeMenuActions.isEmpty()){
        customColorSchemaMenu->addSeparator();
    }

    lookMSASchemesSettingsAction = new QAction(tr("Create new color scheme"), this);
    lookMSASchemesSettingsAction->setObjectName("Create new color scheme");
    connect(lookMSASchemesSettingsAction, SIGNAL(triggered()), SLOT(sl_showCustomSettings()));
    customColorSchemaMenu->addAction(lookMSASchemesSettingsAction);

    colorsSchemeMenu->addMenu(customColorSchemaMenu);
    m->insertMenu(GUIUtils::findAction(m->actions(), MSAE_MENU_EDIT), colorsSchemeMenu);

    QMenu* highlightSchemeMenu = new QMenu(tr("Highlighting"), NULL);

    highlightSchemeMenu->menuAction()->setObjectName("Highlighting");

    foreach(QAction* a, highlightingSchemeMenuActions) {
        MsaSchemesMenuBuilder::addActionOrTextSeparatorToMenu(a, highlightSchemeMenu);
    }
    highlightSchemeMenu->addSeparator();
    highlightSchemeMenu->addAction(useDotsAction);
    m->insertMenu(GUIUtils::findAction(m->actions(), MSAE_MENU_EDIT), highlightSchemeMenu);
}

void MSAEditorSequenceArea::sl_fontChanged(QFont font) {
    Q_UNUSED(font);
    completeRedraw = true;
    repaint();
}

void MSAEditorSequenceArea::sl_alphabetChanged(const MaModificationInfo &mi, const DNAAlphabet *prevAlphabet) {
    updateColorAndHighlightSchemes();

    QString message;
    if (mi.alphabetChanged || mi.type != MaModificationType_Undo) {
        message = tr("The alignment has been modified, so that its alphabet has been switched from \"%1\" to \"%2\". Use \"Undo\", if you'd like to restore the original alignment.")
            .arg(prevAlphabet->getName()).arg(editor->getMaObject()->getAlphabet()->getName());
    }

    if (message.isEmpty()) {
        return;
    }
    const NotificationStack *notificationStack = AppContext::getMainWindow()->getNotificationStack();
    CHECK(notificationStack != NULL, );
    notificationStack->addNotification(message, Info_Not);
}

void MSAEditorSequenceArea::sl_updateActions() {
    MultipleAlignmentObject* maObj = editor->getMaObject();
    assert(maObj != NULL);
    bool readOnly = maObj->isStateLocked();

    createSubaligniment->setEnabled(!isAlignmentEmpty());
    saveSequence->setEnabled(!isAlignmentEmpty());
    addSeqFromProjectAction->setEnabled(!readOnly);
    addSeqFromFileAction->setEnabled(!readOnly);
    sortByNameAction->setEnabled(!readOnly && !isAlignmentEmpty());
    collapseModeSwitchAction->setEnabled(!readOnly && !isAlignmentEmpty());

//Update actions of "Edit" group
    bool canEditAlignment = !readOnly && !isAlignmentEmpty();
    bool canEditSelectedArea = canEditAlignment && !selection.isNull();
    const bool isEditing = (maMode != ViewMode);
    ui->getDelSelectionAction()->setEnabled(canEditSelectedArea);
    ui->getPasteAction()->setEnabled(!readOnly);

    fillWithGapsinsSymAction->setEnabled(canEditSelectedArea && !isEditing);
    bool oneCharacterIsSelected = selection.width() == 1 && selection.height() == 1;
    replaceCharacterAction->setEnabled(canEditSelectedArea && oneCharacterIsSelected);
    delColAction->setEnabled(canEditAlignment);
    reverseComplementAction->setEnabled(canEditSelectedArea && maObj->getAlphabet()->isNucleic());
    reverseAction->setEnabled(canEditSelectedArea);
    complementAction->setEnabled(canEditSelectedArea && maObj->getAlphabet()->isNucleic());
    removeAllGapsAction->setEnabled(canEditAlignment);
}

void MSAEditorSequenceArea::sl_delCol() {
    QObjectScopedPointer<DeleteGapsDialog> dlg = new DeleteGapsDialog(this, editor->getMaObject()->getNumRows());
    dlg->exec();
    CHECK(!dlg.isNull(), );

    if (dlg->result() == QDialog::Accepted) {
        MSACollapsibleItemModel *collapsibleModel = ui->getCollapseModel();
        SAFE_POINT(NULL != collapsibleModel, tr("NULL collapsible model!"), );
        collapsibleModel->reset();

        DeleteMode deleteMode = dlg->getDeleteMode();
        int value = dlg->getValue();

        // if this method was invoked during a region shifting
        // then shifting should be canceled
        cancelShiftTracking();

        MultipleSequenceAlignmentObject* msaObj = getEditor()->getMaObject();
        int gapCount = 0;
        switch(deleteMode) {
        case DeleteByAbsoluteVal:
            gapCount = value;
            break;
        case DeleteByRelativeVal: {
            int absoluteValue = qRound((msaObj->getNumRows() * value) / 100.0);
            if (absoluteValue < 1) {
                absoluteValue = 1;
            }
            gapCount = absoluteValue;
            break;
        }
        case DeleteAll:
            gapCount = msaObj->getNumRows();
            break;
        default:
            FAIL("Unknown delete mode", );
        }

        U2OpStatus2Log os;
        U2UseCommonUserModStep userModStep(msaObj->getEntityRef(), os);
        Q_UNUSED(userModStep);
        SAFE_POINT_OP(os, );
        msaObj->deleteColumnsWithGaps(os, gapCount);
    }
}

void MSAEditorSequenceArea::sl_goto() {
    QObjectScopedPointer<QDialog> dlg = new QDialog(this);
    dlg->setModal(true);
    dlg->setWindowTitle(tr("Go To"));
    int aliLen = editor->getAlignmentLen();
    PositionSelector *ps = new PositionSelector(dlg.data(), 1, aliLen, true);
    connect(ps, SIGNAL(si_positionChanged(int)), SLOT(sl_onPosChangeRequest(int)));
    dlg->exec();
}

void MSAEditorSequenceArea::sl_onPosChangeRequest(int position) {
    ui->getScrollController()->centerBase(position, width());
    setSelection(MaEditorSelection(position-1, selection.y(), 1, 1));
}

void MSAEditorSequenceArea::sl_lockedStateChanged() {
    sl_updateActions();
}

void MSAEditorSequenceArea::sl_removeAllGaps() {
    MultipleSequenceAlignmentObject* msa = getEditor()->getMaObject();
    SAFE_POINT(NULL != msa, tr("NULL msa object!"), );
    assert(!msa->isStateLocked());

    MSACollapsibleItemModel *collapsibleModel = ui->getCollapseModel();
    SAFE_POINT(NULL != collapsibleModel, tr("NULL collapsible model!"), );
    collapsibleModel->reset();

    // if this method was invoked during a region shifting
    // then shifting should be canceled
    cancelShiftTracking();

    U2OpStatus2Log os;
    U2UseCommonUserModStep userModStep(msa->getEntityRef(), os);
    Q_UNUSED(userModStep);
    SAFE_POINT_OP(os, );

    QMap<qint64, QList<U2MsaGap> > noGapModel;
    foreach (qint64 rowId, msa->getMultipleAlignment()->getRowsIds()) {
        noGapModel[rowId] = QList<U2MsaGap>();
    }

    msa->updateGapModel(os, noGapModel);

    MsaDbiUtils::trim(msa->getEntityRef(), os);
    msa->updateCachedMultipleAlignment();

    SAFE_POINT_OP(os, );

    ui->getScrollController()->setFirstVisibleBase(0);
    ui->getScrollController()->setFirstVisibleRowByNumber(0);
    SAFE_POINT_OP(os, );
}

void MSAEditorSequenceArea::sl_createSubaligniment(){
    CHECK(getEditor() != NULL, );
    QObjectScopedPointer<CreateSubalignmentDialogController> dialog = new CreateSubalignmentDialogController(getEditor()->getMaObject(), selection.getRect(), this);
    dialog->exec();
    CHECK(!dialog.isNull(), );

    if (dialog->result() == QDialog::Accepted){
        U2Region window = dialog->getRegion();
        bool addToProject = dialog->getAddToProjFlag();
        QString path = dialog->getSavePath();
        QStringList seqNames = dialog->getSelectedSeqNames();
        Task* csTask = new CreateSubalignmentAndOpenViewTask(getEditor()->getMaObject(),
            CreateSubalignmentSettings(window, seqNames, path, true, addToProject, dialog->getFormatId()));
        AppContext::getTaskScheduler()->registerTopLevelTask(csTask);
    }
}

void MSAEditorSequenceArea::sl_saveSequence(){
    CHECK(getEditor() != NULL, );
    QStringList seqNames;
    MultipleAlignment ma = editor->getMaObject()->getMultipleAlignment();
    QRect selection = editor->getCurrentSelection();
    int startSeq = selection.y();
    int endSeq = selection.y() + selection.height() - 1;
    MSACollapsibleItemModel *model = editor->getUI()->getCollapseModel();
    for (int i = startSeq; i <= endSeq; i++) {
        seqNames.append(ma->getRow(model->mapToRow(i))->getName());
    }

    QObjectScopedPointer<SaveSelectedSequenceFromMSADialogController> d = new SaveSelectedSequenceFromMSADialogController(editor->getMaObject()->getDocument()->getURL().dirPath(),
        (QWidget*)AppContext::getMainWindow()->getQMainWindow(), seqNames, editor->getMaObject()->getGObjectName() + "_sequence");
    const int rc = d->exec();
    CHECK(!d.isNull(), );

    if (rc == QDialog::Rejected) {
        return;
    }
    DocumentFormat *df = AppContext::getDocumentFormatRegistry()->getFormatById(d->getFormat());
    SAFE_POINT(df != NULL, "Unknown document format", );
    QString extension = df->getSupportedDocumentFileExtensions().first();

    AppContext::getTaskScheduler()->registerTopLevelTask(new ExportSequencesTask(getEditor()->getMaObject()->getMsa(), seqNames, d->getTrimGapsFlag(), d->getAddToProjectFlag(), d->getUrl(), d->getFormat(), extension, d->getCustomFileName()));
}

void MSAEditorSequenceArea::sl_modelChanged() {
    MSACollapsibleItemModel *collapsibleModel = ui->getCollapseModel();
    SAFE_POINT(NULL != collapsibleModel, "NULL collapsible model", );

    if (collapsibleModel->isEmpty()) {
        collapseModeSwitchAction->setChecked(false);
        collapseModeUpdateAction->setEnabled(false);
    }

    MaEditorSequenceArea::sl_modelChanged();
}

void MSAEditorSequenceArea::sl_copyCurrentSelection()
{
    CHECK(getEditor() != NULL, );
    // TODO: probably better solution would be to export selection???

    assert(isInRange(selection.topLeft()));
    assert(isInRange(QPoint(selection.x() + selection.width() - 1, selection.y() + selection.height() - 1)));

    MultipleSequenceAlignmentObject* maObj = getEditor()->getMaObject();
    if (selection.isNull()) {
        return;
    }

    MSACollapsibleItemModel* m = ui->getCollapseModel();
    U2Region sel(m->mapToRow(selection.y()), m->mapToRow(selection.y() + selection.height()) - m->mapToRow(selection.y()));

    QString selText;
    U2OpStatus2Log os;
    for (int i = sel.startPos; i < sel.endPos(); ++i) {
        if (ui->getCollapseModel()->rowToMap(i, true) < 0) {
            continue;
        }
        int len = selection.width();
        QByteArray seqPart = maObj->getMsaRow(i)->mid(selection.x(), len, os)->toByteArray(os, len);
        selText.append(seqPart);
        if (i + 1 != sel.endPos()) { // do not add line break into the last line
            selText.append("\n");
        }
    }
    QApplication::clipboard()->setText(selText);
}

void MSAEditorSequenceArea::sl_copyFormattedSelection(){
    const DocumentFormatId& formatId = getCopyFormatedAlgorithmId();
    Task* clipboardTask = new SubalignmentToClipboardTask(getEditor(), selection.getRect(), formatId);
    AppContext::getTaskScheduler()->registerTopLevelTask(clipboardTask);
}

void MSAEditorSequenceArea::sl_paste(){
    MultipleAlignmentObject* msaObject = editor->getMaObject();
    if (msaObject->isStateLocked()) {
        return;
    }
    PasteFactory* pasteFactory = AppContext::getPasteFactory();
    SAFE_POINT(pasteFactory != NULL, "PasteFactory is null", );

    bool focus = ui->isAncestorOf(QApplication::focusWidget());
    PasteTask* task = pasteFactory->pasteTask(!focus);
    if (focus) {
        connect(new TaskSignalMapper(task), SIGNAL(si_taskFinished(Task *)), SLOT(sl_pasteFinished(Task*)));
    }
    AppContext::getTaskScheduler()->registerTopLevelTask(task);
}

void MSAEditorSequenceArea::sl_pasteFinished(Task* _pasteTask){
    CHECK(getEditor() != NULL, );
    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    if (msaObject->isStateLocked()) {
        return;
    }

    PasteTask* pasteTask = qobject_cast<PasteTask*>(_pasteTask);
    if(NULL == pasteTask || pasteTask->isCanceled()) {
        return;
    }
    const QList<Document*>& docs = pasteTask->getDocuments();

    AddSequencesFromDocumentsToAlignmentTask *task = new AddSequencesFromDocumentsToAlignmentTask(msaObject, docs);
    task->setErrorNotificationSuppression(true); // we manually show warning message if needed when task is finished.
    connect(new TaskSignalMapper(task), SIGNAL(si_taskFinished(Task *)), SLOT(sl_addSequencesToAlignmentFinished(Task*)));
    AppContext::getTaskScheduler()->registerTopLevelTask(task);
}

void MSAEditorSequenceArea::sl_addSequencesToAlignmentFinished(Task *task) {
    AddSequencesFromDocumentsToAlignmentTask *addSeqTask = qobject_cast<AddSequencesFromDocumentsToAlignmentTask*>(task);
    CHECK(addSeqTask != NULL, );
    const MaModificationInfo& mi = addSeqTask->getMaModificationInfo();
    if (!mi.rowListChanged) {
        const NotificationStack *notificationStack = AppContext::getMainWindow()->getNotificationStack();
        CHECK(notificationStack != NULL,);
        notificationStack->addNotification(tr("No new rows were inserted: selection contains no valid sequences."), Warning_Not);
    }
}

void MSAEditorSequenceArea::sl_addSeqFromFile()
{
    CHECK(getEditor() != NULL, );
    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    if (msaObject->isStateLocked()) {
        return;
    }

    QString filter = DialogUtils::prepareDocumentsFileFilterByObjType(GObjectTypes::SEQUENCE, true);

    LastUsedDirHelper lod;
    QStringList urls;
#ifdef Q_OS_MAC
    if (qgetenv(ENV_GUI_TEST).toInt() == 1 && qgetenv(ENV_USE_NATIVE_DIALOGS).toInt() == 0) {
        urls = U2FileDialog::getOpenFileNames(this, tr("Open file with sequences"), lod.dir, filter, 0, QFileDialog::DontUseNativeDialog);
    } else
#endif
    urls = U2FileDialog::getOpenFileNames(this, tr("Open file with sequences"), lod.dir, filter);

    if (!urls.isEmpty()) {
        lod.url = urls.first();
        sl_cancelSelection();
        AddSequencesFromFilesToAlignmentTask *task = new AddSequencesFromFilesToAlignmentTask(msaObject, urls);
        TaskWatchdog::trackResourceExistence(msaObject, task, tr("A problem occurred during adding sequences. The multiple alignment is no more available."));
        AppContext::getTaskScheduler()->registerTopLevelTask(task);
    }

}

void MSAEditorSequenceArea::sl_addSeqFromProject()
{
    CHECK(getEditor() != NULL, );
    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    if (msaObject->isStateLocked()) {
        return;
    }

    ProjectTreeControllerModeSettings settings;
    settings.objectTypesToShow.insert(GObjectTypes::SEQUENCE);

    QList<GObject*> objects = ProjectTreeItemSelectorDialog::selectObjects(settings,this);
    QList<DNASequence> objectsToAdd;
    U2OpStatus2Log os;
    foreach(GObject* obj, objects) {
        U2SequenceObject *seqObj = qobject_cast<U2SequenceObject*>(obj);
        if (seqObj) {
            objectsToAdd.append(seqObj->getWholeSequence(os));
            SAFE_POINT_OP(os, );
        }
    }
    if (objectsToAdd.size() > 0) {
        AddSequenceObjectsToAlignmentTask *addSeqObjTask = new AddSequenceObjectsToAlignmentTask(getEditor()->getMaObject(), objectsToAdd);
        AppContext::getTaskScheduler()->registerTopLevelTask(addSeqObjTask);
        sl_cancelSelection();
    }
}

void MSAEditorSequenceArea::sl_sortByName() {
    CHECK(getEditor() != NULL, );
    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    if (msaObject->isStateLocked()) {
        return;
    }
    MultipleSequenceAlignment msa = msaObject->getMultipleAlignmentCopy();
    msa->sortRowsByName();
    QStringList rowNames = msa->getRowNames();
    if (rowNames != msaObject->getMultipleAlignment()->getRowNames()) {
        U2OpStatusImpl os;
        msaObject->updateRowsOrder(os, msa->getRowsIds());
        SAFE_POINT_OP(os, );
    }
    if (ui->isCollapsibleMode()) {
        sl_updateCollapsingMode();
    }
}

void MSAEditorSequenceArea::sl_setCollapsingMode(bool enabled) {
    CHECK(getEditor() != NULL, );
    GCOUNTER(cvar, tvar, "Switch collapsing mode");

    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    if (msaObject == NULL || msaObject->isStateLocked()) {
        if (collapseModeSwitchAction->isChecked()) {
            collapseModeSwitchAction->setChecked(false);
            collapseModeUpdateAction->setEnabled(false);
        }
        return;
    }

    ui->setCollapsibleMode(enabled);
    collapseModeUpdateAction->setEnabled(enabled);
    if (enabled) {
        sl_updateCollapsingMode();
    } else {
        MSACollapsibleItemModel *collapsibleModel = ui->getCollapseModel();
        SAFE_POINT(NULL != collapsibleModel, tr("NULL collapsible model!"), );
        collapsibleModel->reset();
    }

    updateSelection();
    ui->getScrollController()->updateVerticalScrollBar();
}

void MSAEditorSequenceArea::sl_updateCollapsingMode() {
    CHECK(getEditor() != NULL, );
    GCOUNTER(cvar, tvar, "Update collapsing mode");

    CHECK(ui->isCollapsibleMode(), );
    MultipleSequenceAlignmentObject *msaObject = getEditor()->getMaObject();
    SAFE_POINT(NULL != msaObject, tr("NULL Msa Object!"), );

    MSACollapsibleItemModel *collapsibleModel = ui->getCollapseModel();

    Document *doc = msaObject->getDocument();
    SAFE_POINT(NULL != doc, tr("NULL document!"), );

    MultipleSequenceAlignment msa = msaObject->getMultipleAlignmentCopy();
    QVector<U2Region> unitedRows;
    bool sorted = msa->sortRowsBySimilarity(unitedRows);
    collapsibleModel->reset(unitedRows);

    U2OpStatusImpl os;
    if (sorted) {
        msaObject->updateRowsOrder(os, msa->getRowsIds());
        SAFE_POINT_OP(os, );
    }

    MaModificationInfo mi;
    mi.alignmentLengthChanged = false;
    msaObject->updateCachedMultipleAlignment(mi);
}

void MSAEditorSequenceArea::reverseComplementModification(ModificationType& type) {
    CHECK(getEditor() != NULL, );
    if (type == ModificationType::NoType)
        return;
    MultipleSequenceAlignmentObject* maObj = getEditor()->getMaObject();
    if (maObj == NULL || maObj->isStateLocked()) {
        return;
    }
    if (!maObj->getAlphabet()->isNucleic()) {
        return;
    }
    if (selection.height() == 0) {
        return;
    }
    assert(isInRange(selection.topLeft()));
    assert(isInRange(QPoint(selection.x() + selection.width() - 1, selection.y() + selection.height() - 1)));
    if (!selection.isNull()) {
        // if this method was invoked during a region shifting
        // then shifting should be canceled
        cancelShiftTracking();

        const MultipleSequenceAlignment ma = maObj->getMultipleAlignment();
        DNATranslation* trans = AppContext::getDNATranslationRegistry()->lookupComplementTranslation(ma->getAlphabet());
        if (trans == NULL || !trans->isOne2One()) {
            return;
        }

        U2OpStatus2Log os;
        U2UseCommonUserModStep userModStep(maObj->getEntityRef(), os);
        Q_UNUSED(userModStep);
        SAFE_POINT_OP(os, );

        const U2Region& sel = getSelectedRows();

        QList<qint64> modifiedRowIds;
        modifiedRowIds.reserve(sel.length);
        for (int i = sel.startPos; i < sel.endPos(); i++) {
            const MultipleSequenceAlignmentRow currentRow = ma->getMsaRow(i);
            QByteArray currentRowContent = currentRow->toByteArray(os, ma->getLength());
            switch (type.getType())
            {
            case ModificationType::Reverse:
                TextUtils::reverse(currentRowContent.data(), currentRowContent.length());
                break;
            case ModificationType::Complement:
                trans->translate(currentRowContent.data(), currentRowContent.length());
                break;
            case ModificationType::ReverseComplement:
                TextUtils::reverse(currentRowContent.data(), currentRowContent.length());
                trans->translate(currentRowContent.data(), currentRowContent.length());
                break;
            }
            QString name = currentRow->getName();
            ModificationType oldType(ModificationType::NoType);
            if (name.endsWith("|revcompl")) {
                name.resize(name.length() - QString("|revcompl").length());
                oldType = ModificationType::ReverseComplement;
            } else if (name.endsWith("|compl")) {
                name.resize(name.length() - QString("|compl").length());
                oldType = ModificationType::Complement;
            } else if (name.endsWith("|rev")) {
                name.resize(name.length() - QString("|rev").length());
                oldType = ModificationType::Reverse;
            }
            ModificationType newType = type + oldType;
            switch (newType.getType())
            {
            case ModificationType::NoType:
                break;
            case ModificationType::Reverse:
                name.append("|rev");
                break;
            case ModificationType::Complement:
                name.append("|compl");
                break;
            case ModificationType::ReverseComplement:
                name.append("|revcompl");
                break;
            }

            // Split the sequence into gaps and chars
            QByteArray seqBytes;
            QList<U2MsaGap> gapModel;
            MaDbiUtils::splitBytesToCharsAndGaps(currentRowContent, seqBytes, gapModel);

            maObj->updateRow(os, i, name, seqBytes, gapModel);
            modifiedRowIds << currentRow->getRowId();
        }

        MaModificationInfo modInfo;
        modInfo.modifiedRowIds = modifiedRowIds;
        modInfo.alignmentLengthChanged = false;
        maObj->updateCachedMultipleAlignment(modInfo);
    }
}

void MSAEditorSequenceArea::sl_reverseComplementCurrentSelection() {
    ModificationType type(ModificationType::ReverseComplement);
    reverseComplementModification(type);
}

void MSAEditorSequenceArea::sl_reverseCurrentSelection() {
    ModificationType type(ModificationType::Reverse);
    reverseComplementModification(type);
}

void MSAEditorSequenceArea::sl_complementCurrentSelection() {
    ModificationType type(ModificationType::Complement);
    reverseComplementModification(type);
}

void MSAEditorSequenceArea::sl_resetCollapsibleModel() {
    editor->resetCollapsibleModel();
}

void MSAEditorSequenceArea::sl_setCollapsingRegions(const QList<QStringList>& collapsedGroups) {
    CHECK(getEditor() != NULL, );
    MSACollapsibleItemModel* m = ui->getCollapseModel();
    SAFE_POINT(NULL != m, tr("Incorrect pointer to MSACollapsibleItemModel"),);
    m->reset();

    MultipleSequenceAlignmentObject* msaObject = getEditor()->getMaObject();
    QStringList rowNames = msaObject->getMultipleAlignment()->getRowNames();
    QVector<U2Region> collapsedRegions;

    //Calculate regions of the groups
    foreach(const QStringList& seqsGroup, collapsedGroups) {
        int regionStart = rowNames.size(), regionEnd = 0;
        foreach(const QString& seqName, seqsGroup) {
            int seqPos = rowNames.indexOf(seqName);
            regionStart = qMin(seqPos, regionStart);
            regionEnd = qMax(seqPos, regionEnd);
        }
        if(regionStart > 0 && regionEnd <= rowNames.size() && regionEnd > regionStart) {
            collapsedRegions.append(U2Region(regionStart, regionEnd - regionStart + 1));
        }
    }
    //Function 'reset' in 'MSACollapsibleItemModel' work only with sorted regions
    qSort(collapsedRegions);

    if (msaObject == NULL || msaObject->isStateLocked()) {
        if (collapseModeSwitchAction->isChecked()) {
            collapseModeSwitchAction->setChecked(false);
        }
        return;
    }

    ui->setCollapsibleMode(true);

    m->reset(collapsedRegions);

    MaModificationInfo mi;
    msaObject->updateCachedMultipleAlignment(mi);

    ui->getScrollController()->updateVerticalScrollBar();
}

ExportHighligtningTask::ExportHighligtningTask(ExportHighligtingDialogController *dialog, MaEditorSequenceArea *msaese_)
    : Task(tr("Export highlighting"), TaskFlags_FOSCOE | TaskFlag_ReportingIsSupported | TaskFlag_ReportingIsEnabled)
{
    msaese = msaese_;
    startPos = dialog->startPos;
    endPos = dialog->endPos;
    startingIndex = dialog->startingIndex;
    keepGaps = dialog->keepGaps;
    dots = dialog->dots;
    transpose = dialog->transpose;
    url = dialog->url;
}

void ExportHighligtningTask::run(){
    QString exportedData = msaese->exportHighlighting(startPos, endPos, startingIndex, keepGaps, dots, transpose);

    QFile resultFile(url.getURLString());
    CHECK_EXT(resultFile.open(QFile::WriteOnly | QFile::Truncate), url.getURLString(),);
    QTextStream contentWriter(&resultFile);
    contentWriter << exportedData;
}

Task::ReportResult ExportHighligtningTask::report() {
    return ReportResult_Finished;
}

QString ExportHighligtningTask::generateReport() const {
    QString res;
    if(!isCanceled() && !hasError()){
        res += "<b>" + tr("Export highligtning finished successfully") + "</b><br><b>" + tr("Result file:") + "</b> " + url.getURLString();
    }
    return res;
}

}//namespace

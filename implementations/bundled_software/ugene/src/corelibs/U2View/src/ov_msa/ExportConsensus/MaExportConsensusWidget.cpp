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

#include <U2Algorithm/MSAConsensusAlgorithmRegistry.h>

#include <U2Core/AppContext.h>
#include <U2Core/AppSettings.h>
#include <U2Core/BaseDocumentFormats.h>
#include <U2Core/Counter.h>
#include <U2Core/DocumentModel.h>
#include <U2Core/GObjectTypes.h>
#include <U2Core/GUrlUtils.h>
#include <U2Core/L10n.h>
#include <U2Core/MultipleAlignmentObject.h>
#include <U2Core/TaskWatchdog.h>
#include <U2Core/U2IdTypes.h>
#include <U2Core/U2SafePoints.h>
#include <U2Core/UserApplicationsSettings.h>

#include <U2Formats/DocumentFormatUtils.h>

#include <U2Gui/DialogUtils.h>
#include <U2Gui/SaveDocumentController.h>
#include <U2Gui/ShowHideSubgroupWidget.h>
#include <U2Gui/U2WidgetStateStorage.h>

#include "ov_msa/MaEditor.h"
#include "ov_msa/view_rendering/MaEditorConsensusArea.h"
#include "ov_msa/view_rendering/MaEditorWgt.h"

#include <U2View/MaEditorTasks.h>

#include "MaExportConsensusWidget.h"

namespace U2 {

MaExportConsensusWidget::MaExportConsensusWidget(MaEditor* ma_, QWidget *parent)
    : QWidget(parent),
      ma(ma_),
      savableWidget(this, GObjectViewUtils::findViewByName(ma_->getName())),
      saveController(NULL)
{
    setupUi(this);

    hintLabel->setStyleSheet(L10N::infoHintStyleSheet());

    initSaveController();

    MaEditorConsensusArea *consensusArea = ma->getUI()->getConsensusArea();
    showHint(true);

    connect(exportBtn, SIGNAL(clicked()), SLOT(sl_exportClicked()));
    connect(consensusArea, SIGNAL(si_consensusAlgorithmChanged(const QString &)), SLOT(sl_consensusChanged(const QString &)));

    U2WidgetStateStorage::restoreWidgetState(savableWidget);
    sl_consensusChanged(consensusArea->getConsensusAlgorithm()->getId());
}

void MaExportConsensusWidget::sl_exportClicked(){
    GRUNTIME_NAMED_COUNTER(cvat, tvar, "Exporting of consensus", ma->getFactoryId());
    if (saveController->getSaveFileName().isEmpty()) {
        saveController->setPath(getDefaultFilePath());
    }

    ExportMaConsensusTaskSettings settings;
    settings.format = saveController->getFormatIdToSave();
    settings.keepGaps = keepGapsChb->isChecked() || keepGapsChb->isHidden();
    settings.ma = ma;
    settings.name = ma->getMaObject()->getGObjectName() + "_consensus";
    settings.url = saveController->getSaveFileName();
    settings.algorithm = ma->getUI()->getConsensusArea()->getConsensusAlgorithm()->clone();

    Task *t = new ExportMaConsensusTask(settings);
    TaskWatchdog::trackResourceExistence(ma->getMaObject(), t, tr("A problem occurred during export consensus. The multiple alignment is no more available."));
    AppContext::getTaskScheduler()->registerTopLevelTask(t);
}

void MaExportConsensusWidget::showHint( bool showHint ){
    if (showHint){
        hintLabel->show();
        keepGapsChb->hide();
    }else{
        hintLabel->hide();
        keepGapsChb->show();
    }
}

void MaExportConsensusWidget::sl_consensusChanged(const QString& algoId) {
    MSAConsensusAlgorithmFactory *consAlgorithmFactory = AppContext::getMSAConsensusAlgorithmRegistry()->getAlgorithmFactory(algoId);
    SAFE_POINT(consAlgorithmFactory != NULL, "Fetched consensus algorithm factory is NULL", );

    if (consAlgorithmFactory->isSequenceLikeResult()) {
        if (formatCb->count() == 1 ) { //only text
            formatCb->addItem(DocumentFormatUtils::getFormatNameById(BaseDocumentFormats::PLAIN_GENBANK));
            formatCb->addItem(DocumentFormatUtils::getFormatNameById(BaseDocumentFormats::FASTA));
            formatCb->model()->sort(0);
        } else {
            SAFE_POINT(formatCb->count() == 3, "Count of supported 'text' formats is not equal three", );
        }
        showHint(false);
    } else {
        if (formatCb->count() == 3 ) { //all possible formats
            formatCb->setCurrentText(DocumentFormatUtils::getFormatNameById(BaseDocumentFormats::PLAIN_TEXT));
            formatCb->removeItem(formatCb->findText(DocumentFormatUtils::getFormatNameById(BaseDocumentFormats::FASTA)));
            formatCb->removeItem(formatCb->findText(DocumentFormatUtils::getFormatNameById(BaseDocumentFormats::PLAIN_GENBANK)));
        } else {
            SAFE_POINT(formatCb->count() == 1, "Count of supported 'text' formats is not equal one", );
        }
        showHint(true);
    }
}

void MaExportConsensusWidget::initSaveController() {
    SaveDocumentControllerConfig config;
    config.defaultFileName = getDefaultFilePath();
    config.defaultFormatId = BaseDocumentFormats::PLAIN_TEXT;
    config.fileDialogButton = browseBtn;
    config.fileNameEdit = pathLe;
    config.formatCombo = formatCb;
    config.parentWidget = this;
    config.saveTitle = tr("Save file");

    const QList<DocumentFormatId> formats = QList<DocumentFormatId>() << BaseDocumentFormats::PLAIN_TEXT
                                                                      << BaseDocumentFormats::PLAIN_GENBANK
                                                                      << BaseDocumentFormats::FASTA;

    saveController = new SaveDocumentController(config, formats, this);
}

QString MaExportConsensusWidget::getDefaultFilePath() const {
    return GUrlUtils::getDefaultDataPath() + "/" + ma->getMaObject()->getGObjectName() + "_consensus.txt";
}

}

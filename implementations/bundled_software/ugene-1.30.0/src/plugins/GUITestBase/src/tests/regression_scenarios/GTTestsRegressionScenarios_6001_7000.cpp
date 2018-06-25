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

#include <drivers/GTKeyboardDriver.h>
#include <system/GTFile.h>
#include <primitives/PopupChooser.h>

#include <U2View/DetView.h>

#include <drivers/GTKeyboardDriver.h>

#include "GTTestsRegressionScenarios_6001_7000.h"
#include "GTUtilsAnnotationsTreeView.h"
#include "GTUtilsAssemblyBrowser.h"
#include "GTUtilsBookmarksTreeView.h"
#include "GTUtilsCircularView.h"
#include "GTUtilsDashboard.h"
#include "GTUtilsDocument.h"
#include "GTUtilsExternalTools.h"
#include "GTUtilsLog.h"
#include "GTUtilsMcaEditor.h"
#include "GTUtilsMcaEditorSequenceArea.h"
#include "GTUtilsMcaEditorStatusWidget.h"
#include "GTUtilsMdi.h"
#include "GTUtilsMsaEditor.h"
#include "GTUtilsMsaEditorSequenceArea.h"
#include "GTUtilsNotifications.h"
#include "GTUtilsOptionPanelMca.h"
#include "GTUtilsOptionPanelMSA.h"
#include "GTUtilsOptionPanelSequenceView.h"
#include "GTUtilsOptionsPanel.h"
#include "GTUtilsPcr.h"
#include "GTUtilsPhyTree.h"
#include "GTUtilsPrimerLibrary.h"
#include "GTUtilsProject.h"
#include "GTUtilsProjectTreeView.h"
#include "GTUtilsSequenceView.h"
#include "GTUtilsSharedDatabaseDocument.h"
#include "GTUtilsStartPage.h"
#include "GTUtilsTask.h"
#include "GTUtilsTaskTreeView.h"
#include "GTUtilsWizard.h"
#include "GTUtilsWorkflowDesigner.h"

#include "runnables/ugene/corelibs/U2Gui/EditSettingsDialogFiller.h"
#include "runnables/ugene/corelibs/U2Gui/ImportAPRFileDialogFiller.h"

namespace U2 {

namespace GUITest_regression_scenarios {
using namespace HI;

GUI_TEST_CLASS_DEFINITION(test_6031) {
    //1. Open samples/APR/gyrA.apr in read-only mode
    GTUtilsDialog::waitForDialog(os, new ImportAPRFileFiller(os, true));
    GTFileDialog::openFile(os, dataDir + "samples/APR/gyrA.apr");
    GTUtilsTaskTreeView::waitTaskFinished(os);

    //Expected: there is no prefix \27: before name of sequences
    QStringList nameList = GTUtilsMSAEditorSequenceArea::getNameList(os);
    foreach(const QString& name, nameList) {
        CHECK_SET_ERR(!name.startsWith("\27"), QString("Unexpected start of the name"));
    }
}

GUI_TEST_CLASS_DEFINITION(test_6033) {
    //    1. Open 'human_T1.fa'
    GTFileDialog::openFile(os, dataDir + "/samples/FASTA", "human_T1.fa");
    GTUtilsTaskTreeView::waitTaskFinished(os);

    //2. Select the whole sequence.
    GTUtilsSequenceView::selectSequenceRegion(os, 1, 199950);

    //3. Copy translation.
    GTKeyboardDriver::keyClick('t', Qt::ControlModifier);

    //4. Open the Project View, paste the data.
    if (!GTUtilsProjectTreeView::isVisible(os)) {
        GTUtilsProjectTreeView::openView(os);
    }

    GTUtilsProjectTreeView::click(os, "human_T1.fa");
    GTKeyboardDriver::keyClick('v', Qt::ControlModifier);
    GTUtilsTaskTreeView::waitTaskFinished(os);

    bool correct = false;
    foreach(const QString& name, GTUtilsProjectTreeView::getDocuments(os).keys()) {
        if (name.contains("clipboard") && name.contains(".seq")) {
            correct = true;
            break;
        }
    }

    CHECK_SET_ERR(correct, "Incorrect paste operation");
}

GUI_TEST_CLASS_DEFINITION(test_6043) {
//    1. Open "_common_data/ugenedb/sec1_9_ugenedb.ugenedb".
//    Expected state: the assembly is successfully opened, the coverage calculation finished, UGENE doens't crash
    const QString filePath = sandBoxDir + "test_6043.ugenedb";
    GTFile::copy(os, testDir + "_common_data/ugenedb/sec1_9_ugenedb.ugenedb", filePath);

    GTFileDialog::openFile(os, filePath);
    GTUtilsTaskTreeView::waitTaskFinished(os);
    GTGlobals::sleep();
    GTUtilsTaskTreeView::waitTaskFinished(os);

    const bool assemblyExists = GTUtilsProjectTreeView::checkItem(os, "sec1_and_others");
    CHECK_SET_ERR(assemblyExists, "Assembly object is not found in the project view");
}

GUI_TEST_CLASS_DEFINITION(test_6047) {
    //1. Open and convert APR file
    GTUtilsDialog::waitForDialog(os, new ImportAPRFileFiller(os, false, sandBoxDir + "test_6047", "MSF"));
    GTFileDialog::openFile(os, testDir + "_common_data/apr/HCVtest.apr");
    GTUtilsTaskTreeView::waitTaskFinished(os);

    //Check msa length and number of sequences
    CHECK_SET_ERR(GTUtilsMSAEditorSequenceArea::getLength(os) == 488, "Unexpected length of msa");
    CHECK_SET_ERR(GTUtilsMSAEditorSequenceArea::getNameList(os).size() == 231, "Unexpected quantaty of sequences");
}

GUI_TEST_CLASS_DEFINITION(test_6066) {
//    1. Open "data/samples/Genbank/murine.gb".
    GTFileDialog::openFile(os, dataDir + "samples/Genbank/murine.gb");
    GTUtilsTaskTreeView::waitTaskFinished(os);

//    2. Select "Edit" -> "Annotations settings on sequence editing..." menu item in the Details View context menu.
//    3. Choose "Split (separate annotations parts)" and press "OK".
    GTUtilsDialog::waitForDialog(os, new PopupChooserByText(os, QStringList() << "Edit" << "Annotations settings on sequence editing..."));
    GTUtilsDialog::waitForDialog(os, new EditSettingsDialogFiller(os, EditSettingsDialogFiller::SplitSeparateAnnotationParts, false));
    GTWidget::click(os, GTUtilsSequenceView::getDetViewByNumber(os), Qt::RightButton);

//    4. Turn on the editing mode.
    GTUtilsSequenceView::enableEditingMode(os);

//    5. Set cursor after position 60.
    GTUtilsSequenceView::setCursor(os, 60);

//    6. Click Space key.
    GTKeyboardDriver::keyClick(Qt::Key_Space);

//    Expected state: a gap is inserted, the annotation is split.
    // Do not check it here, to avoid view state changing

//    7. Doubleclick the first part if the split annotation and click Delete key.
    GTUtilsSequenceView::clickAnnotationDet(os, "misc_feature", 2, 0, true);
    GTGlobals::sleep(100);
    GTKeyboardDriver::keyClick(Qt::Key_Delete);
    GTGlobals::sleep();

//    Expected state: the annotation is removed from the Details View and from the Annotations Tree View.
    // Do not check it here, to avoid view state changing

//    8. Doubleclick the second part of the split annotation.
    GTUtilsSequenceView::clickAnnotationDet(os, "misc_feature", 3, 0, true);

//    Expected state: UGENE doesn't crash.
}

GUI_TEST_CLASS_DEFINITION(test_6071) {
    //1. Open "data/samples/Genbank/murine.gb".
    GTFileDialog::openFile(os, dataDir + "samples/Genbank/murine.gb");
    GTUtilsTaskTreeView::waitTaskFinished(os);

    //2. Select 2-th CDS annotation in the Zoom view
    GTUtilsSequenceView::clickAnnotationPan(os, "CDS", 2970);

    //3. Scroll to the 3874 coordinate.
    GTUtilsSequenceView::goToPosition(os, 3874);

    DetView* dw = GTUtilsSequenceView::getDetViewByNumber(os);
    const U2Region firstVisibleRange = dw->getVisibleRange();

    //4. Click on 2-th CDS join annotation
    GTUtilsSequenceView::clickAnnotationDet(os, "CDS", 3412);

    //Expected: visible range was not changed
    const U2Region secondVisibleRange = dw->getVisibleRange();
    CHECK_SET_ERR(firstVisibleRange == secondVisibleRange, "Visible range was changed after clicking on the annotation");
}

} // namespace GUITest_regression_scenarios

} // namespace U2

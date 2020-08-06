/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2020 UniPro <ugene@unipro.ru>
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

#include "FindEnzymesDialogFiller.h"
#include <primitives/GTCheckBox.h>
#include <primitives/GTLineEdit.h>
#include <primitives/GTTreeWidget.h>
#include <primitives/GTWidget.h>

#include <QApplication>
#include <QCheckBox>
#include <QTreeWidget>

namespace U2 {

#define GT_CLASS_NAME "FindEnzymesDialogFiller"

FindEnzymesDialogFiller::FindEnzymesDialogFiller(HI::GUITestOpStatus &os, const QStringList &enzymesToFind, CustomScenario *scenario)
    : Filler(os, "FindEnzymesDialog", scenario),
      enzymesToFind(enzymesToFind),
      searchStart(-1),
      searchEnd(-1),
      excludeStart(-1),
      excludeEnd(-1) {
}

FindEnzymesDialogFiller::FindEnzymesDialogFiller(GUITestOpStatus &os, const QStringList &enzymesToFind, qint64 searchRegionStart, qint64 searchRegionEnd, qint64 excludedRegionStart, qint64 excludedRegionEnd, CustomScenario *scenario)
    : Filler(os, "FindEnzymesDialog", scenario),
      enzymesToFind(enzymesToFind),
      searchStart(searchRegionStart),
      searchEnd(searchRegionEnd),
      excludeStart(excludedRegionStart),
      excludeEnd(excludedRegionEnd) {
}

#define GT_METHOD_NAME "run"
void FindEnzymesDialogFiller::commonScenario() {
    QWidget *dialog = QApplication::activeModalWidget();
    GT_CHECK(NULL != dialog, "activeModalWidget is NULL");

    QWidget *enzymesSelectorWidget = GTWidget::findWidget(os, "enzymesSelectorWidget");
    GT_CHECK(NULL != enzymesSelectorWidget, "enzymesSelectorWidget is NULL");

    GTWidget::click(os, GTWidget::findWidget(os, "selectNoneButton", enzymesSelectorWidget));

    QTreeWidget *enzymesTree = qobject_cast<QTreeWidget *>(GTWidget::findWidget(os, "tree", enzymesSelectorWidget));
    foreach (const QString &enzyme, enzymesToFind) {
        QTreeWidgetItem *item = GTTreeWidget::findItem(os, enzymesTree, enzyme);
        GTTreeWidget::checkItem(os, item);
    }

    if (searchStart != -1 && searchEnd != -1) {
        QWidget *regionSelector = GTWidget::findWidget(os, "region_selector_with_excluded");
        GT_CHECK(regionSelector != NULL, "range_selector not found");

        QLineEdit *start = GTWidget::findExactWidget<QLineEdit *>(os, "startLineEdit", regionSelector);
        GT_CHECK(start != NULL, "startLineEdit of 'Search In' region not found");
        GTLineEdit::setText(os, start, QString::number(searchStart));

        QLineEdit *end = GTWidget::findExactWidget<QLineEdit *>(os, "endLineEdit", regionSelector);
        GTWidget::click(os, end);
        GT_CHECK(end != NULL, "endLineEdit of 'Search In' region not found");
        GTLineEdit::setText(os, end, QString::number(searchEnd));
    }

    if (excludeStart != -1 && excludeEnd != -1) {
        QCheckBox *exclude = GTWidget::findExactWidget<QCheckBox *>(os, "excludeCheckBox");
        GTCheckBox::setChecked(os, exclude);

        QLineEdit *start = GTWidget::findExactWidget<QLineEdit *>(os, "excludeStartLineEdit");
        GT_CHECK(start != NULL, "excludeStartLineEdit of 'Exclude' region not found");
        GTLineEdit::setText(os, start, QString::number(excludeStart));

        QLineEdit *end = GTWidget::findExactWidget<QLineEdit *>(os, "excludeEndLinEdit");
        GTWidget::click(os, end);
        GT_CHECK(end != NULL, "excludeEndLinEdit of 'Exclude' region not found");
        GTLineEdit::setText(os, end, QString::number(excludeEnd));
    }

    GTUtilsDialog::clickButtonBox(os, dialog, QDialogButtonBox::Ok);
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}    // namespace U2

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

#include <QDesktopServices>
#include <QDesktopWidget>
#include <QFile>
#include <QScrollBar>
#include <QWebFrame>

#include <U2Core/Version.h>
#include <U2Core/U2SafePoints.h>

#include "StatisticalReportController.h"

#include "utils/MultilingualHtmlView.h"

namespace U2 {

StatisticalReportController::StatisticalReportController(const QString &newHtmlFilepath, QWidget *parent) : QDialog(parent) {
    setupUi(this);
    lblStat->setText(tr("<b>Optional:</b> Help make UGENE better by automatically sending anonymous usage statistics."));

    Version v = Version::appVersion();
    setWindowTitle(tr("Welcome to UGENE %1.%2").arg(v.major).arg(v.minor));

    htmlView = new MultilingualHtmlView(newHtmlFilepath, this);
    frameLayout->addWidget(htmlView);
    htmlView->setMinimumSize(400, 10);
    connect(buttonBox, SIGNAL(accepted()), SLOT(accept()));
}

bool StatisticalReportController::isInfoSharingAccepted() const {
    return chkStat->isChecked();
}

void StatisticalReportController::paintEvent(QPaintEvent *event) {
    QWidget::paintEvent(event);
    CHECK(!htmlView->page()->mainFrame()->scrollBarGeometry(Qt::Vertical).isEmpty(), );

    // adjust size to avoid scroll bars
    while (!htmlView->page()->mainFrame()->scrollBarGeometry(Qt::Vertical).isEmpty()) {
        htmlView->setMinimumHeight(htmlView->size().height() + 1);
    }
    htmlView->setMinimumHeight(htmlView->size().height() + 10);
#ifndef Q_OS_MAC
    // UGENE crashes on the update event processing on mac
    // It has some connection with htmlView loading method
    // There was no crash before f3a45ef1cd53fe28faf90a763d195e964bc6c752 commit
    // Find a solution and fix it, if you have some free time
    move(x(), (qApp->desktop()->screenGeometry().height() / 2) - htmlView->minimumHeight());
#endif
}

void StatisticalReportController::accept() {
    QDialog::close();
}

}

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

#ifndef _U2_APR_IMPORT_WIDGET_H_
#define _U2_APR_IMPORT_WIDGET_H_

#include <U2Formats/AprImporter.h>

#include "ImportWidget.h"
#include "ui_AprImportWidget.h"

namespace U2 {

class SaveDocumentController;

class AprImportWidget : public ImportWidget, public Ui_AprImportWidget {
    Q_OBJECT
public:
    AprImportWidget(const GUrl &url, const QVariantMap &settings);
    virtual QVariantMap getSettings() const;

    void initSaveController(const QString &url, const DocumentFormatId defaultFormatId);

private:
    DocumentFormatId getFormatId(const QVariantMap &settings);
    static QString getDefaultValues();
};

}    // namespace U2

#endif    // _U2_APR_IMPORT_WIDGET_H_

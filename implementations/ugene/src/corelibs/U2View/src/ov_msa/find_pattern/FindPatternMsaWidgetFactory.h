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

#ifndef _U2_FIND_PATTERN_MSA_WIDGET_FACTORY_H_
#define _U2_FIND_PATTERN_MSA_WIDGET_FACTORY_H_

#include <U2Gui/OPWidgetFactory.h>

namespace U2 {

class GObjectView;

class U2VIEW_EXPORT FindPatternMsaWidgetFactory : public OPWidgetFactory {
    Q_OBJECT
public:
    FindPatternMsaWidgetFactory();

    QWidget *createWidget(GObjectView *objView, const QVariantMap &options) override;

    OPGroupParameters getOPGroupParameters() override;

    void applyOptionsToWidget(QWidget *widget, const QVariantMap &options) override;

    static const QString &getGroupId();

    static const QVariantMap getOptionsToActivateSearchInSequences();

    static const QVariantMap getOptionsToActivateSearchInNames();

private:
    static const QString GROUP_ID;
    static const QString GROUP_ICON_STR;
    static const QString GROUP_DOC_PAGE;
};

}    // namespace U2

#endif

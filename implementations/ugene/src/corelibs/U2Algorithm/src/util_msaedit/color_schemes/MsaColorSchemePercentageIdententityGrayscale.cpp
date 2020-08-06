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

#include "MsaColorSchemePercentageIdententityGrayscale.h"

namespace U2 {

MsaColorSchemePercentageIdententityGrayscale::MsaColorSchemePercentageIdententityGrayscale(QObject *parent, const MsaColorSchemeFactory *factory, MultipleAlignmentObject *maObj)
    : MsaColorSchemePercentageIdentity(parent, factory, maObj) {
    colorsByRange[0] = QColor("#646464");
    colorsByRange[1] = QColor("#999999");
    colorsByRange[2] = QColor("#CCCCCC");
}

MsaColorSchemePercentageIdententityGrayscaleFactory::MsaColorSchemePercentageIdententityGrayscaleFactory(QObject *parent, const QString &id, const QString &name, const AlphabetFlags &supportedAlphabets)
    : MsaColorSchemeFactory(parent, id, name, supportedAlphabets) {
}

MsaColorScheme *MsaColorSchemePercentageIdententityGrayscaleFactory::create(QObject *parent, MultipleAlignmentObject *maObj) const {
    return new MsaColorSchemePercentageIdententityGrayscale(parent, this, maObj);
}

}    // namespace U2

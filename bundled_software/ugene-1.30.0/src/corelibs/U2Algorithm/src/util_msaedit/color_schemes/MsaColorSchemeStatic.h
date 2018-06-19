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

#ifndef _U2_MSA_COLOR_SCHEME_STATIC_H_
#define _U2_MSA_COLOR_SCHEME_STATIC_H_

#include <QColor>
#include <QVector>

#include "MsaColorScheme.h"

namespace U2 {

class U2ALGORITHM_EXPORT MsaColorSchemeStatic : public MsaColorScheme {
    Q_OBJECT
public:
    MsaColorSchemeStatic(QObject *parent, const MsaColorSchemeFactory *factory, MultipleAlignmentObject *maObj, const QVector<QColor> &colorsPerChar);

    QColor getColor(int seq, int pos, char c) const;

private:
    QVector<QColor> colorsPerChar;
};

class MsaColorSchemeStaticFactory : public MsaColorSchemeFactory {
    Q_OBJECT
public:
    MsaColorSchemeStaticFactory(QObject *parent, const QString &id, const QString &name, const AlphabetFlags &supportedAlphabets, const QVector<QColor> &colorsPerChar);

    MsaColorScheme * create(QObject* parent, MultipleAlignmentObject * maObj) const;

private:
    QVector<QColor> colorsPerChar;
};

}   // namespace U2

#endif // _U2_MSA_COLOR_SCHEME_STATIC_H_

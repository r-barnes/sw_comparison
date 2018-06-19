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

#ifndef _U2_JAVASUPPORT_H_
#define _U2_JAVASUPPORT_H_

#include <U2Core/ExternalToolRegistry.h>

#define ET_JAVA "java"

namespace U2 {

class JavaSupport : public ExternalTool {
    Q_OBJECT
public:
    enum Architecture {
        x32,
        x64
    };
    JavaSupport(const QString &name, const QString &path = "");
    void getAdditionalParameters(const QString& output);
    Architecture getArchitecture() const;

private slots:
    void sl_toolValidationStatusChanged(bool isValid);

private:
    static QString architecture2string(Architecture architecture);
    static Architecture string2architecture(const QString &string);

    static const QString ARCHITECTURE;
    static const QString ARCHITECTURE_X32;
    static const QString ARCHITECTURE_X64;
};

} // U2

#endif // _U2_JAVASUPPORT_H_

/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2017 UniPro <ugene@unipro.ru>
 * http://ugene.unipro.ru
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

#include "DashboardInfo.h"

#include <QDir>

namespace U2 {

DashboardInfo::DashboardInfo()
    : opened(false) {
}

DashboardInfo::DashboardInfo(const QString &dirPath, bool opened)
    : path(dirPath),
      opened(opened) {
    dirName = QDir(path).dirName();
}

const QString &DashboardInfo::getId() const {
    return path;
}

bool DashboardInfo::operator==(const DashboardInfo &other) const {
    return path == other.path;
}

}    // namespace U2

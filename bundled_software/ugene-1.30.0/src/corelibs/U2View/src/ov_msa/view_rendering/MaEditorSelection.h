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

#ifndef _U2_MA_EDITOR_SELECTION_
#define _U2_MA_EDITOR_SELECTION_

#include <QRect>

#include <U2Core/U2Region.h>

namespace U2 {

/************************************************************************/
/* MaEditorSelection */
/************************************************************************/
class U2VIEW_EXPORT MaEditorSelection {
public:
    MaEditorSelection();
    MaEditorSelection(int left, int top, int width, int height);
    MaEditorSelection(const QPoint& topLeft, const QPoint& bottomRight);
    MaEditorSelection(const QPoint& topLeft, int width, int height);

    bool isNull() const;
    bool isEmpty() const;

    QPoint topLeft() const;
    QPoint bottomRight() const;

    const QRect& getRect() const;

    int x() const;
    int y() const;

    int width() const;
    int height() const;

    int bottom() const;

    U2Region getXRegion() const;
    U2Region getYRegion() const;

    bool operator==(const MaEditorSelection& other) const;

    MaEditorSelection intersected(const MaEditorSelection& selection) const;

private:
    explicit MaEditorSelection(QRect& rect);
    QRect selArea;
};

} // namespace

#endif // _U2_MA_EDITOR_SELECTION_


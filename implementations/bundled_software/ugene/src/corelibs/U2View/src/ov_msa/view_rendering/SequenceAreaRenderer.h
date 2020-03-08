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

#ifndef _U2_SEQUENCE_AREA_RENDERER_H_
#define _U2_SEQUENCE_AREA_RENDERER_H_

#include <U2Core/DNAChromatogram.h>

#include <U2View/MSAEditorSequenceArea.h>

#include <QPen>

namespace U2 {

class SequenceAreaRenderer : public QObject {
    Q_OBJECT
public:
    SequenceAreaRenderer(MaEditorWgt *ui, MaEditorSequenceArea* seqAreaWgt);

    bool drawContent(QPainter &painter, const U2Region& region, const QList<int> &rowNumbers, int xStart, int yStart) const;

    void drawSelection(QPainter &painter) const;
    void drawFocus(QPainter &painter) const;

protected:
    // returns the height of the drawn row
    virtual int drawRow(QPainter &painter, const MultipleAlignment &ma, int rowIndex, const U2Region &region, int xStart, int yStart) const;

    MaEditorWgt *ui;
    MaEditorSequenceArea* seqAreaWgt;

    bool drawLeadingAndTrailingGaps;
};

} // namespace

#endif // _U2_SEQUENCE_AREA_RENDERER_H_


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

#ifndef _U2_SCROLL_CONTROLLER_H_
#define _U2_SCROLL_CONTROLLER_H_

#include <U2Core/U2Region.h>

namespace U2 {

class GScrollBar;
class MaEditor;
class MaEditorSelection;
class MaEditorWgt;
class MSACollapsibleItemModel;

class U2VIEW_EXPORT ScrollController : public QObject {
    Q_OBJECT
public:
    enum Direction {
        None = 0,
        Up = 1 << 0,
        Down = 1 << 1,
        Left = 1 << 2,
        Right = 1 << 3
    };
    Q_DECLARE_FLAGS(Directions, Direction)

    ScrollController(MaEditor *maEditor, MaEditorWgt *ui, MSACollapsibleItemModel *collapsibleModel);

    void init(GScrollBar *hScrollBar, GScrollBar *vScrollBar);

    QPoint getScreenPosition() const;       // in pixels
    QPoint getGlobalMousePosition(const QPoint& mousePos) const;

    void updateHorizontalScrollBar();
    void updateVerticalScrollBar();

    void scrollToRowByNumber(int rowNumber, int widgetHeight);
    void scrollToBase(int baseNumber, int widgetWidth);
    void scrollToPoint(const QPoint &maPoint, const QSize &screenSize);

    void centerBase(int baseNumber, int widgetWidth);
    void centerRow(int rowNumber, int widgetHeight);
    void centerPoint(const QPoint &maPoint, const QSize &widgetSize);

    void setHScrollbarValue(int value);
    void setVScrollbarValue(int value);

    void setFirstVisibleBase(int firstVisibleBase);
    void setFirstVisibleRowByNumber(int firstVisibleRowNumber);
    void setFirstVisibleRowByIndex(int firstVisibleRowIndex);

    void scrollSmoothly(const Directions &directions);
    void stopSmoothScrolling();

    void scrollStep(Direction direction);
    void scrollPage(Direction direction);
    void scrollToEnd(Direction direction);

    void scrollToMovedSelection(int deltaX, int deltaY);
    void scrollToMovedSelection(Direction direction);

    int getFirstVisibleBase(bool countClipped = false) const;
    int getLastVisibleBase(int widgetWidth, bool countClipped = false) const;
    int getFirstVisibleRowIndex(bool countClipped = false) const;
    int getFirstVisibleRowNumber(bool countClipped = false) const;
    int getLastVisibleRowIndex(int widgetHeight, bool countClipped = false) const;
    int getLastVisibleRowNumber(int widgetHeight, bool countClipped = false) const;

    QPoint getMaPointByScreenPoint(const QPoint &point) const;    // can be out of MA boundaries

    GScrollBar *getHorizontalScrollBar() const;
    GScrollBar *getVerticalScrollBar() const;

signals:
    void si_visibleAreaChanged();

public slots:
    void sl_updateScrollBars();
    void sl_zoomScrollBars();

private slots:
    void sl_collapsibleModelIsAboutToBeChanged();
    void sl_collapsibleModelChanged();

private:
    int getAdditionalXOffset() const;       // in pixels;
    int getAdditionalYOffset() const;       // in pixels;

    U2Region getHorizontalRangeToDrawIn(int widgetWidth) const;     // in pixels
    U2Region getVerticalRangeToDrawIn(int widgetHeight) const;       // in pixels

    void zoomHorizontalScrollBarPrivate();
    void zoomVerticalScrollBarPrivate();
    void updateHorizontalScrollBarPrivate();
    void updateVerticalScrollBarPrivate();

    MaEditor *maEditor;
    MaEditorWgt *ui;
    MSACollapsibleItemModel *collapsibleModel;
    GScrollBar *hScrollBar;
    GScrollBar *vScrollBar;

    int savedFirstVisibleRowIndex;
    int savedFirstVisibleRowAdditionalOffset;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(ScrollController::Directions)

}   // namespace U2

#endif // _U2_SCROLL_CONTROLLER_H_

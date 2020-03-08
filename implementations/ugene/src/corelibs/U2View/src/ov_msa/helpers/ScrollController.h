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

#ifndef _U2_SCROLL_CONTROLLER_H_
#define _U2_SCROLL_CONTROLLER_H_

#include <U2Core/U2Region.h>

namespace U2 {

class GScrollBar;
class MaEditor;
class MaEditorSelection;
class MaEditorWgt;
class MaCollapseModel;

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

    ScrollController(MaEditor *maEditor, MaEditorWgt *ui, MaCollapseModel *collapsibleModel);

    void init(GScrollBar *hScrollBar, GScrollBar *vScrollBar);

    QPoint getScreenPosition() const;       // in pixels
    QPoint getGlobalMousePosition(const QPoint& mousePos) const;

    void updateVerticalScrollBar();

    void scrollToViewRow(int viewRowIndex, int widgetHeight);
    void scrollToBase(int baseNumber, int widgetWidth);
    void scrollToPoint(const QPoint &maPoint, const QSize &screenSize);

    void centerBase(int baseNumber, int widgetWidth);
    void centerViewRow(int viewRowIndex, int widgetHeight);
    void centerPoint(const QPoint &maPoint, const QSize &widgetSize);

    void setHScrollbarValue(int value);
    void setVScrollbarValue(int value);

    void setFirstVisibleBase(int firstVisibleBase);
    void setFirstVisibleViewRow(int viewRowIndex);
    void setFirstVisibleMaRow(int maRowIndex);

    void scrollSmoothly(const Directions &directions);
    void stopSmoothScrolling();

    void scrollStep(Direction direction);
    void scrollPage(Direction direction);
    void scrollToEnd(Direction direction);

    void scrollToMovedSelection(int deltaX, int deltaY);
    void scrollToMovedSelection(Direction direction);

    int getFirstVisibleBase(bool countClipped = false) const;
    int getLastVisibleBase(int widgetWidth, bool countClipped = false) const;
    int getFirstVisibleMaRowIndex(bool countClipped = false) const;
    int getFirstVisibleViewRowIndex(bool countClipped = false) const;
    int getLastVisibleViewRowIndex(int widgetHeight, bool countClipped = false) const;


    /*
     * Maps screen coordinates into QPoint(row, column).
     * Returns QPoint(-1, -1) if geom. position can't be mapped to any base and reportOverflow is false.
     * If reportOverflow is true and one of the coordinates has overflow, returns rowCount/columnsCount for it.
     */
    QPoint getViewPosByScreenPoint(const QPoint& point, bool reportOverflow = true) const;

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
    MaCollapseModel *collapsibleModel;
    GScrollBar *hScrollBar;
    GScrollBar *vScrollBar;

    int savedFirstVisibleMaRow;
    int savedFirstVisibleMaRowOffset;
};

Q_DECLARE_OPERATORS_FOR_FLAGS(ScrollController::Directions)

}   // namespace U2

#endif // _U2_SCROLL_CONTROLLER_H_

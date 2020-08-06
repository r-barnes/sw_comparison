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

#ifndef _U2_MSAEDITOR_MULTI_TREE_VIEWER_H_
#define _U2_MSAEDITOR_MULTI_TREE_VIEWER_H_

#include <QStringList>
#include <QToolBar>
#include <QWidget>

#include <U2Core/global.h>

namespace U2 {

class MSAEditor;
class MsaEditorTreeTabArea;
class GObjectViewWindow;
class MsaEditorTreeTab;

class U2VIEW_EXPORT MSAEditorMultiTreeViewer : public QWidget {
    Q_OBJECT
public:
    MSAEditorMultiTreeViewer(QString _title, MSAEditor *_editor);
    ~MSAEditorMultiTreeViewer() {
    }

    void addTreeView(GObjectViewWindow *treeView);

    QWidget *getCurrentWidget() const;

    MsaEditorTreeTab *getCurrentTabWidget() const;

    const QStringList &getTreeNames() const;
signals:
    void si_tabsCountChanged(int tabsCount);
public slots:
    void sl_onTabCloseRequested(QWidget *);

private:
    MsaEditorTreeTabArea *treeTabs;
    QWidget *titleWidget;
    MSAEditor *editor;
    QList<QWidget *> treeViews;
    QStringList tabsNames;
};

}    // namespace U2
#endif

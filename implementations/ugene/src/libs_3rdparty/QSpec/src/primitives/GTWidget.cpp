/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2017 UniPro <ugene@unipro.ru>
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

#include "primitives/GTWidget.h"

#include <QApplication>
#include <QComboBox>
#include <QDesktopWidget>
#include <QGuiApplication>
#include <QLineEdit>
#include <QStyle>

#include "drivers/GTMouseDriver.h"
#include "primitives/GTMainWindow.h"
#include "utils/GTThread.h"

namespace HI {
#define GT_CLASS_NAME "GTWidget"

#define GT_METHOD_NAME "click"
void GTWidget::click(GUITestOpStatus &os, QWidget *widget, Qt::MouseButton mouseButton, QPoint p) {
    GT_CHECK(widget != nullptr, "widget is NULL");

    if (p.isNull()) {
        p = widget->rect().center();
        // TODO: this is a fast fix
        if (widget->objectName().contains("ADV_single_sequence_widget")) {
            p += QPoint(0, 8);
        }
    }
    QPoint globalPoint = widget->mapToGlobal(p);
    GTMouseDriver::moveTo(globalPoint);
    GTMouseDriver::click(mouseButton);
    GTThread::waitForMainThread();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "setFocus"
void GTWidget::setFocus(GUITestOpStatus &os, QWidget *w) {
    GT_CHECK(w != NULL, "widget is NULL");

    GTWidget::click(os, w);
    GTGlobals::sleep(200);

    if (!qobject_cast<QComboBox *>(w)) {
        GT_CHECK(w->hasFocus(), QString("Can't set focus on widget '%1'").arg(w->objectName()));
    }
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "findWidget"
QWidget *GTWidget::findWidget(GUITestOpStatus &os, const QString &widgetName, QWidget const *const parentWidget, const GTGlobals::FindOptions &options) {
    QWidget *widget = nullptr;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && widget == nullptr; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        if (parentWidget == nullptr) {
            QList<QWidget *> allWidgetList;
            foreach (QWidget *parent, GTMainWindow::getMainWindowsAsWidget(os)) {
                allWidgetList << parent->findChildren<QWidget *>(widgetName);
            }
            int nMatches = allWidgetList.count();
            GT_CHECK_RESULT(nMatches < 2, QString("There are %1 widgets with name '%2'").arg(nMatches).arg(widgetName), nullptr);
            widget = nMatches == 1 ? allWidgetList.first() : nullptr;
        } else {
            widget = parentWidget->findChild<QWidget *>(widgetName);
        }
        if (!options.failIfNotFound) {
            break;
        }
    }
    if (options.failIfNotFound) {
        GT_CHECK_RESULT(widget != nullptr, QString("Widget '%1' not found").arg(widgetName), NULL);
    }
    return widget;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getWidgetCenter"
QPoint GTWidget::getWidgetCenter(QWidget *widget) {
    return widget->mapToGlobal(widget->rect().center());
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "findButtonByText"
QAbstractButton *GTWidget::findButtonByText(GUITestOpStatus &os, const QString &text, QWidget *parentWidget, const GTGlobals::FindOptions &options) {
    QList<QAbstractButton *> resultButtonList;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && resultButtonList.isEmpty(); time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        QList<QAbstractButton *> allButtonList;
        if (parentWidget == nullptr) {
            foreach (QWidget *mainWidget, GTMainWindow::getMainWindowsAsWidget(os)) {
                allButtonList << mainWidget->findChildren<QAbstractButton *>();
            }
        } else {
            allButtonList << parentWidget->findChildren<QAbstractButton *>();
        }
        foreach (QAbstractButton *button, allButtonList) {
            if (button->text().contains(text, Qt::CaseInsensitive)) {
                resultButtonList << button;
            }
        }
        if (!options.failIfNotFound) {
            break;
        }
    }
    GT_CHECK_RESULT(resultButtonList.count() <= 1, QString("There are %1 buttons with text").arg(resultButtonList.count()), nullptr);
    if (options.failIfNotFound) {
        GT_CHECK_RESULT(resultButtonList.count() != 0, QString("Button with the text <%1> is not found").arg(text), nullptr);
    }
    return resultButtonList.isEmpty() ? nullptr : resultButtonList.first();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "findLabelByText"
QList<QLabel *> GTWidget::findLabelByText(GUITestOpStatus &os,
                                          const QString &text,
                                          QWidget *parentWidget,
                                          const GTGlobals::FindOptions &options) {
    QList<QLabel *> resultLabelList;
    for (int time = 0; time < GT_OP_WAIT_MILLIS; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS && resultLabelList.isEmpty() : 0);
        QList<QLabel *> allLabelList;
        if (parentWidget == nullptr) {
            foreach (QWidget *windowWidget, GTMainWindow::getMainWindowsAsWidget(os)) {
                allLabelList << windowWidget->findChildren<QLabel *>();
            }
        } else {
            allLabelList << parentWidget->findChildren<QLabel *>();
        }
        foreach (QLabel *label, allLabelList) {
            if (label->text().contains(text, Qt::CaseInsensitive)) {
                resultLabelList << label;
            }
        }
    }
    if (options.failIfNotFound) {
        GT_CHECK_RESULT(resultLabelList.count() > 0, QString("Label with this text <%1> not found").arg(text), QList<QLabel *>());
    }
    return resultLabelList;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "close"
void GTWidget::close(GUITestOpStatus &os, QWidget *widget) {
    GT_CHECK(widget != nullptr, "Widget is NULL");

    class Scenario : public CustomScenario {
    public:
        Scenario(QWidget *widget)
            : widget(widget) {
        }

        void run(GUITestOpStatus &os) {
            Q_UNUSED(os);
            CHECK_SET_ERR(widget != nullptr, "Widget is NULL");
            widget->close();
            GTGlobals::sleep(100);
        }

    private:
        QWidget *widget;
    };

    GTThread::runInMainThread(os, new Scenario(widget));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "showMaximized"
void GTWidget::showMaximized(GUITestOpStatus &os, QWidget *widget) {
    GT_CHECK(widget != nullptr, "Widget is NULL");

    class Scenario : public CustomScenario {
    public:
        Scenario(QWidget *widget)
            : widget(widget) {
        }

        void run(GUITestOpStatus &os) {
            Q_UNUSED(os);
            CHECK_SET_ERR(widget != nullptr, "Widget is NULL");
            widget->showMaximized();
            GTGlobals::sleep(100);
        }

    private:
        QWidget *widget;
    };

    GTThread::runInMainThread(os, new Scenario(widget));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "showNormal"
void GTWidget::showNormal(GUITestOpStatus &os, QWidget *widget) {
    GT_CHECK(widget != nullptr, "Widget is NULL");

    class Scenario : public CustomScenario {
    public:
        Scenario(QWidget *widget)
            : widget(widget) {
        }

        void run(GUITestOpStatus &os) {
            Q_UNUSED(os);
            CHECK_SET_ERR(widget != nullptr, "Widget is NULL");
            widget->showNormal();
            GTGlobals::sleep(100);
        }

    private:
        QWidget *widget;
    };

    GTThread::runInMainThread(os, new Scenario(widget));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getColor"
QColor GTWidget::getColor(GUITestOpStatus &os, QWidget *widget, const QPoint &point) {
    GT_CHECK_RESULT(widget != nullptr, "Widget is NULL", QColor());
    return QColor(getImage(os, widget).pixel(point));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getImage"
QImage GTWidget::getImage(GUITestOpStatus &os, QWidget *widget) {
    GT_CHECK_RESULT(widget != nullptr, "Widget is NULL", QImage());

    class Scenario : public CustomScenario {
    public:
        Scenario(QWidget *widget, QImage &image)
            : widget(widget),
              image(image) {
        }

        void run(GUITestOpStatus &os) {
            CHECK_SET_ERR(widget != nullptr, "Widget to grab is NULL");
            image = widget->grab(widget->rect()).toImage();
        }

    private:
        QWidget *widget;
        QImage &image;
    };

    QImage image;
    GTThread::runInMainThread(os, new Scenario(widget, image));
    return image;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickLabelLink"
void GTWidget::clickLabelLink(GUITestOpStatus &os, QWidget *label, int step, int indent) {
    QRect r = label->rect();

    int left = r.left();
    int right = r.right();
    int top = r.top() + indent;
    int bottom = r.bottom();
    for (int i = left; i < right; i += step) {
        for (int j = top; j < bottom; j += step) {
            GTMouseDriver::moveTo(label->mapToGlobal(QPoint(i, j)));
            if (label->cursor().shape() == Qt::PointingHandCursor) {
                GTGlobals::sleep(500);
                GTMouseDriver::click();
                return;
            }
        }
    }
    GT_CHECK(false, "label does not contain link");
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickWindowTitle"
void GTWidget::clickWindowTitle(GUITestOpStatus &os, QWidget *window) {
    GT_CHECK(window != nullptr, "Window is NULL");

    QStyleOptionTitleBar opt;
    opt.initFrom(window);
    const QRect titleLabelRect = window->style()->subControlRect(QStyle::CC_TitleBar, &opt, QStyle::SC_TitleBarLabel);
    GTMouseDriver::moveTo(getWidgetGlobalTopLeftPoint(os, window) + titleLabelRect.center());
    GTMouseDriver::click();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "moveWidgetTo"
void GTWidget::moveWidgetTo(GUITestOpStatus &os, QWidget *window, const QPoint &point) {
    //QPoint(window->width()/2,3) - is hack
    GTMouseDriver::moveTo(getWidgetGlobalTopLeftPoint(os, window) + QPoint(window->width() / 2, 3));
    const QPoint p0 = getWidgetGlobalTopLeftPoint(os, window) + QPoint(window->width() / 2, 3);
    const QPoint p1 = point + QPoint(window->width() / 2, 3);
    GTMouseDriver::dragAndDrop(p0, p1);
    GTGlobals::sleep(1000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "resizeWidget"
void GTWidget::resizeWidget(GUITestOpStatus &os, QWidget *widget, const QSize &size) {
    GT_CHECK(widget != nullptr, "Widget is NULL");

    QRect displayRect = QApplication::desktop()->screenGeometry();
    GT_CHECK((displayRect.width() >= size.width()) && (displayRect.height() >= size.height()), "Specified the size larger than the size of the screen");

    bool isRequiredPositionFound = false;
    QSize oldSize = widget->size();

    QPoint topLeftPos = getWidgetGlobalTopLeftPoint(os, widget) + QPoint(5, 5);
    for (int i = 0; i < 5; i++) {
        GTMouseDriver::moveTo(topLeftPos);
        QPoint newTopLeftPos = topLeftPos + QPoint(widget->frameGeometry().width() - 1, widget->frameGeometry().height() - 1) - QPoint(size.width(), size.height());
        GTMouseDriver::dragAndDrop(topLeftPos, newTopLeftPos);
        if (widget->size() != oldSize) {
            isRequiredPositionFound = true;
            break;
        } else {
            topLeftPos -= QPoint(1, 1);
        }
    }
    GT_CHECK(isRequiredPositionFound, "Required mouse position to start window resize was not found");
    GTGlobals::sleep(1000);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getWidgetGlobalTopLeftPoint"
QPoint GTWidget::getWidgetGlobalTopLeftPoint(GUITestOpStatus &os, QWidget *widget) {
    GT_CHECK_RESULT(widget != nullptr, "Widget is NULL", QPoint());
    return (widget->isWindow() ? widget->pos() : widget->parentWidget()->mapToGlobal(QPoint(0, 0)));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getActiveModalWidget"
QWidget *GTWidget::getActiveModalWidget(GUITestOpStatus &os) {
    QWidget *modalWidget = nullptr;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && modalWidget == nullptr; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        modalWidget = QApplication::activeModalWidget();
    }
    GT_CHECK_RESULT(modalWidget != nullptr, "Active modal widget is NULL", nullptr);
    return modalWidget;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getActivePopupWidget"
QWidget *GTWidget::getActivePopupWidget(GUITestOpStatus &os) {
    QWidget *popupWidget = nullptr;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && popupWidget == nullptr; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        popupWidget = QApplication::activePopupWidget();
    }
    GT_CHECK_RESULT(popupWidget != nullptr, "Active popup widget is NULL", nullptr);
    return popupWidget;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getActivePopupMenu"
QMenu *GTWidget::getActivePopupMenu(GUITestOpStatus &os) {
    QMenu *popupWidget = nullptr;
    for (int time = 0; time < GT_OP_WAIT_MILLIS && popupWidget == nullptr; time += GT_OP_CHECK_MILLIS) {
        GTGlobals::sleep(time > 0 ? GT_OP_CHECK_MILLIS : 0);
        popupWidget = qobject_cast<QMenu *>(QApplication::activePopupWidget());
    }
    GT_CHECK_RESULT(popupWidget != nullptr, "Active popup menu is NULL", nullptr);
    return popupWidget;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "checkEnabled"
void GTWidget::checkEnabled(GUITestOpStatus &os, QWidget *widget, bool expectedEnabledState) {
    GT_CHECK(widget != nullptr, "Widget is NULL");
    GT_CHECK(widget->isVisible(), "Widget is not visible");
    bool actualEnabledState = widget->isEnabled();
    GT_CHECK(expectedEnabledState == actualEnabledState,
             QString("Widget state is incorrect: expected '%1', got '%'2")
                 .arg(expectedEnabledState ? "enabled" : "disabled")
                 .arg(actualEnabledState ? "enabled" : "disabled"));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "checkEnabled"
void GTWidget::checkEnabled(GUITestOpStatus &os, const QString &widgetName, bool expectedEnabledState, QWidget const *const parent) {
    checkEnabled(os, GTWidget::findWidget(os, widgetName, parent), expectedEnabledState);
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}    // namespace HI

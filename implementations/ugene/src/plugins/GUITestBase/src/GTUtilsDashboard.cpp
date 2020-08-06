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

#include <GTUtilsMdi.h>
#include <primitives/GTTabWidget.h>
#include <primitives/GTWebView.h>
#include <primitives/GTWidget.h>

#include <QRegularExpression>
#include <QTabWidget>

#include <U2Designer/Dashboard.h>

#include "GTUtilsDashboard.h"

namespace U2 {
using namespace HI;

#define GT_CLASS_NAME "GTUtilsDashboard"
QString GTUtilsDashboard::getNodeSpanId(const QString &nodeId) {
    // It is defined in ExternalToolsWidget.js.
    return nodeId + "_span";
}

HIWebElement GTUtilsDashboard::getCopyButton(GUITestOpStatus &os, const QString &toolRunNodeId) {
    const QString selector = QString("SPAN#%1 > BUTTON").arg(getNodeSpanId(toolRunNodeId));

    GTGlobals::FindOptions options;
    options.searchInHidden = true;

    return GTWebView::findElementBySelector(os, getDashboardWebView(os), selector, options);
}

HIWebElement GTUtilsDashboard::getNodeSpan(GUITestOpStatus &os, const QString &nodeId) {
    const QString selector = QString("SPAN#%1").arg(getNodeSpanId(nodeId));

    GTGlobals::FindOptions options;
    options.searchInHidden = true;

    return GTWebView::findElementBySelector(os, getDashboardWebView(os), selector, options);
}

#define GT_METHOD_NAME "clickOutputFile"
QString GTUtilsDashboard::getLogUrlFromElement(GUITestOpStatus &os, const HIWebElement &element) {
    Q_UNUSED(os);
    const QString onclickFunction = element.attribute(ON_CLICK);
    QRegularExpression urlFetcher("openLog\\(\\\'(.*)\\\'\\)");
    const QRegularExpressionMatch match = urlFetcher.match(onclickFunction);
    GT_CHECK_RESULT(match.hasMatch(),
                    QString("Can't get URL with a regexp from an element: regexp is '%1', element ID is '%2', element class is '%3'")
                        .arg(urlFetcher.pattern())
                        .arg(element.id())
                        .arg(element.attribute("class")),
                    QString());
    return match.captured(1);
}
#undef GT_METHOD_NAME

const QString GTUtilsDashboard::TREE_ROOT_ID = "treeRoot";
const QString GTUtilsDashboard::PARENT_LI = "parent_li";

const QString GTUtilsDashboard::TITLE = "title";
const QString GTUtilsDashboard::COLLAPSED_NODE_TITLE = "Expand this branch";
const QString GTUtilsDashboard::ON_CLICK = "onclick";

WebView *GTUtilsDashboard::getDashboardWebView(HI::GUITestOpStatus &os) {
    Dashboard *dashboard = findDashboard(os);
    return dashboard == nullptr ? nullptr : dashboard->getWebView();
}

QTabWidget *GTUtilsDashboard::getTabWidget(HI::GUITestOpStatus &os) {
    return GTWidget::findExactWidget<QTabWidget *>(os, "WorkflowTabView", GTUtilsMdi::activeWindow(os));
}

QToolButton *GTUtilsDashboard::findLoadSchemaButton(HI::GUITestOpStatus &os) {
    Dashboard *dashboard = findDashboard(os);
    return dashboard == nullptr ? nullptr : dashboard->findChild<QToolButton *>("loadSchemaButton");
}

const QString GTUtilsDashboard::getDashboardName(GUITestOpStatus &os, int dashboardNumber) {
    return GTTabWidget::getTabName(os, getTabWidget(os), dashboardNumber);
}

QStringList GTUtilsDashboard::getOutputFiles(HI::GUITestOpStatus &os) {
    QString selector = "div#outputWidget button.btn.full-width.long-text";
    QList<HIWebElement> outputFilesButtons = GTWebView::findElementsBySelector(os, getDashboardWebView(os), selector, GTGlobals::FindOptions(false));
    QStringList outputFilesNames;
    foreach (const HIWebElement &outputFilesButton, outputFilesButtons) {
        const QString outputFileName = outputFilesButton.toPlainText();
        if (!outputFileName.isEmpty()) {
            outputFilesNames << outputFileName;
        }
    }
    return outputFilesNames;
}

#define GT_METHOD_NAME "clickOutputFile"
void GTUtilsDashboard::clickOutputFile(GUITestOpStatus &os, const QString &outputFileName) {
    const QString selector = "div#outputWidget button.btn.full-width.long-text";
    const QList<HIWebElement> outputFilesButtons = GTWebView::findElementsBySelector(os, getDashboardWebView(os), selector);
    foreach (const HIWebElement &outputFilesButton, outputFilesButtons) {
        QString buttonText = outputFilesButton.toPlainText();
        if (buttonText == outputFileName) {
            click(os, outputFilesButton);
            return;
        }

        if (buttonText.endsWith("...")) {
            buttonText.chop(QString("...").length());
            if (!buttonText.isEmpty() && outputFileName.startsWith(buttonText)) {
                click(os, outputFilesButton);
                return;
            }
        }
    }

    GT_CHECK(false, QString("The output file with name '%1' not found").arg(outputFileName));
}
#undef GT_METHOD_NAME

HIWebElement GTUtilsDashboard::findElement(HI::GUITestOpStatus &os, QString text, QString tag, bool exactMatch) {
    return GTWebView::findElement(os, getDashboardWebView(os), text, tag, exactMatch);
}

HIWebElement GTUtilsDashboard::findTreeElement(HI::GUITestOpStatus &os, QString text) {
    return GTWebView::findTreeElement(os, getDashboardWebView(os), text);
}

HIWebElement GTUtilsDashboard::findContextMenuElement(HI::GUITestOpStatus &os, QString text) {
    return GTWebView::findContextMenuElement(os, getDashboardWebView(os), text);
}

void GTUtilsDashboard::click(HI::GUITestOpStatus &os, HIWebElement el, Qt::MouseButton button) {
    GTWebView::click(os, getDashboardWebView(os), el, button);
}

bool GTUtilsDashboard::areThereNotifications(HI::GUITestOpStatus &os) {
    openTab(os, Overview);
    return GTWebView::doesElementExist(os, getDashboardWebView(os), "Notifications", "DIV", true);
}

QString GTUtilsDashboard::getTabObjectName(Tabs tab) {
    switch (tab) {
    case Overview:
        return "overviewTabButton";
    case Input:
        return "inputTabButton";
    case ExternalTools:
        return "externalToolsTabButton";
    }
    return "unknown tab";
}

Dashboard *GTUtilsDashboard::findDashboard(HI::GUITestOpStatus &os) {
    QTabWidget *tabWidget = getTabWidget(os);
    return tabWidget == nullptr ? nullptr : qobject_cast<Dashboard *>(tabWidget->currentWidget());
}

#define GT_METHOD_NAME "openTab"
void GTUtilsDashboard::openTab(HI::GUITestOpStatus &os, Tabs tab) {
    QWidget *dashboard = findDashboard(os);
    GT_CHECK(dashboard != nullptr, "Dashboard widget not found");

    QString tabButtonObjectName = getTabObjectName(tab);
    QToolButton *tabButton = GTWidget::findExactWidget<QToolButton *>(os, tabButtonObjectName, dashboard);
    GT_CHECK(tabButton != nullptr, "Tab button not found: " + tabButtonObjectName);

    GTWidget::click(os, tabButton);
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "doesTabExist"
bool GTUtilsDashboard::doesTabExist(HI::GUITestOpStatus &os, Tabs tab) {
    QWidget *dashboard = findDashboard(os);
    GT_CHECK_RESULT(dashboard != nullptr, "Dashboard is not found", false);

    QString tabButtonObjectName = getTabObjectName(tab);
    QWidget *button = dashboard->findChild<QWidget *>(tabButtonObjectName);
    return button != nullptr && button->isVisible();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getNodeText"
QString GTUtilsDashboard::getNodeText(GUITestOpStatus &os, const QString &nodeId) {
    return getNodeSpan(os, nodeId).toPlainText();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getChildrenNodesCount"
int GTUtilsDashboard::getChildrenNodesCount(GUITestOpStatus &os, const QString &nodeId) {
    const QString selector = QString("UL#%1 > LI.%2 > UL").arg(nodeId).arg(PARENT_LI);

    GTGlobals::FindOptions options;
    options.failIfNotFound = false;
    options.searchInHidden = true;

    return GTWebView::findElementsBySelector(os, getDashboardWebView(os), selector, options).size();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getChildNodeId"
QString GTUtilsDashboard::getChildNodeId(GUITestOpStatus &os, const QString &nodeId, int childNum) {
    return getDescendantNodeId(os, nodeId, {childNum});
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getDescendantNodeId"
QString GTUtilsDashboard::getDescendantNodeId(GUITestOpStatus &os, const QString &nodeId, const QList<int> &childNums) {
    QString selector = QString("UL#%1").arg(nodeId);
    foreach (const int childNum, childNums) {
        selector += QString(" > LI:nth-of-type(%1) > UL").arg(childNum + 1);
    }

    GTGlobals::FindOptions options;
    options.searchInHidden = true;

    return GTWebView::findElementBySelector(os, getDashboardWebView(os), selector, options).id();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getChildWithTextId"
QString GTUtilsDashboard::getChildWithTextId(GUITestOpStatus &os, const QString &nodeId, const QString &text) {
    const int childrenCount = getChildrenNodesCount(os, nodeId);
    QString resultChildId;
    QStringList quotedChildrenTexts;
    for (int i = 0; i < childrenCount; i++) {
        const QString currentChildId = getChildNodeId(os, nodeId, i);
        const QString childText = getNodeText(os, currentChildId);
        quotedChildrenTexts << "\'" + childText + "\'";
        if (text == childText) {
            GT_CHECK_RESULT(resultChildId.isEmpty(),
                            QString("Expected text '%1' is not unique among the node with ID '%2' children")
                                .arg(text)
                                .arg(nodeId),
                            "");
            resultChildId = currentChildId;
        }
    }

    GT_CHECK_RESULT(!resultChildId.isEmpty(),
                    QString("Child with text '%1' not found among the node with ID '%2' children; there are children with the following texts: %3")
                        .arg(text)
                        .arg(nodeId)
                        .arg(quotedChildrenTexts.join(", ")),
                    "");

    return resultChildId;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "doesNodeHaveLimitationMessageNode"
bool GTUtilsDashboard::doesNodeHaveLimitationMessageNode(GUITestOpStatus &os, const QString &nodeId) {
    const QString selector = QString("UL#%1 > LI.%2 > SPAN.limitation-message").arg(nodeId).arg(PARENT_LI);

    GTGlobals::FindOptions options;
    options.failIfNotFound = false;

    return !GTWebView::findElementsBySelector(os, getDashboardWebView(os), selector, options).isEmpty();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getLimitationMessageNodeText"
QString GTUtilsDashboard::getLimitationMessageNodeText(GUITestOpStatus &os, const QString &nodeId) {
    const QString selector = QString("UL#%1 > LI.%2 > SPAN.limitation-message").arg(nodeId).arg(PARENT_LI);
    return GTWebView::findElementBySelector(os, getDashboardWebView(os), selector).toPlainText();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getLimitationMessageLogUrl"
QString GTUtilsDashboard::getLimitationMessageLogUrl(GUITestOpStatus &os, const QString &nodeId) {
    const QString selector = QString("UL#%1 > LI.%2 > SPAN.limitation-message > A").arg(nodeId).arg(PARENT_LI);
    return getLogUrlFromElement(os, GTWebView::findElementBySelector(os, getDashboardWebView(os), selector));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getCopyButtonSize"
QSize GTUtilsDashboard::getCopyButtonSize(GUITestOpStatus &os, const QString &toolRunNodeId) {
    return getCopyButton(os, toolRunNodeId).geometry().size();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "clickCopyButton"
void GTUtilsDashboard::clickCopyButton(GUITestOpStatus &os, const QString &toolRunNodeId) {
    click(os, getCopyButton(os, toolRunNodeId));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isNodeVisible"
bool GTUtilsDashboard::isNodeVisible(GUITestOpStatus &os, const QString &nodeId) {
    return getNodeSpan(os, nodeId).isVisible();
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "isNodeCollapsed"
bool GTUtilsDashboard::isNodeCollapsed(GUITestOpStatus &os, const QString &nodeId) {
    const HIWebElement nodeSpanElement = getNodeSpan(os, nodeId);
    return nodeSpanElement.attribute(TITLE, "") == COLLAPSED_NODE_TITLE;
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "collapseNode"
void GTUtilsDashboard::collapseNode(GUITestOpStatus &os, const QString &nodeId) {
    GT_CHECK(isNodeVisible(os, nodeId),
             QString("SPAN of the node with ID '%1' is not visible. Some of the parent nodes are collapsed?").arg(nodeId));

    GT_CHECK(!isNodeCollapsed(os, nodeId),
             QString("UL of the node with ID '%1' is not visible. It is already collapsed.").arg(nodeId));

    click(os, getNodeSpan(os, nodeId));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "expandNode"
void GTUtilsDashboard::expandNode(GUITestOpStatus &os, const QString &nodeId) {
    GT_CHECK(isNodeVisible(os, nodeId),
             QString("SPAN of the node with ID '%1' is not visible. Some of the parent nodes are collapsed?").arg(nodeId));

    GT_CHECK(isNodeCollapsed(os, nodeId),
             QString("UL of the node with ID '%1' is visible. It is already expanded.").arg(nodeId));

    click(os, getNodeSpan(os, nodeId));
}
#undef GT_METHOD_NAME

#define GT_METHOD_NAME "getLogUrlFromNode"
QString GTUtilsDashboard::getLogUrlFromNode(GUITestOpStatus &os, const QString &outputNodeId) {
    const QString logFileLinkSelector = QString("SPAN#%1 A").arg(getNodeSpanId(outputNodeId));
    return getLogUrlFromElement(os, GTWebView::findElementBySelector(os, getDashboardWebView(os), logFileLinkSelector));
}
#undef GT_METHOD_NAME

#undef GT_CLASS_NAME

}    // namespace U2

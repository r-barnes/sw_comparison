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

#include <QDesktopServices>
#include <QMainWindow>
#include <QMessageBox>
#include <QUrl>

#include <U2Core/AppContext.h>
#include <U2Core/Counter.h>
#include <U2Core/IdRegistry.h>
#include <U2Core/L10n.h>
#include <U2Core/ProjectModel.h>
#include <U2Core/Task.h>
#include <U2Core/U2SafePoints.h>

#include <U2Gui/MainWindow.h>
#include <U2Gui/WelcomePageAction.h>

#include "WelcomePageJsAgent.h"

namespace U2 {

WelcomePageJsAgent::WelcomePageJsAgent(QObject *parent)
    : JavaScriptAgent(parent)
{

}

void WelcomePageJsAgent::message(const QString &message) {
    coreLog.error(message);
}

void WelcomePageJsAgent::performAction(const QString &actionId) {
    GRUNTIME_NAMED_COUNTER(cvar, tvar, "Welcome Page: " + actionId, "");
    IdRegistry<WelcomePageAction> *welcomePageActions = AppContext::getWelcomePageActionRegistry();
    SAFE_POINT(NULL != welcomePageActions, L10N::nullPointerError("Welcome Page Actions"), );

    WelcomePageAction *action = welcomePageActions->getById(actionId);
    if (NULL != action) {
        action->perform();
    } else if (BaseWelcomePageActions::CREATE_WORKFLOW == actionId) {
        QMessageBox::warning(AppContext::getMainWindow()->getQMainWindow(), L10N::warningTitle(),
            tr("The Workflow Designer plugin is not loaded. You can add it using the menu Settings -> Plugins. Then you need to restart UGENE."));
    } else {
        FAIL("Unknown welcome page action", );
    }
}

void WelcomePageJsAgent::openUrl(const QString &urlId) {
    GRUNTIME_NAMED_COUNTER(cvar, tvar, "Welcome Page: " + urlId, "");
    QString url = getUrlById(urlId);
    SAFE_POINT(!url.isEmpty(), "Unknown URL ID: " + urlId, );
    QDesktopServices::openUrl(QUrl(url));
}

void WelcomePageJsAgent::openFile(const QString &url) {
    GCOUNTER(cvar, tvar, "Welcome Page: recent files");
    QList<GUrl> urls;
    urls << url;
    Task *t = AppContext::getProjectLoader()->openWithProjectTask(urls);
    CHECK(NULL != t, );
    AppContext::getTaskScheduler()->registerTopLevelTask(t);
}

QString WelcomePageJsAgent::getUrlById(const QString &urlId) {
    if ("facebook" == urlId) {
        return "https://www.facebook.com/groups/ugene";
    }
    if ("twitter" == urlId) {
        return "https://twitter.com/uniprougene";
    }
    if ("linkedin" == urlId) {
        return "https://www.linkedin.com/profile/view?id=200543736";
    }
    if ("youtube" == urlId) {
        return "http://www.youtube.com/user/UniproUGENE";
    }
    if ("vkontakte" == urlId) {
        return "http://vk.com/uniprougene";
    }
    if ("rss" == urlId) {
        return "http://feeds2.feedburner.com/NewsOfUgeneProject";
    }
    if ("quick_start" == urlId) {
        return "https://ugene.net/wiki/display/QSG/Quick+Start+Guide";
    }
    return "";
}

}   // namespace U2

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
 
function addRecentItem(containerId, item, link) {
    var container = document.getElementById(containerId);
    var stringToAdd;
    if(link === "") {
        stringToAdd = item;
    } else {
        stringToAdd = "<a class=\"recentLink\" href=\"#\" onclick=\"ugene.openFile('" + item +
        "')\" title=\"" + item + "\">- " + link + "</a>";
    }
    container.insertAdjacentHTML('beforeend', stringToAdd);
    updateLinksVisibility();
}

function clearRecent(containerId) {
    var container = document.getElementById(containerId);
    container.innerHTML = '';
    updateLinksVisibility();
}

function isVisible(el, p) {
    var elRect = el.getBoundingClientRect();
    var pRect = p.getBoundingClientRect();
    return elRect.bottom <= pRect.bottom;
}

function updateVisibility(containerId) {
    var container = document.getElementById(containerId)
    var children = document.querySelectorAll("#" + containerId + " .recentLink");
    for (var i=children.length-1; i>=0; i--) {
        var child = children[i]
        if (child.className === "recentLink") {
            if (isVisible(child, container)) {
                child.style.color = ""
            } else {
                child.style.color = "transparent"
            }
        }
    }
}

function updateLinksVisibility() {
    updateVisibility("recentFilesBlock")
    updateVisibility("recentProjectsBlock")
}

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

#include <math.h>

#include "StrPackUtils.h"

namespace U2 {

const QBitArray StrPackUtils::charactersToEscape = StrPackUtils::initCharactersToEscape();
const QString StrPackUtils::LIST_SEPARATOR = ",";
const QString StrPackUtils::MAP_SEPARATOR = ";";
const QString StrPackUtils::PAIR_CONNECTOR = "=";
const QRegExp StrPackUtils::listSeparatorRegExp(QString("^\\\"|(?!\\\\)\\\"%1\\\"|\\\"$").arg(LIST_SEPARATOR));
const QRegExp StrPackUtils::mapSeparatorRegExp(QString("(?!\\\\)\\\"%1\\\"").arg(MAP_SEPARATOR));
const QRegExp StrPackUtils::pairSeparatorRegExp(QString("^\\\"|(?!\\\\)\\\"%1\\\"|\\\"$").arg(PAIR_CONNECTOR));

QString StrPackUtils::packStringList(const QStringList &list) {
    QString packedList;
    foreach (const QString &string, list) {
        packedList += wrapString(escapeCharacters(string)) + LIST_SEPARATOR;
    }
    packedList.chop(LIST_SEPARATOR.size());
    return packedList;
}

QStringList StrPackUtils::unpackStringList(const QString &string) {
    QStringList unpackedList;
    foreach (const QString &escapedString, string.split(listSeparatorRegExp, QString::SkipEmptyParts)) {
        unpackedList << unescapeCharacters(escapedString);
    }
    return unpackedList;
}

QString StrPackUtils::packMap(const StrStrMap &map) {
    QString string;
    foreach (const QString &key, map.keys()) {
        string += wrapString(escapeCharacters(key)) + PAIR_CONNECTOR + wrapString(escapeCharacters(map[key])) + MAP_SEPARATOR;
    }
    string.chop(MAP_SEPARATOR.size());
    return string;
}

StrStrMap StrPackUtils::unpackMap(const QString &string) {
    StrStrMap map;
    foreach (const QString &pair, string.split(mapSeparatorRegExp, QString::SkipEmptyParts)) {
        const QStringList splittedPair = pair.split(pairSeparatorRegExp, QString::SkipEmptyParts);
        Q_ASSERT(splittedPair.size() <= 2);
        map.insert(splittedPair.first(), splittedPair.size() > 1 ? splittedPair[1] : "");
    }
    return map;
}

QBitArray StrPackUtils::initCharactersToEscape() {
    QBitArray map(pow(2, 8 * sizeof(char)));
    map[(int)'\\'] = true;
    map[(int)'\"'] = true;
    return map;
}

QString StrPackUtils::escapeCharacters(QString string) {
    for (int i = 0; i < charactersToEscape.size(); i++) {
        if (charactersToEscape[i]) {
            const char c = (char)i;
            string.replace(c, QString("\\") + c);
        }
    }

    return string;
}

QString StrPackUtils::unescapeCharacters(QString string) {
    for (int i = 0; i < charactersToEscape.size(); i++) {
        if (charactersToEscape[i]) {
            const char c = (char)i;
            string.replace(QString("\\") + c, QString(1, c));
        }
    }

    return string;
}

QString StrPackUtils::wrapString(const QString &string) {
    return "\"" + string + "\"";
}

}   // namespace U2

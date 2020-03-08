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

#include "MSAEditorSequenceArea.h"
#include "MsaEditorStatusBar.h"

#include <U2Core/DNAAlphabet.h>
#include <U2Core/MSAUtils.h>
#include <U2Core/MultipleAlignmentObject.h>

#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLineEdit>
#include <QPushButton>

namespace U2 {

MsaEditorStatusBar::MsaEditorStatusBar(MultipleAlignmentObject* mobj, MaEditorSequenceArea* seqArea)
    : MaEditorStatusBar(mobj, seqArea) {
    setObjectName("msa_editor_status_bar");

    lineLabel->setPatterns(tr("Seq %1 / %2"), tr("Sequence %1 of %2"));

    connect(mobj, SIGNAL(si_alphabetChanged(const MaModificationInfo&, const DNAAlphabet *)), SLOT(sl_alphabetChanged()));

    prevButton = new QPushButton();
    prevButton->setObjectName("Find backward");
    prevButton->setToolTip(tr("Find backward <b>(SHIFT + Enter)</b>"));
    prevButton->setIcon(QIcon(":core/images/msa_find_prev.png"));
    prevButton->setFlat(true);
    nextButton = new QPushButton();
    nextButton->setObjectName("Find forward");
    nextButton->setToolTip(tr("Find forward <b>(Enter)</b>"));
    nextButton->setIcon(QIcon(":core/images/msa_find_next.png"));
    nextButton->setFlat(true);

    connect(prevButton, SIGNAL(clicked()), SLOT(sl_findPrev()));
    connect(nextButton, SIGNAL(clicked()), SLOT(sl_findNext()));

    findLabel = new QLabel();
    findLabel->setText(tr("Find:"));

    searchEdit = new QLineEdit();
    searchEdit->setObjectName("searchEdit");
    searchEdit->installEventFilter(this);
    searchEdit->setMaxLength(1000);
    validator = new MaSearchValidator(mobj->getAlphabet(), this);
    searchEdit->setValidator(validator);
    findLabel->setBuddy(searchEdit);

    findAction = new QAction(tr("Find in alignment"), this);//this action is used only to enable shortcut to change focus today
    findAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_F));
    findAction->setShortcutContext(Qt::WindowShortcut);
    connect(findAction, SIGNAL(triggered()), SLOT(sl_findFocus()));
    addAction(findAction);

    updateLabels();
    setupLayout();
}

bool MsaEditorStatusBar::eventFilter(QObject*, QEvent* ev) {
    if (ev->type() == QEvent::KeyPress) {
        QKeyEvent* kev = (QKeyEvent*)ev;
        if (kev->key() == Qt::Key_Enter || kev->key() == Qt::Key_Return) {
            if (kev->modifiers() == Qt::SHIFT) {
                prevButton->click();
            } else {
                nextButton->click();
            }
        } else if (kev->key() == Qt::Key_Escape) {
            seqArea->setFocus();
        }
    }
    return false;
}

void MsaEditorStatusBar::sl_alphabetChanged(){
    if (!aliObj->getAlphabet()->isRaw()){
        QByteArray alphabetChars = aliObj->getAlphabet()->getAlphabetChars(true);
        //remove special characters
        alphabetChars.remove(alphabetChars.indexOf('*'), 1);
        alphabetChars.remove(alphabetChars.indexOf('-'), 1);
        validator->setRegExp(QRegExp(QString("[%1]+").arg(alphabetChars.constData())));
    }else{
        validator->setRegExp(QRegExp(".*"));
    }

    //check is pattern clean required
    QString currentPattern = QString(searchEdit->text());
    int pos = 0;
    if(validator->validate(currentPattern, pos) != QValidator::Acceptable){
        searchEdit->clear();
    }
}

void MsaEditorStatusBar::sl_findNext( ) {
    if (seqArea->isAlignmentEmpty()) {
        return;
    }

    QByteArray pat = searchEdit->text( ).toLocal8Bit( );
    if ( pat.isEmpty( ) ) {
        return;
    }
    const MultipleAlignment msa = aliObj->getMultipleAlignment();
    if ( !msa->getAlphabet( )->isCaseSensitive( ) ) {
        pat = pat.toUpper( );
    }
    const int aliLen = msa->getLength( );
    const int nSeq = seqArea->getNumDisplayableSequences( );
    QPoint selectionTopLeft = seqArea->getSelection( ).topLeft( );

    if ( selectionTopLeft == lastSearchPos ) {
        selectionTopLeft.setX( selectionTopLeft.x( ) + 1 );
    }
    for (int s = selectionTopLeft.y(); s < nSeq; s++) {
        const int rowIndex = seqArea->getEditor()->getUI()->getCollapseModel()->mapToRow(s);
        const MultipleAlignmentRow row = msa->getRow(rowIndex);
        // if s == pos.y -> search from the current base, otherwise search from the seq start
        int p = ( s == selectionTopLeft.y( ) ) ? selectionTopLeft.x( ) : 0;
        for ( ; p < ( aliLen - pat.length( ) + 1 ); p++ ) {
            char c = row->charAt( p );
            int selLength = 0;
            if ( U2Msa::GAP_CHAR != c && MSAUtils::equalsIgnoreGaps(row, p, pat, selLength) ) {
                // select the result now
                MaEditorSelection sel( p, s, selLength, 1 );
                seqArea->setSelection(sel, true);
                seqArea->centerPos(sel.topLeft());
                lastSearchPos = seqArea->getSelection().topLeft();
                return;
            }
        }
    }
}

void MsaEditorStatusBar::sl_findPrev( ) {
    if (seqArea->isAlignmentEmpty()) {
        return;
    }

    QByteArray pat = searchEdit->text( ).toLocal8Bit( );
    if ( pat.isEmpty( ) ) {
        return;
    }
    const MultipleAlignment msa = aliObj->getMultipleAlignment();
    if ( !msa->getAlphabet( )->isCaseSensitive( ) ) {
        pat = pat.toUpper( );
    }
    int aliLen = msa->getLength( );
    QPoint pos = seqArea->getSelection( ).topLeft( );
    if ( pos == lastSearchPos ) {
        pos.setX( pos.x( ) - 1 );
    }
    for ( int s = pos.y( ); 0 <= s; s-- ) {
        const int rowIndex = seqArea->getEditor()->getUI()->getCollapseModel()->mapToRow(s);
        const MultipleAlignmentRow row = msa->getRow(rowIndex);
        //if s == pos.y -> search from the current base, otherwise search from the seq end
        int p = ( s == pos.y( ) ? pos.x( ) : ( aliLen - pat.length( ) + 1) );
        while ( 0 <= p ) {
            int selectionLength = 0;
            if ( U2Msa::GAP_CHAR != row->charAt( p )
                && MSAUtils::equalsIgnoreGaps( row, p, pat, selectionLength ) )
            {
                // select the result now
                MaEditorSelection sel( p, s, selectionLength, 1 );
                seqArea->setSelection( sel, true );
                seqArea->centerPos( sel.topLeft( ) );
                lastSearchPos = seqArea->getSelection( ).topLeft( );
                return;
            }
            p--;
        }
    }
}

void MsaEditorStatusBar::sl_findFocus() {
    searchEdit->setFocus();
}

void MsaEditorStatusBar::setupLayout() {
    layout->addWidget(findLabel);
    layout->addWidget(prevButton);
    layout->addWidget(searchEdit);
    layout->addWidget(nextButton);

    layout->addWidget(lineLabel);
    layout->addWidget(colomnLabel);
    layout->addWidget(positionLabel);
    layout->addWidget(selectionLabel);

    layout->addWidget(lockLabel);
}

void MsaEditorStatusBar::updateLabels() {
    updateLineLabel();
    updatePositionLabel();
    updateColumnLabel();
    updateSelectionLabel();
}

MaSearchValidator::MaSearchValidator(const DNAAlphabet* alphabet, QObject *parent)
: QRegExpValidator(parent)
{
    if (!alphabet->isRaw()){
        QByteArray alphabetChars = alphabet->getAlphabetChars(true);
        //remove special characters
        alphabetChars.remove(alphabetChars.indexOf('*'), 1);
        alphabetChars.remove(alphabetChars.indexOf('-'), 1);
        setRegExp(QRegExp(QString("[%1]+").arg(alphabetChars.constData())));
    }
}

QValidator::State MaSearchValidator::validate(QString &input, int &pos) const {
    input = input.simplified();
    input = input.toUpper();
    input.remove(" ");
    input.remove("-"); // Gaps are not used in search model
    return QRegExpValidator::validate(input, pos);
}

} // namespace

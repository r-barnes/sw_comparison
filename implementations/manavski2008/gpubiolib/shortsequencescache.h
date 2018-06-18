#ifndef SHORTSEQUENCESCACHE_H
#define SHORTSEQUENCESCACHE_H

/**
	@author Svetlin Manavski <svetlin.a@manavski.com>
*/

#include "alignmentpair.h"

#include <QMutex>
#include <QHash>


class ShortSequencesCache {
public:
    ShortSequencesCache();
    ~ShortSequencesCache();

	EstResultSummary getQueryResults(const BioSequence &);
	void addQueryGood(const BioSequence &, const EstResultSummary &);
	void addQueryBad(const BioSequence &);
	bool isQueryBad(const BioSequence &);
	string getQueryBadnName(const BioSequence &);

	unsigned getSizeBad();
	unsigned getSizeGood();
private:
    ShortSequencesCache(const ShortSequencesCache &);

	QHash<QString, EstResultSummary> goodQueries; // associate the query to its previous results
	QMutex mtx_goodQueries;

	QHash<QString, string> badQueries; //  associate the query to its name
	QMutex mtx_badQueries;

};

#endif


#include "shortsequencescache.h"

#include <QMutexLocker>

ShortSequencesCache::ShortSequencesCache()
{
}


ShortSequencesCache::~ShortSequencesCache()
{
}

EstResultSummary ShortSequencesCache::getQueryResults(const BioSequence &q)
{
	QMutexLocker locker(&mtx_goodQueries);

	string seq = q.getSequence();

	const EstResultSummary r = goodQueries.value(seq.c_str());

	return r;
}

void ShortSequencesCache::addQueryGood(const BioSequence &q, const EstResultSummary &res)
{
	QMutexLocker locker(&mtx_goodQueries);

	string seq = q.getSequence();
	
	goodQueries.insert(seq.c_str(), res);  //removes previous value if any
}

void ShortSequencesCache::addQueryBad(const BioSequence &q)
{
	QMutexLocker locker(&mtx_badQueries);

	string seq = q.getSequence();
	
	badQueries.insert(seq.c_str(), q.getName()); //removes previous value if any
}

bool ShortSequencesCache::isQueryBad(const BioSequence &q)
{
	QMutexLocker locker(&mtx_badQueries);

	string seq = q.getSequence();

	return badQueries.contains(seq.c_str());
}

string ShortSequencesCache::getQueryBadnName(const BioSequence &q)
{
	QMutexLocker locker(&mtx_badQueries);

	string seq = q.getSequence();

	return badQueries.value(seq.c_str());
}



unsigned ShortSequencesCache::getSizeBad()
{
	QMutexLocker locker(&mtx_badQueries);

	return badQueries.size();
}

unsigned ShortSequencesCache::getSizeGood()
{
	QMutexLocker locker(&mtx_goodQueries);

	return goodQueries.size();
}


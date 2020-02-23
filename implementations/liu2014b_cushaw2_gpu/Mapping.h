/*
 * Mapping.h
 *
 *  Created on: Jan 18, 2012
 *      Author: yongchao
 */

#ifndef MAPPING_H_
#define MAPPING_H_
#include "Macros.h"

struct Mapping
{
	Mapping()
	{
		_align = NULL;
	}
	Mapping(const Mapping& mapping)
	{
		_align = NULL;
		if (mapping._align)
		{
			_align = mapping._align;
		}
		_position = mapping._position;
		_gposition = mapping._gposition;
		_genomeIndex = mapping._genomeIndex;
		_strand = mapping._strand;
		_mapQual = mapping._mapQual;
	}
	Mapping(CigarAlign* align, int64_t position, int64_t gposition, int strand, int genomeIndex, int mapQual)
	{
		_align = NULL;

		init(align, position, gposition, strand, genomeIndex, mapQual);
	}
	~Mapping()
	{
		if (_align)
		{
			delete _align;
		}
	}
	inline void init(CigarAlign* align, int64_t position, int64_t gposition, int strand,
			int genomeIndex, int mapQual)
	{
		if (_align)
		{
			delete _align;
		}
		_align = align;
		_position = position;
		_gposition = gposition;
		_genomeIndex = genomeIndex;
		_strand = strand;
		_mapQual = mapQual;
	}

	/*member variables*/
	CigarAlign* _align;
	int64_t _position;
	int64_t _gposition;
	int _genomeIndex;
	uint32_t _strand : 1;
	uint32_t _mapQual : 31;
};

#endif /* MAPPING_H_ */

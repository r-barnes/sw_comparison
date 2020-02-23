/*
 * Seed.h
 *
 *  Created on: Jan 26, 2012
 *      Author: yongchao
 */

#ifndef SEED_H_
#define SEED_H_
#include "Macros.h"

struct Seed
{
	uint32_t _targetPosition;
	uint32_t _strand :1;
	uint32_t _queryPosition :31;
	uint32_t _best :1;
	uint32_t _seedLength :31;
};

typedef struct
{
	int32_t _score;
	uint32_t _seedLength : 16;
	uint32_t _seedPosition : 16;
	int32_t _seedIndex;
	uint32_t _targetPosition;
}AlignScore;

bool operator>(const AlignScore& p, const AlignScore& q);
bool operator<(const AlignScore& p, const AlignScore& q);
bool operator==(const AlignScore& p, const AlignScore& q);
#endif /* SEED_H_ */

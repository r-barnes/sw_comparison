/*
 * Structs.h
 *
 *  Created on: Dec 27, 2011
 *      Author: yongchao
 */

#ifndef STRUCTS_H_
#define STRUCTS_H_

#ifndef __CUDACC__

/*structure for int2*/
struct int2
{
	int32_t x;
	int32_t y;
};
#define make_int2(x, y) (int2){x, y}

/*structure for uint2*/
struct uint2
{
	uint32_t x;
	uint32_t y;
};
#define make_uint2(x, y) (uint2){x, y}

/*structure for uint3*/
struct uint3
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
};
#define make_uint3(x, y, z) (uint3){x, y, z}

/*structure for uint4*/
struct uint4
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
	uint32_t w;
};
#define make_uint4(x, y, z, w) (uint4){x, y, z, w}


#include "Seed.h"
#endif

#endif /* STRUCTS_H_ */

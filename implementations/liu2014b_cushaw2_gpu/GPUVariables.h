/*
 * GPUVariables.h
 *
 *  Created on: Jan 17, 2013
 *      Author: yongchao
 */

#ifndef GPUVARIABLES_H_
#define GPUVARIABLES_H_

#include "BwtMacros.h"

#include <cstdint>

/*BWT data*/
__constant__ uint32_t _cudaBwtWidthShift;
__constant__ uint32_t _cudaBwtWidthMask;
__constant__ uint32_t _cudaBwtDollar;
__constant__ uint32_t _cudaBwtSeqLength;
__constant__ uint32_t _cudaBwtCCounts[BWT_NUM_OCC];
texture<uint32_t, 2, cudaReadModeElementType> _texBWT;


/*reads and hash tables*/
texture <uint32_t, 1, cudaReadModeElementType>_texPacReads;	/*packed reads*/
texture<int4, 1, cudaReadModeElementType> _texHash; /*hash table for read batches*/

/*suffix array interval calculation*/
__constant__ uint32_t _cudaMaxSeedOcc; /*the maximal number of seed occurrence*/
__constant__ uint32_t _cudaMaxReadLength;	/*maximum read length in the batch*/

/*determine mapping positions*/
__constant__ uint32_t _cudaSaFactor;

__constant__ uint32_t _cudaGenomeWidthShift;
__constant__ uint32_t _cudaGenomeWidthMask;
texture<uint8_t, 2, cudaReadModeElementType> _texPacGenome;

__constant__ uint32_t _cudaNumGenomicSeqs;
texture<uint32_t, 1, cudaReadModeElementType> _texBwtAnns;

/*for Smith-Waterman*/
__constant__ int32_t _cudaGapOE; /*sum of gap open and extension penalites*/
__constant__ int32_t _cudaGapExtend; /*sum of gap extend*/
__constant__ int32_t _cudaMatchScore; /*score for a match*/
__constant__ int32_t _cudaMismatchScore; /*penalty for a mismatch*/

texture<uint32_t, 1, cudaReadModeElementType> _texDevReadOccs;
texture<uint2, 1, cudaReadModeElementType> _texDevSeeds;
texture<uint32_t, 1, cudaReadModeElementType> _texDevReadIndices;


/*constraints*/
__constant__ uint32_t _cudaMinAlignScore;
__constant__ float _cudaMinID;
__constant__ float _cudaMinBaseRatio;
#endif /* GPUVARIABLES_H_ */

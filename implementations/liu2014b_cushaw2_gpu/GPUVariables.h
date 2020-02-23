/*
 * GPUVariables.h
 *
 *  Created on: Jan 17, 2013
 *      Author: yongchao
 */

#ifndef GPUVARIABLES_H_
#define GPUVARIABLES_H_

/*BWT data*/
extern __constant__ uint32_t _cudaBwtWidthShift;
extern __constant__ uint32_t _cudaBwtWidthMask;
extern __constant__ uint32_t _cudaBwtDollar;
extern __constant__ uint32_t _cudaBwtSeqLength;
extern __constant__ uint32_t _cudaBwtCCounts[BWT_NUM_OCC];
extern texture<uint32_t, 2, cudaReadModeElementType> _texBWT;


/*reads and hash tables*/
extern texture <uint32_t, 1, cudaReadModeElementType>_texPacReads;	/*packed reads*/
extern texture<int4, 1, cudaReadModeElementType> _texHash; /*hash table for read batches*/

/*suffix array interval calculation*/
extern __constant__ uint32_t _cudaMaxSeedOcc; /*the maximal number of seed occurrence*/
extern __constant__ uint32_t _cudaMaxReadLength;	/*maximum read length in the batch*/

/*determine mapping positions*/
extern __constant__ uint32_t _cudaSaFactor;

extern __constant__ uint32_t _cudaGenomeWidthShift;
extern __constant__ uint32_t _cudaGenomeWidthMask;
extern texture<uint8_t, 2, cudaReadModeElementType> _texPacGenome;

extern __constant__ uint32_t _cudaNumGenomicSeqs;
extern texture<uint32_t, 1, cudaReadModeElementType> _texBwtAnns;

/*for Smith-Waterman*/
extern __constant__ int32_t _cudaGapOE; /*sum of gap open and extension penalites*/
extern __constant__ int32_t _cudaGapExtend; /*sum of gap extend*/
extern __constant__ int32_t _cudaMatchScore; /*score for a match*/
extern __constant__ int32_t _cudaMismatchScore; /*penalty for a mismatch*/

extern texture<uint32_t, 1, cudaReadModeElementType> _texDevReadOccs;
extern texture<uint2, 1, cudaReadModeElementType> _texDevSeeds;
extern texture<uint32_t, 1, cudaReadModeElementType> _texDevReadIndices;


/*constraints*/
extern __constant__ uint32_t _cudaMinAlignScore;
extern __constant__ float _cudaMinID;
extern __constant__ float _cudaMinBaseRatio;
#endif /* GPUVARIABLES_H_ */

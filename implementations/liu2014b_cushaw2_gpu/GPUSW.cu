/*
 * GPUSW.cu
 *
 *  Created on: Jan 11, 2013
 *      Author: yongchao
 */
#include "GPUSW.h"
#include "GPUSeeds.h"
#include "GPUBWT.h"
#include "GPUVariables.h"

/*get an element from the genome*/
static __device__           inline uint8_t _cudaGetGenomeElement(uint32_t position) {
	return tex2D(_texPacGenome, position & _cudaGenomeWidthMask,
			position >> _cudaGenomeWidthShift);
}
/*get a single base from the read*/
static __device__            inline uint32_t SW_TB_GET_READ_BASE(uint32_t rposition,
		uint32_t index) {
	return (tex1Dfetch(_texPacReads, rposition + (index >> 2))
			>> ((index & 3) << 3)) & 0x0ff;
}
/*get a single base from the genome*/
static __device__            inline uint32_t SW_TB_GET_GENOME_BASE(uint32_t gposition,
		uint32_t index) {
	return (_cudaGetGenomeElement(gposition + (index >> 2))
			>> ((3 - (index & 3)) << 1)) & 3;
}

/*calculate the substitution score*/
#define DEV_GET_SUB_SCORE(score, rbase, gbase, dummy) \
	dummy = (rbase == DUMMY_BASE) || (gbase == DUMMY_BASE);	\
	score = (rbase == gbase) ? _cudaMatchScore : _cudaMismatchScore;	\
	score = dummy ? 0 : score;

/*core function for score-only Smith Waterman*/
static __device__ int32_t _devSWAlignScore(uint32_t rposition, uint32_t rlength,
		uint32_t gposition, uint32_t glength) {
	int32_t i, j, k;
	int32_t maxHH, e;
	int32_t rbase, gbase, dummy, subScore;
	uint32_t rpac;
	uint8_t gpac, gpac2;
	int4 f, h, p; //the first 4 rows
	int4 f0, h0, p0; //the second 4 rows

	ushort2 HD;
	ushort2 initHD = make_ushort2(0, 0);
	int4 zero = make_int4(0, 0, 0, 0);
	ushort2 global[MAX_SEQ_LENGTH];
	for (i = 0; i < rlength; i++) {
		global[i] = initHD;
	}

	maxHH = 0;
	glength >>= 3;
	for (i = 0; i < glength; ++i) { /*genome in rows*/
		h = p = zero;
		f = zero;

		h0 = p0 = zero;
		f0 = zero;

		/*read 4 bases*/
		gpac = _cudaGetGenomeElement(gposition++);

		/*read another 4 bases*/
		gpac2 = _cudaGetGenomeElement(gposition++);

		/*the inner loops*/
		for (j = 0; j < rlength; j += 4) { /*read in columns*/
			//load the packed 4 residues
			rpac = tex1Dfetch(_texPacReads, rposition + (j >> 2));

			//compute the cell block SEQ_LENGTH_ALIGNED x 4
			for (k = 0; k < 4; k++) {
				//get the (j + k)-th residue
				rbase = (rpac >> (k << 3)) & 0x0ff;

				//load data
				HD = global[j + k];

				/*get a base from read*/
				gbase = (gpac >> 6) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				//compute the cell (0, 0);
				f.x = max(h.x - _cudaGapOE, f.x - _cudaGapExtend);
				e = max(HD.x - _cudaGapOE, HD.y - _cudaGapExtend);

				h.x = p.x + subScore;
				h.x = max(h.x, f.x);
				h.x = max(h.x, e);
				h.x = max(h.x, 0);
				maxHH = max(maxHH, h.x);
				p.x = HD.x;

				//compute cell (0, 1)
				gbase = (gpac >> 4) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.y = max(h.y - _cudaGapOE, f.y - _cudaGapExtend);
				e = max(h.x - _cudaGapOE, e - _cudaGapExtend);

				h.y = p.y + subScore;
				h.y = max(h.y, f.y);
				h.y = max(h.y, e);
				h.y = max(h.y, 0);
				maxHH = max(maxHH, h.y);
				p.y = h.x;

				//compute cell (0, 2);
				gbase = (gpac >> 2) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.w = max(h.w - _cudaGapOE, f.w - _cudaGapExtend);
				e = max(h.y - _cudaGapOE, e - _cudaGapExtend);

				h.w = p.w + subScore;
				h.w = max(h.w, f.w);
				h.w = max(h.w, e);
				h.w = max(h.w, 0);
				maxHH = max(maxHH, h.w);
				p.w = h.y;

				//compute cell (0, 3)
				gbase = gpac & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.z = max(h.z - _cudaGapOE, f.z - _cudaGapExtend);
				e = max(h.w - _cudaGapOE, e - _cudaGapExtend);

				h.z = p.z + subScore;
				h.z = max(h.z, f.z);
				h.z = max(h.z, e);
				h.z = max(h.z, 0);
				maxHH = max(maxHH, h.z);
				p.z = h.w;

				//compute cell(0, 4)
				gbase = (gpac2 >> 6) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.x = max(h0.x - _cudaGapOE, f0.x - _cudaGapExtend);
				e = max(h.z - _cudaGapOE, e - _cudaGapExtend);

				h0.x = p0.x + subScore;
				h0.x = max(h0.x, f0.x);
				h0.x = max(h0.x, e);
				h0.x = max(h0.x, 0);
				maxHH = max(maxHH, h0.x);
				p0.x = h.z;

				//compute cell(0, 5)
				gbase = (gpac2 >> 4) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.y = max(h0.y - _cudaGapOE, f0.y - _cudaGapExtend);
				e = max(h0.x - _cudaGapOE, e - _cudaGapExtend);

				h0.y = p0.y + subScore;
				h0.y = max(h0.y, f0.y);
				h0.y = max(h0.y, e);
				h0.y = max(h0.y, 0);
				maxHH = max(maxHH, h0.y);
				p0.y = h0.x;

				//compute cell (0, 6)
				gbase = (gpac2 >> 2) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.w = max(h0.w - _cudaGapOE, f0.w - _cudaGapExtend);
				e = max(h0.y - _cudaGapOE, e - _cudaGapExtend);

				h0.w = p0.w + subScore;
				h0.w = max(h0.w, f0.w);
				h0.w = max(h0.w, e);
				h0.w = max(h0.w, 0);
				maxHH = max(maxHH, h0.w);
				p0.w = h0.y;

				//compute cell(0, 7)
				gbase = gpac2 & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.z = max(h0.z - _cudaGapOE, f0.z - _cudaGapExtend);
				e = max(h0.w - _cudaGapOE, e - _cudaGapExtend);

				h0.z = p0.z + subScore;
				h0.z = max(h0.z, f0.z);
				h0.z = max(h0.z, e);
				h0.z = max(h0.z, 0);
				maxHH = max(maxHH, h0.z);
				p0.z = h0.w;

				/*save the values*/
				HD.x = h0.z;
				HD.y = max(e, 0);
				global[j + k] = HD;
			}
		}
	}
	return maxHH;
}

static __global__ void kernelSWAlignScore(uint32_t* cudaAlignScores,
		size_t numSeeds) {
	int32_t score;
	int64_t lower, upper;
	int4 hash;
	uint2 seed;
	uint32_t seedLength, seedPosition, strand, readIndex;
	uint32_t gposition, glength;
	int32_t gid = threadIdx.x + blockIdx.x * blockDim.x;

	if (gid >= numSeeds) {
		return;
	}

	/*get the seed*/
	seed = tex1Dfetch(_texDevSeeds, gid);
	seedLength = GET_SEED_LENGTH(seed.y);
	seedPosition = GET_SEED_POS(seed.y);
	strand = GET_SEED_STRAND(seed.y);

	/*get the read information*/
	readIndex = cudaAlignScores[gid]; /*at this point, this array stores only read indices*/
	hash = tex1Dfetch(_texHash, readIndex);

	/*get the genome region*/
	lower = seed.x;
	lower -= 2 * (seedPosition + 1);
	lower = llmax(lower, 0);
	lower = (lower >> 3) << 3; /*algined to 8*/

	upper = seed.x;
	upper += 2 * (hash.y - seedPosition) - seedLength;
	upper = ((upper + 7) >> 3) << 3; /*aligned to 8*/
	upper = llmin(upper, _cudaBwtSeqLength);
	upper = (upper >> 3) << 3; /*in case that _cudaBwtSeqLength is not multiples of 8*/

	/*compute the score*/
	gposition = lower >> 2;
	glength = llmax(upper - lower, 0);
	score = _devSWAlignScore(hash.x + (strand ? hash.z >> 2 : 0), hash.z,
			gposition, glength);

	/*printf("gid: %d seedPosition %d seedLength %d lower %ld upper %ld strand %d score %d hash.x %d hash.z %d\n", gid, seedPosition, seedLength, lower, upper, strand, score, hash.x, hash.z);*/
	/*save the score*/
	cudaAlignScores[gid] = make_align_score(cudaAlignScores[gid], score);
}

void GPUSW::initAlignScore(uint2*devSeeds, size_t numSeeds) {
	cudaChannelFormatDesc channelDes = cudaCreateChannelDesc<uint2>();

	/*bind texture memory*/
	_texDevSeeds.addressMode[0] = cudaAddressModeClamp;
	_texDevSeeds.filterMode = cudaFilterModePoint;
	_texDevSeeds.normalized = false;

	cudaBindTexture(NULL, _texDevSeeds, devSeeds, channelDes,
			numSeeds * sizeof(uint2));
	myCheckCudaError;
}
void GPUSW::finalizeAlignScore() {

	cudaUnbindTexture(_texDevSeeds);
	myCheckCudaError;
}
void GPUSW::alignScore(uint32_t* devAlignScores, size_t numSeeds, int nthreads,
		cudaStream_t stream) {

	int32_t nblocks = (numSeeds + nthreads - 1) / nthreads;
	dim3 grid(nblocks, 1);
	dim3 blocks(nthreads, 1);

	kernelSWAlignScore<<<grid, blocks, 0, stream>>>(devAlignScores, numSeeds);
	myCheckCudaError;

	cudaDeviceSynchronize();
}

/*Smith-Waterman with traceback*/
static inline __device__ void SW_TB_SAVE_DIR(uint32_t& value, uint32_t dir,
		uint32_t cx, uint32_t cy) {
	uint32_t pos = (cy * 4 + cx) << 1;
	value |= (dir << pos);
}
static inline __device__ uint32_t SW_TB_GET_DIR(uint32_t *localTB,
		uint32_t localTBWidth, int32_t cx, int32_t cy,
		uint32_t& tile, int32_t& tilecx, int32_t& tilecy) {
	int32_t pos;
	int32_t trow = cy >> 2; /*row number of tiled tables*/
	int32_t tcol = cx >> 2; /*col number of tiled tables*/

	/*get the value of the tile*/
	if(trow != tilecy || tcol != tilecx){
		/*get the new tile value*/
		tile = *(localTB + trow * localTBWidth + tcol);
		/*save the tile coordinates*/
		tilecx = tcol;
		tilecy = trow;
	}

	/*position the value in a single tile*/
	cx &= 3;
	cy &= 3;
	pos = (cy * 4 + cx) << 1;

	return (tile >> pos) & 3;
}
static __device__ int2 _devSWAlign(uint32_t rposition, uint32_t rlength,
		uint32_t gposition, uint32_t glength, ushort2* global,
		uint32_t *localTB, uint32_t localTBWidth) {
	uint32_t i, j, k;
	int32_t maxHH, e;
	int32_t rbase, gbase, dummy, subScore;
	uint32_t rpac;
	uint8_t gpac, gpac2;
	int32_t cx, cy;
	int4 f, h, p; //the first 4 rows
	int4 f0, h0, p0; //the second 4 rows
	uint32_t pos, dirpac, dirpac2;
	uint32_t dir;
	int2 maxXY;

	ushort2 HD;
	ushort2 initHD = make_ushort2(0, 0);
	int4 zero = make_int4(0, 0, 0, 0);
	for (i = 0; i < rlength; i++) {
		global[i] = initHD;
	}

	maxHH = 0;
	maxXY = make_int2(-1, -1);

	glength >>= 3;
	for (i = 0; i < glength; ++i) { /*rows*/
		h = p = zero;
		f = zero;

		h0 = p0 = zero;
		f0 = zero;

		/*read 4 bases*/
		gpac = _cudaGetGenomeElement(gposition++);

		/*read another 4 bases*/
		gpac2 = _cudaGetGenomeElement(gposition++);

		/*coordinate*/
		cy = i << 3;

		/*the inner loops*/
		for (j = 0; j < rlength; j += 4) { /*reads in columns*/
			//load the packed 4 residues
			rpac = tex1Dfetch(_texPacReads, rposition + (j >> 2));

			/*initialize two tiles*/
			dirpac = dirpac2 = 0;

			/*base coordinate*/
			cx = j;

			//compute the cell block SEQ_LENGTH_ALIGNED x 4
			for (k = 0; k < 4; k++, ++cx) {

				//get the (j + k)-th residue
				rbase = (rpac >> (k << 3)) & 0x0ff;

				//load data
				HD = global[cx];

				/*get a base from read*/
				gbase = (gpac >> 6) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				//compute the cell (0, 0);
				f.x = max(h.x - _cudaGapOE, f.x - _cudaGapExtend);
				e = max(HD.x - _cudaGapOE, HD.y - _cudaGapExtend);

				h.x = p.x + subScore;
				h.x = max(h.x, f.x);
				h.x = max(h.x, e);
				h.x = max(h.x, 0);
				maxXY = (maxHH < h.x) ? make_int2(cx, cy) : maxXY;
				maxHH = max(maxHH, h.x);
				p.x = HD.x;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h.x == f.x) ? ALIGN_DIR_LEFT : dir;
				dir = (h.x == e) ? ALIGN_DIR_UP : dir;
				dir = (h.x == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac, dir, k, 0);

				//compute cell (0, 1)
				gbase = (gpac >> 4) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.y = max(h.y - _cudaGapOE, f.y - _cudaGapExtend);
				e = max(h.x - _cudaGapOE, e - _cudaGapExtend);

				h.y = p.y + subScore;
				h.y = max(h.y, f.y);
				h.y = max(h.y, e);
				h.y = max(h.y, 0);
				maxXY = (maxHH < h.y) ? make_int2(cx, cy + 1) : maxXY;
				maxHH = max(maxHH, h.y);
				p.y = h.x;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h.y == f.y) ? ALIGN_DIR_LEFT : dir;
				dir = (h.y == e) ? ALIGN_DIR_UP : dir;
				dir = (h.y == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac, dir, k, 1);

				//compute cell (0, 2);
				gbase = (gpac >> 2) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.w = max(h.w - _cudaGapOE, f.w - _cudaGapExtend);
				e = max(h.y - _cudaGapOE, e - _cudaGapExtend);

				h.w = p.w + subScore;
				h.w = max(h.w, f.w);
				h.w = max(h.w, e);
				h.w = max(h.w, 0);
				maxXY = (maxHH < h.w) ? make_int2(cx, cy + 2) : maxXY;
				maxHH = max(maxHH, h.w);
				p.w = h.y;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h.w == f.w) ? ALIGN_DIR_LEFT : dir;
				dir = (h.w == e) ? ALIGN_DIR_UP : dir;
				dir = (h.w == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac, dir, k, 2);

				//compute cell (0, 3)
				gbase = gpac & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f.z = max(h.z - _cudaGapOE, f.z - _cudaGapExtend);
				e = max(h.w - _cudaGapOE, e - _cudaGapExtend);

				h.z = p.z + subScore;
				h.z = max(h.z, f.z);
				h.z = max(h.z, e);
				h.z = max(h.z, 0);
				maxXY = (maxHH < h.z) ? make_int2(cx, cy + 3) : maxXY;
				maxHH = max(maxHH, h.z);
				p.z = h.w;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h.z == f.z) ? ALIGN_DIR_LEFT : dir;
				dir = (h.z == e) ? ALIGN_DIR_UP : dir;
				dir = (h.z == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac, dir, k, 3);

				//compute cell(0, 4)
				gbase = (gpac2 >> 6) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.x = max(h0.x - _cudaGapOE, f0.x - _cudaGapExtend);
				e = max(h.z - _cudaGapOE, e - _cudaGapExtend);

				h0.x = p0.x + subScore;
				h0.x = max(h0.x, f0.x);
				h0.x = max(h0.x, e);
				h0.x = max(h0.x, 0);
				maxXY = (maxHH < h0.x) ? make_int2(cx, cy + 4) : maxXY;
				maxHH = max(maxHH, h0.x);
				p0.x = h.z;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h0.x == f0.x) ? ALIGN_DIR_LEFT : dir;
				dir = (h0.x == e) ? ALIGN_DIR_UP : dir;
				dir = (h0.x == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac2, dir, k, 0);

				//compute cell(0, 5)
				gbase = (gpac2 >> 4) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.y = max(h0.y - _cudaGapOE, f0.y - _cudaGapExtend);
				e = max(h0.x - _cudaGapOE, e - _cudaGapExtend);

				h0.y = p0.y + subScore;
				h0.y = max(h0.y, f0.y);
				h0.y = max(h0.y, e);
				h0.y = max(h0.y, 0);
				maxXY = (maxHH < h0.y) ? make_int2(cx, cy + 5) : maxXY;
				maxHH = max(maxHH, h0.y);
				p0.y = h0.x;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h0.y == f0.y) ? ALIGN_DIR_LEFT : dir;
				dir = (h0.y == e) ? ALIGN_DIR_UP : dir;
				dir = (h0.y == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac2, dir, k, 1);

				//compute cell (0, 6)
				gbase = (gpac2 >> 2) & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.w = max(h0.w - _cudaGapOE, f0.w - _cudaGapExtend);
				e = max(h0.y - _cudaGapOE, e - _cudaGapExtend);

				h0.w = p0.w + subScore;
				h0.w = max(h0.w, f0.w);
				h0.w = max(h0.w, e);
				h0.w = max(h0.w, 0);
				maxXY = (maxHH < h0.w) ? make_int2(cx, cy + 6) : maxXY;
				maxHH = max(maxHH, h0.w);
				p0.w = h0.y;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h0.w == f0.w) ? ALIGN_DIR_LEFT : dir;
				dir = (h0.w == e) ? ALIGN_DIR_UP : dir;
				dir = (h0.w == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac2, dir, k, 2);

				//compute cell(0, 7)
				gbase = gpac2 & 3;
				DEV_GET_SUB_SCORE(subScore, rbase, gbase, dummy);

				f0.z = max(h0.z - _cudaGapOE, f0.z - _cudaGapExtend);
				e = max(h0.w - _cudaGapOE, e - _cudaGapExtend);

				h0.z = p0.z + subScore;
				h0.z = max(h0.z, f0.z);
				h0.z = max(h0.z, e);
				h0.z = max(h0.z, 0);
				maxXY = (maxHH < h0.z) ? make_int2(cx, cy + 7) : maxXY;
				maxHH = max(maxHH, h0.z);
				p0.z = h0.w;

				/*save the traceback*/
				dir = ALIGN_DIR_DIAGONAL;
				dir = (h0.z == f0.z) ? ALIGN_DIR_LEFT : dir;
				dir = (h0.z == e) ? ALIGN_DIR_UP : dir;
				dir = (h0.z == 0) ? ALIGN_DIR_STOP : dir;
				SW_TB_SAVE_DIR(dirpac2, dir, k, 3);

				/*save the values*/
				HD.x = h0.z;
				HD.y = max(e, 0);
				global[cx] = HD;
			}

			/*save the packed traceback*/
			pos = (cy >> 2) * localTBWidth + (j >> 2);
			localTB[pos] = dirpac;

			pos += localTBWidth; /*move to the next tile row*/
			localTB[pos] = dirpac2;
		}
	}
	//printf("tid: %d score %d\n", threadIdx.x, maxHH);
	return maxXY;
}

static __device__ float _devSWAlign2Cigar(uint16_t* cigars, uint32_t& numCigars,
		uint32_t* localTB, uint32_t localTBWidth, int2 endXY,
		uint32_t rposition, uint32_t gposition, int2 &startXY) {
	uint32_t dir;
	uint32_t numOps = 0;
	uint32_t lastOp = ALIGN_DIR_STOP;
	uint8_t op = ALIGN_OP_S;
	uint32_t c1, c2;
	uint32_t editDistance = 0;
	uint32_t alignLength = 0;
	bool isDiagnal;
	int cx = endXY.x, cy = endXY.y;
	uint32_t tile;
	int32_t tilecx = -1, tilecy = -1;

	numCigars = 0;
	startXY = endXY;
	while (1) {
		/*get the direction of the alignment*/
		dir = (cx < 0 || cy < 0) ?
				ALIGN_DIR_STOP : SW_TB_GET_DIR(localTB, localTBWidth, cx, cy, tile, tilecx, tilecy);

		/*check if save the operation*/
		isDiagnal = dir == ALIGN_DIR_DIAGONAL;
		if (lastOp == dir) {
			++numOps;
		} else {
			if (numOps) {
				cigars[numCigars++] = (numOps << 2) | op;
			}
			/*reset the values*/
			numOps = 1;
			lastOp = dir;
			op = isDiagnal ?
					ALIGN_OP_M :
					(dir == ALIGN_DIR_LEFT ? ALIGN_OP_I : ALIGN_OP_D);
		}
		if (dir == ALIGN_DIR_STOP) {
			break;
		}

		/*calculate the edit distances*/
		if (isDiagnal) {
			c1 = SW_TB_GET_GENOME_BASE(gposition, cy);
			c2 = SW_TB_GET_READ_BASE(rposition, cx);
			//printf("c1 %c c2 %c\n", "ACGTN"[c1], "ACGTN"[c2]);
		}
#if 0
		if (threadIdx.x == 0) {
			if (isDiagnal) {
				printf("%c %c\n", "ACGTN"[c1], "ACGTN"[c2]);
			} else if (dir == ALIGN_DIR_LEFT) {
				printf("- %c\n", "ACGTN"[c2]);
			} else if (dir == ALIGN_DIR_UP) {
				printf("%c -\n", "ACGTN"[c1]);
			}
		}
#endif
		editDistance += isDiagnal ? c1 != c2 : 1;
		++alignLength;

		/*adjust the coordinates*/
		startXY = make_int2(cx, cy); /*save the coordinates*/
		cx -= (isDiagnal || dir == ALIGN_DIR_LEFT);
		cy -= (isDiagnal || dir == ALIGN_DIR_UP);
	}
#if 0
	printf("tid: %d alignment length: %d edit distance: %d #cigars %d\n",
			threadIdx.x, alignLength, editDistance, numCigars);
#endif
	/*no local alignment*/
	if (alignLength == 0) {
		return 0;
	}

	/*calcualte percentage identity*/
	float pid = 100 * (alignLength - editDistance);
	pid = pid / alignLength;
	return pid;
}
static __device__ int _devGetGenomeIndex(uint32_t position) {
	uint32_t left, right, mid;

	left = 0;
	mid = 0;
	right = _cudaNumGenomicSeqs;
	while (left < right) {
		mid = (left + right) >> 1;
		if (position >= tex1Dfetch(_texBwtAnns, mid)) {
			if (mid == _cudaNumGenomicSeqs - 1)
				break;
			if (position < tex1Dfetch(_texBwtAnns, mid + 1))
				break;

			left = mid + 1;
		} else {
			right = mid;
		}
	}
	return mid;
}

#define MAX_SW_TB_SIZE		(MAX_SEQ_LENGTH >> 1)
static __global__ void kernelSWAlign(uint16_t* cudaCigars,
		uint32_t cudaCigarWidth, uint32_t maxNumCigars, uint8_t *cudaNumCigars,
		uint2* cudaAligns, float* cudaBasePortions, size_t numSeeds) {
	int64_t lower, upper;
	int4 hash;
	uint2 seed;
	uint32_t seedLength, seedPosition, strand, readIndex;
	ushort2 global[MAX_SEQ_LENGTH];
	uint32_t localTB[(MAX_SEQ_LENGTH >> 2) * MAX_SW_TB_SIZE], localTBWidth;
	int2 startXY, endXY;
	int32_t gid;
	uint32_t numCigars;
	uint16_t *inCigars, *outCigars, *cigars;
	uint32_t rposition, gposition, glength;
	uint32_t genomeIndex;
	float pid;

	/*get the global ID of the current thread*/
	gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= numSeeds) {
		return;
	}

	/*get the seed*/
	seed = tex1Dfetch(_texDevSeeds, gid);
	seedLength = GET_SEED_LENGTH(seed.y);
	seedPosition = GET_SEED_POS(seed.y);
	strand = GET_SEED_STRAND(seed.y);

	/*get the read information*/
	readIndex = GET_READ_INDEX(tex1Dfetch(_texDevReadIndices, gid)); /*At this point, read index and alignment score are packed together*/
	hash = tex1Dfetch(_texHash, readIndex);

	/*get the genome region*/
	lower = seed.x;
	lower -= 2 * (seedPosition + 1);
	lower = llmax(lower, 0);
	lower = (lower >> 3) << 3; /*algined to 8*/

	upper = seed.x;
	//upper += seedLength + 2 * (hash.y - seedPosition - seedLength);
	upper += 2 * (hash.y - seedPosition) - seedLength;
	upper = ((upper + 7) >> 3) << 3; /*aligned to 8*/
	upper = llmin(upper, _cudaBwtSeqLength);
	upper = (upper >> 3) << 3; /*in case of _cudaBwtSeqLength is not multiples of 8*/

	/*compute the alignment*/
	localTBWidth = (_cudaMaxReadLength + 3) >> 2;	/*aligned to 4*/
	rposition = hash.x + (strand ? hash.z >> 2 : 0);
	gposition = lower >> 2;

	glength = llmax(upper - lower, 0);
	endXY = _devSWAlign(rposition, hash.z, gposition, glength, global, localTB,
			localTBWidth);

	/*trace back to get the alignment path in  CIGAR format*/
	numCigars = 0;
	cigars = (uint16_t*) global;
	inCigars = cigars + 1;
	pid = _devSWAlign2Cigar(inCigars, numCigars, localTB, localTBWidth, endXY,
			rposition, gposition, startXY);

	/*calculate the genome index and mapping positions on the genome*/
	genomeIndex = _devGetGenomeIndex(seed.x);
	lower += startXY.y; /*genome alignment offset*/
	lower = lower - tex1Dfetch(_texBwtAnns, genomeIndex) + 1;

#if 0
	/*calcualte the ratio of bases in the alignment. For debugging*/
	float baseRatio = (float) (endXY.x - startXY.x + 1) * 100.0f / hash.y;
	printf(
			"gid: %d seedPosition %d seedLength %d lower %ld upper %ld strand %d hash.x %d pid %g baseRatio %g\n",
			gid, seedPosition, seedLength, lower, upper, strand, hash.x, pid,
			baseRatio);

	printf("tid %d genomeIndex %d lower %ld startXY %d %d endXY %d %d\n",
			threadIdx.x, genomeIndex, lower, startXY.x, startXY.y, endXY.x,
			endXY.y);
#endif

	/*extend the cigars*/
	inCigars += numCigars;
	if (startXY.x > 0) {
		/*extend to the start of the read*/
		*cigars = (startXY.x << 2) | ALIGN_OP_S; /*soft cliping*/
		++numCigars;
	}
	if (endXY.x + 1 < hash.y) {
		/*extend to the end of the read*/
		*inCigars = ((hash.y - endXY.x - 1) << 2) | ALIGN_OP_S;
		++numCigars;
	}
	inCigars = (startXY.x > 0) ? cigars : cigars + 1;

	/*if the alignment does not meet the contraints, we consider this read does not have any alignment on the genome*/
	float basePortion = 100 * (endXY.x - startXY.x + 1);
	basePortion = basePortion / hash.y;

	numCigars =
			(pid < _cudaMinID || basePortion < _cudaMinBaseRatio) ?
					0 : numCigars;

	/*make sure that the number of cigars does not exceed the memory limit*/
	numCigars = (numCigars > maxNumCigars) ? 0 : numCigars;

	/*save the number of cigars*/
	*(cudaNumCigars + gid) = numCigars;

	/*save the cigars*/
	outCigars = cudaCigars + gid;
	for (uint32_t i = 0; i < numCigars; ++i) {
		*outCigars = *inCigars++;
		outCigars += cudaCigarWidth;
	}

	/*save the alignment*/
	cudaAligns[gid] = make_uint2((genomeIndex << 1) | strand, lower);
	cudaBasePortions[gid] = basePortion * 0.01;
}

void GPUSW::configKernels() {
	cudaFuncSetCacheConfig(kernelSWAlignScore, cudaFuncCachePreferL1);
	myCheckCudaError;

	cudaFuncSetCacheConfig(kernelSWAlign, cudaFuncCachePreferL1);
	myCheckCudaError;
}
void GPUSW::initAlign(uint2*devSeeds, uint32_t* devReadIndices,
		size_t numSeeds) {
	cudaChannelFormatDesc channelDes = cudaCreateChannelDesc<uint2>();

	/*bind texture memory*/
	_texDevSeeds.addressMode[0] = cudaAddressModeClamp;
	_texDevSeeds.filterMode = cudaFilterModePoint;
	_texDevSeeds.normalized = false;

	cudaBindTexture(NULL, _texDevSeeds, devSeeds, channelDes,
			numSeeds * sizeof(uint2));
	myCheckCudaError;

	channelDes = cudaCreateChannelDesc<uint32_t>();
	/*bind texture memory*/
	_texDevReadIndices.addressMode[0] = cudaAddressModeClamp;
	_texDevReadIndices.filterMode = cudaFilterModePoint;
	_texDevReadIndices.normalized = false;

	cudaBindTexture(NULL, _texDevReadIndices, devReadIndices, channelDes,
			numSeeds * sizeof(uint32_t));
	myCheckCudaError;
}
void GPUSW::finalizeAlign() {

	cudaUnbindTexture(_texDevSeeds);
	myCheckCudaError;

	cudaUnbindTexture(_texDevReadIndices);
	myCheckCudaError;
}

void GPUSW::align(uint16_t* devCigars, uint32_t devCigarWidth,
		uint32_t maxNumCigars, uint8_t* devNumCigars, uint2* devAligns,
		float* devBasePoritions, size_t numSeeds, int nthreads,
		cudaStream_t stream) {
	int32_t nblocks = (numSeeds + nthreads - 1) / nthreads; /*number of thread blocks*/
	dim3 grid(nblocks, 1);
	dim3 blocks(nthreads, 1);

	kernelSWAlign<<<grid, blocks, 0, stream>>>(devCigars, devCigarWidth, maxNumCigars, devNumCigars, devAligns, devBasePoritions, numSeeds);
	myCheckCudaError;
}

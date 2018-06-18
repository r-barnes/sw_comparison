/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "CFastaSWScalar.h"
#define CUERR { cudaError_t err;			\
	if ((err = cudaGetLastError()) != cudaSuccess) {		\
  		printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }}

__device__ __constant__ int cudaSubMatrix[32][32];
__device__ __constant__ int cudaGapOpen;
__device__ __constant__ int cudaGapExtend;
__device__ __constant__ int cudaGapOE;		//cudaGapOpen + cudaGapExtend;
__device__ __constant__ unsigned char cudaQuery[MAX_QUERY_LEN];
__device__ __constant__ int cudaQueryLen;
__device__ __constant__ int cudaQueryAlignedLen;


texture<unsigned int, 2, cudaReadModeElementType> InterSeqs;
texture<unsigned char, 2, cudaReadModeElementType> IntraSeqs;
texture<int4, 2, cudaReadModeElementType> InterQueryPrf;

CFastaSWScalar::CFastaSWScalar() : CFastaSW()
{
}
CFastaSWScalar::~CFastaSWScalar()
{
	//do nothing
}
//global functions
void CFastaSWScalar::swMemcpyParameters(int matrix[32][32], int gapOpen, int gapExtend)
{
	int gapoe = gapOpen + gapExtend;

	cudaMemcpyToSymbol(cudaSubMatrix,matrix,32 * 32 *sizeof(int));
	CUERR
	
	cudaMemcpyToSymbol(cudaGapOpen, &gapOpen, sizeof(int));
	CUERR

	cudaMemcpyToSymbol(cudaGapExtend, &gapExtend,sizeof(int));
	CUERR

	cudaMemcpyToSymbol(cudaGapOE, &gapoe, sizeof(int));
	CUERR
}
void CFastaSWScalar::swMemcpyQuery(unsigned char* query, int qlen, int qAlignedLen, int offset, int matrix[32][32])
{
	int i, j;
	int4* hostQueryPrf;

	//copy the query sequence length
    cudaMemcpyToSymbol(cudaQuery, query, qlen * sizeof(unsigned char), offset, cudaMemcpyHostToDevice);
    CUERR

    cudaMemcpyToSymbol(cudaQueryLen, &qlen, sizeof(int));
    CUERR
    cudaMemcpyToSymbol(cudaQueryAlignedLen, &qAlignedLen, sizeof(int));
    CUERR

	//build the profile for inter-task parallelization
	int prfLength = qAlignedLen / 4;
	cudaInterQueryPrf = swMallocArray(prfLength, 32, pChannelFormatKindSignedInt4);
	hostQueryPrf = (int4*)pMallocHost(sizeof(int4) * prfLength * 32);
	for(i = 0; i < 32; i++){
		int4* p = hostQueryPrf + i * prfLength;
		for(j = 0; j < qAlignedLen; j += 4){
			p->x = matrix[i][query[j]];
			p->y = matrix[i][query[j + 1]];
			p->w = matrix[i][query[j + 2]];
			p->z = matrix[i][query[j + 3]];
			//increase the pointer
			p++;
		}
	}
	pMemcpy2DToArray(cudaInterQueryPrf, 0, 0, hostQueryPrf, prfLength * sizeof(int4), prfLength * sizeof(int4), 32, pMemcpyHostToDevice);
	pFreeHost(hostQueryPrf);

}
void CFastaSWScalar::swMemcpyQueryLength( int qlen, int qAlignedLen)
{
    cudaMemcpyToSymbol(cudaQueryLen, &qlen, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUERR

    cudaMemcpyToSymbol(cudaQueryAlignedLen, &qAlignedLen, sizeof(int), 0, cudaMemcpyHostToDevice);
    CUERR
}
void CFastaSWScalar::swBindTextureToArray()
{
    cudaBindTextureToArray(InterSeqs,(cudaArray*)cudaInterSeqs, uint_channelDesc);
    CUERR

    InterSeqs.addressMode[0] = cudaAddressModeClamp;
    InterSeqs.addressMode[1] = cudaAddressModeClamp;
    InterSeqs.filterMode = cudaFilterModePoint;
    InterSeqs.normalized = false;
	
	cudaBindTextureToArray(IntraSeqs,(cudaArray*)cudaIntraSeqs, uchar_channelDesc);
	CUERR

	IntraSeqs.addressMode[0] = cudaAddressModeClamp;
    IntraSeqs.addressMode[1] = cudaAddressModeClamp;
    IntraSeqs.filterMode = cudaFilterModePoint;
    IntraSeqs.normalized = false;  

}
void CFastaSWScalar::swBindQueryProfile()
{
	cudaBindTextureToArray(InterQueryPrf,(cudaArray*)cudaInterQueryPrf, sint4_channelDesc);
	CUERR

	InterQueryPrf.addressMode[0] = cudaAddressModeClamp;
    InterQueryPrf.addressMode[1] = cudaAddressModeClamp;
    InterQueryPrf.filterMode = cudaFilterModePoint;
    InterQueryPrf.normalized = false;  

}
void CFastaSWScalar::swUnbindTexture()
{
	cudaUnbindTexture(InterSeqs);
	CUERR

	cudaUnbindTexture(IntraSeqs);
	CUERR
}
void CFastaSWScalar::swUnbindQueryProfile()
{
	cudaUnbindTexture(InterQueryPrf);
	CUERR
	
	//release the CUDA array
	pFreeArray(cudaInterQueryPrf);
	cudaInterQueryPrf = 0; 
}
void CFastaSWScalar::swInterMallocThreadSlots(int threads, int multiProcessors, int slotSize)
{
	int slots;
	//calculate the number of slots to be allocated
	slots = threads * multiProcessors;
	cudaGlobal = (ushort2*)pMallocPitch(sizeof(ushort2),slots, slotSize, &cudaGlobalPitch);
}
void CFastaSWScalar::swInterFreeThreadSlots()
{
	pFree(cudaGlobal);
}

/*************************************************************
		Smith-Waterman for inter-task parallelization
**************************************************************/
__device__ int InterGlobalSmithWaterman_DB(ushort2* global, size_t gpitch, 
							int qlen, int db_cx, int db_cy, int dblen)
{
	int i, j, k;
	int sa, sb;
	int maxHH, e;

	int4 sub;
    int4 f, h, p;	//the first 4 rows
	int4 f0, h0, p0; //the second 4 rows

	ushort2	HD; 
	ushort2 initHD = {0, 0};
	int4 zero = {0, 0, 0, 0};
	
    for (i = 0; i <= dblen; i++)
    {
		*((ushort2*)(((char*)global) + i * gpitch))= initHD;
    }

    maxHH = 0;
    for (i = 1; i <= qlen; i += SEQ_LENGTH_ALIGNED)
    {
        h = p = zero;
		f = zero;

		h0 = p0 = zero;
		f0 = zero;

        for (j = 1; j <= dblen; j += 4)
        {
			//load the packed 4 residues
			unsigned int pac4= tex2D(InterSeqs, db_cx, db_cy + (j >> 2)); //no need to use (j - 1) >> 2
			/*packed[0] = res.x;
			packed[1] = res.y;
			packed[2] = res.w;
			packed[3] = res.z;*/
			//compute the cell block SEQ_LENGTH_ALIGNED x 4
			for(k = 0; k < 4; k++){
				//get the (j + k)-th residue
				sa = (pac4 >> (k << 3)) & 0x0FF;

				//load data
				HD = *((ushort2*)(((char*)global) + (j + k) * gpitch));
				sb = i >> 2;	//no need to use (i - 1) >> 2;
				//load substitution matrix
				sub = tex2D(InterQueryPrf, sb++, sa);
			
				//compute the cell (0, 0);
				f.x = max(h.x - cudaGapOE, f.x - cudaGapExtend);
				e = max(HD.x - cudaGapOE, HD.y - cudaGapExtend);

    			h.x = p.x + sub.x;
				h.x = max(h.x, f.x);
				h.x = max(h.x, e);
				h.x = max(h.x, 0);
				maxHH = max(maxHH, h.x);
			
				p.x = HD.x;

				//compute cell (0, 1)
				f.y = max(h.y - cudaGapOE, f.y - cudaGapExtend);
    	        e = max(h.x - cudaGapOE, e - cudaGapExtend);
	
				h.y = p.y + sub.y;
				h.y = max(h.y, f.y);
				h.y = max(h.y, e);
				h.y = max(h.y, 0);
				maxHH = max(maxHH, h.y);
		
				p.y = h.x;
				
				//compute cell (0, 2);
				f.w = max(h.w - cudaGapOE, f.w - cudaGapExtend);
				e = max(h.y - cudaGapOE, e - cudaGapExtend);
				
				h.w = p.w + sub.w;
				h.w = max(h.w, f.w);
				h.w = max(h.w, e);
				h.w = max(h.w, 0);
				maxHH = max(maxHH, h.w);
		
				p.w = h.y;
					
				//compute cell (0, 3)
 				f.z = max(h.z - cudaGapOE, f.z - cudaGapExtend);
				e = max(h.w - cudaGapOE, e - cudaGapExtend);
		
				h.z = p.z + sub.z;
				h.z = max(h.z, f.z);
				h.z = max(h.z, e);
				h.z = max(h.z, 0);
				maxHH = max(maxHH, h.z);
	
				p.z = h.w;
				
				//load substitution matrix
				sub = tex2D(InterQueryPrf, sb, sa);

				//compute cell(0, 4)
				f0.x = max(h0.x - cudaGapOE, f0.x - cudaGapExtend);
				e = max(h.z - cudaGapOE, e - cudaGapExtend);
	
				h0.x = p0.x + sub.x;
				h0.x = max(h0.x, f0.x);
				h0.x = max(h0.x, e);
				h0.x = max(h0.x, 0);
				maxHH = max(maxHH, h0.x);
	
				p0.x = h.z;
					
				//compute cell(0, 5)
				f0.y = max(h0.y - cudaGapOE, f0.y - cudaGapExtend);
				e = max(h0.x - cudaGapOE, e - cudaGapExtend);
	
				h0.y = p0.y + sub.y;
				h0.y = max(h0.y, f0.y);
				h0.y = max(h0.y, e);
				h0.y = max(h0.y, 0);
				maxHH = max(maxHH, h0.y);
	
				p0.y = h0.x;
					
				//compute cell (0, 6)
				f0.w = max(h0.w - cudaGapOE, f0.w - cudaGapExtend);
				e = max(h0.y - cudaGapOE, e - cudaGapExtend);
	
				h0.w = p0.w + sub.w;
				h0.w = max(h0.w, f0.w);
				h0.w = max(h0.w, e);
				h0.w = max(h0.w, 0);
				maxHH = max(maxHH, h0.w);
	
				p0.w = h0.y;
					
				//compute cell(0, 7)
				f0.z = max(h0.z - cudaGapOE, f0.z - cudaGapExtend);
				e = max(h0.w - cudaGapOE, e - cudaGapExtend);
	
				h0.z = p0.z + sub.z;
				h0.z = max(h0.z, f0.z);
				h0.z = max(h0.z, e);
				h0.z = max(h0.z, 0);
				maxHH = max(maxHH, h0.z);
	
				p0.z = h0.w;
	
				HD.x = min(h0.z, 0x0FFFF);
				e = max(e, 0);
				HD.y = min(e, 0x0FFFF);
	
				//save data cell(0, 7)
				*((ushort2*)(((char*)global) + (j + k) * gpitch)) = HD;
			}	
    	}
	}
    return maxHH;
}
__global__ void interSWUsingSIMT(ushort2* cudaGlobal, size_t cudaGlobalPitch, DatabaseHash* hash,
				SeqEntry* cudaResult, int numSeqs, int firstBlk)
{
	int tidx;
	unsigned int tid;
	int seqidx;
	int score;

	tid = threadIdx.y * blockDim.x + threadIdx.x;
	//calculate the index of the current thread block;
	tidx = blockIdx.y * gridDim.x + blockIdx.x;
	//calculate the index of the first thread
	tidx *= blockDim.x * blockDim.y;
	//calculate the index of the current thread
	tidx += tid;

	seqidx = tidx + firstBlk * blockDim.x * blockDim.y;

	if(seqidx >= numSeqs){
		return;
	}
	//get the hash item
	DatabaseHash dbhash = hash[seqidx];
	
	score = InterGlobalSmithWaterman_DB(cudaGlobal + tidx, cudaGlobalPitch,
               	cudaQueryAlignedLen, dbhash.cx, dbhash.cy, dbhash.alignedLen);

	cudaResult[seqidx].value = min(score, 0x0FFFF);
}

void CFastaSWScalar::InterRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstBlk)
{
	dim3 grid (blknum, 1, 1);
	dim3 blocks (threads, 1, 1);

	interSWUsingSIMT<<<grid, blocks, 0>>>((ushort2*)cudaGlobal, cudaGlobalPitch,
				cudaSeqHash, cudaResult, numSeqs, firstBlk);	
	
	CUERR
	//kernel-level synchronization
	cudaThreadSynchronize();
}
/*******************************************************************
		Smith-Waterman for intra-task parallelization
********************************************************************/
__device__ int IntraGlobalSmithWatermanScalar(int matrix[32][32], ushort*D_A, ushort*D_B, ushort*D_C, 
		ushort*H, ushort*V_O, ushort*V_C, int db_cx, int db_cy, int n, unsigned char* query, int m)
{
    
	int i,j;
	unsigned char a,b;
	int dd,h,v;
	int lx,mn;
	int score;

	//the maximum number of threads is 256
	extern __shared__ int maxHH[];

	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int step = blockDim.x * blockDim.y;

	score = 0;
	mn = m;
	for(i = tid; i <= mn; i += step){
		D_A[i] = 0;
		D_B[i] = 0;
		D_C[i] = 0;
		
		H[i] = 0;
		V_O[i] = 0;
		V_C[i] = 0;
	}
	__syncthreads();
	
	/*the horizontal sequence in the edit graph is the database sequence;
		the vertical sequence in the edit graph is the query sequence*/

	mn = min(m, n);
	for( i = 1; i <= m + n - 1; i++){

		lx = i;
		int sj = 0;
		int ej = min(min(i, m + n - i), mn);
		if(i > n){
			lx = n;
			sj = i- n;
		}
		ej += sj;
		sj ++;
	
		lx -= tid;
		__syncthreads();
			
		for( j = sj + tid; j <= ej ; j += step, lx -= step){

			//calculate the vetical value
			h = max(H[j] - cudaGapExtend, D_B[j] - cudaGapOE);
			h = max(h, 0);
			//calcuate the horizontal value
			v = max(V_O[j-1] - cudaGapExtend, D_B[j-1] - cudaGapOE);
			v = max(v, 0);

			//calculate the diagonal value
			a = tex2D(IntraSeqs, db_cx + lx, db_cy);
			b = query[j];
			dd = D_A[j-1] + matrix[a][b];
			dd = max(dd, h);
			dd = max(dd, v);
			dd = min(dd, 0x0FFFF);

			//save the values
			D_C[j] = dd;
			H[j] = min(h, 0x0FFFF);
			V_C[j] = min(v, 0x0FFFF);
		
			score = max(score, dd);
		}
		
		//swap the buffers A <- B; B <- C;
		ushort* tmp = D_A;
		D_A = D_B;
		D_B = D_C;
		D_C = tmp;

		tmp = V_C;
		V_C = V_O;
		V_O = tmp;
	
	}
	maxHH[tid] = score;
	__syncthreads();
	
	if(tid == 0){
		score = maxHH[0];
		for( i = 1; i < step; i++){
			score = max(score, maxHH[i]);
		}
	}
	return score;
}

#define MAX_SM_QUERY_LENGTH				11264	//11KB by default, actually up to 12192
__global__ void intraSWUsingSIMT(ushort* D_A, size_t daPitch, ushort* D_B, size_t dbPitch, 
				ushort* D_C, size_t dcPitch, ushort* H, size_t hPitch, ushort* V_O, size_t voPitch, ushort* V_C, size_t vcPitch, DatabaseHash* hash, 
				SeqEntry* cudaResult, int numSeqs, int firstSeq)
{
	int score;
	int blkid;
	unsigned int tid;
	int step;
	int seqidx;
	int i, x, y;

	extern __shared__ unsigned char smQuery[];
	__shared__ int matrix[32][32];

	tid = threadIdx.y * blockDim.x + threadIdx.x;
	step = blockDim.x * blockDim.y;
	blkid = blockIdx.y * gridDim.x + blockIdx.x;

	//load substitution matrix from constant memory to shared memory
	x = tid & 0x1f;  // tid %32
	y = tid >> 5; //tid / 32
	for(i = 0;i < 32;i += 8){
		matrix[i + y][x] = cudaSubMatrix[i + y][x];
	}
	__syncthreads();
	
	//compute the index of the database sequence corresponding to the current thread block
	seqidx = blkid + firstSeq;

	if(seqidx >= numSeqs){
		return;
	}
	//get the hash item of the database sequence
	DatabaseHash dbhash = hash[seqidx];

	if(cudaQueryLen >= MAX_SM_QUERY_LENGTH){
	
		score = IntraGlobalSmithWatermanScalar(matrix,
				(ushort*)(((char*)D_A) + blkid * daPitch), (ushort*)(((char*)D_B) + blkid * dbPitch),
				(ushort*)(((char*)D_C) + blkid * dcPitch), (ushort*)(((char*)H) + blkid * hPitch),
				(ushort*)(((char*)V_O) + blkid * voPitch), (ushort*)(((char*)V_C) + blkid * vcPitch),
				dbhash.cx, dbhash.cy, dbhash.length, cudaQuery, cudaQueryLen);
	}else{
		//load query sequence into the shared memory
		for( i = tid + 1; i <= cudaQueryLen; i += step){
			smQuery[i] = cudaQuery[i];
		}
		__syncthreads();

	  	score = IntraGlobalSmithWatermanScalar(matrix,
                (ushort*)(((char*)D_A) + blkid * daPitch), (ushort*)(((char*)D_B) + blkid * dbPitch),
                (ushort*)(((char*)D_C) + blkid * dcPitch), (ushort*)(((char*)H) + blkid * hPitch),
                (ushort*)(((char*)V_O) + blkid * voPitch), (ushort*)(((char*)V_C) + blkid * vcPitch),
				dbhash.cx, dbhash.cy, dbhash.length, smQuery, cudaQueryLen);
	}

	if(tid == 0){
		cudaResult[seqidx].value = score;
	}
}
void CFastaSWScalar::IntraRunGlobalDatabaseScanning(int blknum, int threads, int numSeqs, int firstSeq)
{
    dim3 grid (blknum, 1, 1);
    dim3 blocks (threads, 1, 1);

   	intraSWUsingSIMT<<<grid, blocks, MAX_SM_QUERY_LENGTH>>>((ushort*)cudaDA, cudaDAPitch,
            (ushort*)cudaDB, cudaDBPitch, (ushort*)cudaDC, cudaDCPitch,
            (ushort*)cudaHH, cudaHHPitch, (ushort*)cudaVV_O, cudaVV_OPitch,
			(ushort*)cudaVV_C, cudaVV_CPitch, cudaSeqHash,
            cudaResult, numSeqs, firstSeq);

    CUERR
    //kernel-level synchronization
    cudaThreadSynchronize();
}


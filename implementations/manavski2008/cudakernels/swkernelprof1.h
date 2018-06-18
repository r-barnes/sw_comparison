
#ifndef _TEMPLATE_KERNELPROF1_H_
#define _TEMPLATE_KERNELPROF1_H_

// was 2052 with CUDA 0.8
#define LOCAL_MEMORY	(2048/4)

#define MAX_BLOCK_SIZE	64
#define BIAS			100
#define PADDING_FACT	4

texture<unsigned, 1, cudaReadModeElementType> texProf;

struct __align__(16) SWValuesGroup {
	short int Hi;
	short int Hip1;
	short int Hip2;
	short int Hip3;
	short int Ei;
	short int Eip1;
	short int Eip2;
	short int Eip3;
};

__global__ void sw_kernelprof1( const unsigned querySize, const char* g_seqlib, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_scores) {

	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	//numero di sequenza
	unsigned seqNum = (blid*MAX_BLOCK_SIZE) + tid + seqOffset;

	//offset di tale sequenza nel DB
	unsigned libOffset = g_offsets[seqNum];

	unsigned sizeA = g_sizes[seqNum];
	unsigned sizeB = querySize;

	unsigned numQueryBlocks = querySize / 4;

	__syncthreads();

	// 4 sono i valori utili di E ed H in SWValuesGroup
	// 2 * perche' serve un array per la riga corrente ed uno per la lettura della riga precedente
	SWValuesGroup HE[ (2*LOCAL_MEMORY) ];
	SWValuesGroup curIn, curOut;

	curIn.Hi   = 0;
	curIn.Hip1 = 0;
	curIn.Hip2 = 0;
	curIn.Hip3 = 0;
	curIn.Ei   = 0;
	curIn.Eip1 = 0;
	curIn.Eip2 = 0;
	curIn.Eip3 = 0;

	for (unsigned cnt=0; cnt<=LOCAL_MEMORY; ++cnt) {
		HE[cnt] = curIn;
	}

	//variabili locali
	short int h_north_west;
	short int score = 0;
	short int f_jab;
	short int h_jab;
	short int f;
	short int h, e;
	short int tmp1, tmp2;

	char a;
	int idx = 0;
	int pos;
	unsigned profElem;

	unsigned j;
	for (unsigned i=1; i<sizeA; ++i) {
		//carattere i-esimo della sequenza
		a = g_seqlib[ (libOffset + i) ];
		pos = ( (unsigned)a - 60 ) * numQueryBlocks;

		//if (i==2) printf("a:%c pos:%d row:%d\n", a, pos, row);

		f_jab=0, h_jab=0;

		h_north_west = 0;
		for (j=1; j<=numQueryBlocks; ++j) {

			profElem = tex1Dfetch(texProf, pos++);

			//if (i==2) printf("profelem:%d\n", profElem);

			curIn = HE[ ( __umul24(LOCAL_MEMORY, idx) + j) ];

			//calcolo e, h ,f per il 1 elemento
			tmp1 = curIn.Ei - beta;
			tmp2 = curIn.Hi - alpha;
			e	= max( tmp1, tmp2 );

 			tmp1 = f_jab - beta;
 			tmp2 = h_jab - alpha;
 			f	= max( tmp1, tmp2 );

			tmp1 = (profElem & 0xff) - BIAS;
			h	= h_north_west + tmp1;
			h	= max( 0, h );
			h	= max( h, e );
 			h	= max( h, f );

			score = max(score, h);

			//if (i==2) printf("e:%d f:%d h:%d score:%d\n", e, f, h, score);

 			f_jab = f;
 			h_jab = h;
			
			curOut.Hi = h;
			curOut.Ei = e;

			//calcolo e, h ,f per il 2 elemento
			tmp1 = curIn.Eip1 - beta;
			tmp2 = curIn.Hip1 - alpha;
			e	= max( tmp1, tmp2 );

 			tmp1 = f_jab - beta;
 			tmp2 = h_jab - alpha;
 			f	= max( tmp1, tmp2 );

			tmp1 = ((profElem >> 8) & 0xff) - BIAS;
			h	= curIn.Hi + tmp1 ;
			h	= max( 0, h );
			h	= max( h, e );
 			h	= max( h, f );

			score = max(score, h);

			//if (i==2) printf("e:%d f:%d h:%d score:%d\n", e, f, h, score);
			
 			f_jab = f;
 			h_jab = h;
			
			curOut.Hip1 = h;
			curOut.Eip1 = e;

			//calcolo e, h ,f per il 3 elemento
			tmp1 = curIn.Eip2 - beta;
			tmp2 = curIn.Hip2 - alpha;
			e	= max( tmp1, tmp2 );

 			tmp1 = f_jab - beta;
 			tmp2 = h_jab - alpha;
 			f	= max( tmp1, tmp2 );

			tmp1 = ((profElem >> 16) & 0xff) - BIAS;
			h	= curIn.Hip1 + tmp1 ;
			h	= max( 0, h );
			h	= max( h, e );
 			h	= max( h, f );

			score = max(score, h);

			//if (i==2) printf("e:%d f:%d h:%d score:%d\n", e, f, h, score);
			
 			f_jab = f;
 			h_jab = h;
			
			curOut.Hip2 = h;
			curOut.Eip2 = e;

			//calcolo e, h ,f per il 4 elemento
			tmp1 = curIn.Eip3 - beta;
			tmp2 = curIn.Hip3 - alpha;
			e	= max( tmp1, tmp2 );

 			tmp1 = f_jab - beta;
 			tmp2 = h_jab - alpha;
 			f	= max( tmp1, tmp2 );

			tmp1 = ((profElem >> 24) & 0xff) - BIAS;
			h	= curIn.Hip2 + tmp1 ;
			h	= max( 0, h );
			h	= max( h, e );
 			h	= max( h, f );

			score = max(score, h);

			//if (i==2) printf("e:%d f:%d h:%d score:%d\n", e, f, h, score);

 			f_jab = f;
 			h_jab = h;

			curOut.Hip3 = h;
			curOut.Eip3 = e;

			h_north_west = curIn.Hip3;

			HE[ (__umul24(LOCAL_MEMORY, 1-idx) + j) ] = curOut;
		}

		idx = 1 - idx;

	}
	// write data to global memory
	g_scores[ blid*MAX_BLOCK_SIZE + seqOffset + tid ] = score;
}

#endif

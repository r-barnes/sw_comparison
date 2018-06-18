
#ifndef _TEMPLATE_KERNEL_TXT_1_H_
#define _TEMPLATE_KERNEL_TXT_1_H_

#define HSH( index )      CUT_BANK_CHECKER(Hsh, (index))
#define ESH( index )      CUT_BANK_CHECKER(Esh, (index))

extern  __shared__  short int int_sharedspace_txt_1[];
extern  __shared__  char char_sharedspace_txt_1[];

#define MAX_SEQUENCE_LEN 64

// contiene: int pam[32][32], Bseq
#define DIMSHAREDSPACE_TXT_1 ((1024 + MAX_SEQUENCE_LEN + 1)*sizeof(short int))

#define BSEQ64_TXT_1( mac_idx )		CUT_BANK_CHECKER(char_sharedspace_txt_1, (1024*sizeof(short int) + (mac_idx)))
#define PAM_TXT_1( mac_i, mac_j )	CUT_BANK_CHECKER(int_sharedspace_txt_1, ((mac_i)*32 + (mac_j)))

texture<unsigned char, 2> texLib;


__device__ void loadPAM_Txt1_64threads(const int &tid)
{
	// read the substitution matrix
	unsigned idxp = tid/32, restp = tid%32;

	PAM_TXT_1(idxp, restp)				= blosum50[idxp][restp];	//riga 0,1
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 2,3
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 4,5
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 6,7
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 8,9
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 10,11
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 12,13
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 14,15
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 16,17
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 18,19
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 20,21
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 22,23
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 24,25
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 26,27
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 28,29
	idxp += 2; PAM_TXT_1(idxp, restp)	= blosum50[idxp][restp];	//riga 30,31
}

__device__ void swcalc_kernel_txt_1(const char &aElem, const char &bElem, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const unsigned int &alpha, const unsigned int &beta, int &h, int &e, int &f)
{
	int tmp1 = h_n - alpha;
	int tmp2 = f_n - beta;
	f = max(tmp1, tmp2);
	
	tmp1 = h_w - alpha;
	tmp2 = e_w - beta;
	e = max(tmp1, tmp2);
	
	tmp1 = aElem - 60;
	tmp2 = bElem - 60;

	h = h_nw + PAM_TXT_1(tmp1, tmp2);
	h = max(0, h);
	h = max(e, h);
	h = max(f, h);
}

__device__ char textureFetcher(const unsigned &seqNum, const unsigned numChar, unsigned* g_offsets)
{	
	unsigned qt1= numChar / 9;
	unsigned qt2 = seqNum / 64;
	unsigned off = g_offsets[qt2]+qt1;
	unsigned seq = seqNum % 64;
	unsigned seqChar = numChar % 9;

	int x = (seqChar/3)*8 + (seq/8);
	int y = (seqChar%3)*8 + (seq%8);

	char elem = texfetch(texLib, x, y);

	return elem;
}

__global__ void sw_kernel_txt_1( const char* g_strToAlign, const unsigned sizeNotPad, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_scores) {

	const unsigned int MAX_NUM_THREADS = 32;

	const unsigned numThreads = blockDim.x;

	// shared memory
	//ho bisogno di una riga di E e di una riga di H per ogni sequenza
	__shared__  short int Hsh[MAX_NUM_THREADS*(MAX_SEQUENCE_LEN+1)];
	__shared__  short int Esh[MAX_NUM_THREADS*(MAX_SEQUENCE_LEN+1)];

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// read the substitution matrix
	unsigned elemPerThread = MAX_NUM_THREADS / numThreads;
	for (unsigned cnt=0; cnt<elemPerThread; cnt++) {
		loadPAM_Txt1_64threads(cnt + (tid*elemPerThread));
		loadPAM_Txt1_64threads(cnt + (tid*elemPerThread) + MAX_NUM_THREADS);
	}

	unsigned tempVar = (blid*MAX_NUM_THREADS) + tid + seqOffset;

	unsigned libOffset = g_offsets[tempVar];

	unsigned sizeA = g_sizes[tempVar];
	unsigned sizeB = sizeNotPad;

	// the one we search for (only for 64 for now)
	for (unsigned cnt=0; cnt<elemPerThread; cnt++) {
		tempVar = cnt + (tid*elemPerThread);
		BSEQ64_TXT_1(tempVar) = g_strToAlign[tempVar];
		tempVar = cnt + (tid*elemPerThread) + MAX_NUM_THREADS;
		BSEQ64_TXT_1(tempVar) = g_strToAlign[tempVar];
	}
	BSEQ64_TXT_1(MAX_SEQUENCE_LEN) = g_strToAlign[MAX_SEQUENCE_LEN];

	unsigned columnOffsets = tid * (MAX_SEQUENCE_LEN + 1);

	for (unsigned i=0; i<MAX_SEQUENCE_LEN+1; i++) {
		tempVar = columnOffsets + i;
		HSH(tempVar) = 0;
		ESH(tempVar) = 0;
	}

	int score = 0;

	__syncthreads();
	
	int f=0, e=0, h=0;
	int h_jsucc=0, e_jsucc=0, f_jsucc=0;
	unsigned j; char aElem;
	for (unsigned i=1; i<sizeA; i++) {

		//taking the single element of the compared sequence
		//aElem = g_seqlib[libOffset + i];
		//aElem =  texfetch( texLib, static_cast<int>(libOffset + i) );
		//aElem = textureFetcher(tempVar, i, g_offsets);

		h_jsucc = e_jsucc = f_jsucc=0;

		j = 1;
		for (; j<sizeB; j++ ) {

			tempVar = columnOffsets + j;

			swcalc_kernel_txt_1(aElem, BSEQ64_TXT_1(j), h_jsucc, f_jsucc, HSH(tempVar), ESH(tempVar), HSH(tempVar-1), alpha, beta, h, e, f);

			score = max(score, h);

			tempVar = columnOffsets + j - 1;
			HSH(tempVar) = h_jsucc;
			ESH(tempVar) = e_jsucc;
			h_jsucc = h;
			f_jsucc = f;
			e_jsucc = e;
		}
		tempVar = columnOffsets + j - 1;
		HSH(tempVar) = h_jsucc;
		ESH(tempVar) = e_jsucc;
	}

	// write data to global memory
	__syncthreads();
	g_scores[ blid*MAX_NUM_THREADS + seqOffset + tid ] = score;

}

#endif



#ifndef _TEMPLATE_SWUTILS_H_
#define _TEMPLATE_SWUTILS_H_

union IntChar4
{
	int union_int;
	char union_a, union_b, union_c, union_d;
};

// contiene: int pam[32][32], scores[128] , Aseq[NUM_THREADS+1], Bseq[NUM_THREADS+1]
// totale DIMSHAREDSPACE1_2 = 32*32 + 128 = 1152 integers
extern  __shared__  IntChar4 sharedspace[];

#define DIMSHAREDSPACE1_2 (1152 + 130)*sizeof(IntChar4)
#define DIMSHAREDSPACE3 (1152 + 256)*sizeof(IntChar4)


// i,j: 0 - 31
//#define PAM( i, j )   CUT_BANK_CHECKER(sharedspace, ((i)*32 + (j)))
#define PAM( mac_i, mac_j )   (sharedspace[((mac_i)*32 + (mac_j))]).union_int

// index: 0-127
//#define SCORES( index )   CUT_BANK_CHECKER( sharedspace, (1024 + (index)) )
#define SCORES( mac_idx )   (sharedspace[1024 + (mac_idx)]).union_int


#define ASEQ64( mac_idx )	(sharedspace[1152 + (mac_idx)]).union_a
#define BSEQ64( mac_idx )	(sharedspace[1217 + (mac_idx)]).union_a


#define ASEQ128( mac_idx )	(sharedspace[1152 + (mac_idx)]).union_a
#define BSEQ128( mac_idx )	(sharedspace[1280 + (mac_idx)]).union_a

// calcola SW di una antidiagonale, considerando che i thread utili sono tid < curDiagIdx-1 
// n = north, w = west
// curDiagIdx va da 2 a 65 compreso per un totale di 64 diagonali
__device__ void swcalc_upleft(const unsigned &curDiagIdx, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const int &alpha, const int &beta, int &h, int &e, int &f) 
{

		int f1 = h_n - alpha;
		int f2 = f_n - beta;
		f = (threadIdx.x < curDiagIdx-1) ? max(f1, f2) : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		int e1 = h_w - alpha;
		int e2 = e_w - beta;
		e = (threadIdx.x < curDiagIdx-1) ? max(e1, e2) : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		int idxSeqA = curDiagIdx - threadIdx.x - 1; 
		idxSeqA = (idxSeqA > 0) ? idxSeqA : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!
		int idxSeqB = threadIdx.x + 1;
		idxSeqB = (idxSeqB >= curDiagIdx) ? 0 : idxSeqB; //!!!!!!!!!!!!!!!!!!!!!!!!!!!

		int pos1 = ASEQ64(idxSeqA)-60;
		int pos2 = BSEQ64(idxSeqB)-60;

		int h1 = h_nw + PAM(pos1, pos2); 
		h = (threadIdx.x < curDiagIdx-1) ? max(0, h1) : 0;
		//h = max(0, h1); 
		h = max(h, e); 
		h = max(h, f); 

		//printf("tid=%u diag=%u %c-%c res=%d h_n=%d h_w=%d h_nw=%d f_n=%d e_w=%d\t\t", threadIdx.x, curDiagIdx, ASEQ64(idxSeqA), BSEQ64(idxSeqB), PAM(pos1, pos2), h_n, h_w, h_nw, f_n, e_w);

}

// curDiagIdx va da 0 a 62 compreso per un totale di 63 diagonali
__device__ void swcalc_downright(const unsigned &curDiagIdx, const unsigned sizeA, const unsigned sizeB, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const int &alpha, const int &beta, int &h, int &e, int &f) 
{

		int f1 = h_n - alpha;
		int f2 = f_n - beta;
		f = (threadIdx.x > curDiagIdx) ? max(f1, f2) : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!
		
		int e1 = h_w - alpha;
		int e2 = e_w - beta;
		e = (threadIdx.x > curDiagIdx) ? max(e1, e2) : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!

		//ho inserito 65 hardcoded invece di sizeA perchè non deve dipendere da sizeA. probabilmente lo stesso andrà fatto per b quando si farà una lunghezza maggiore
		int idxSeqA = 65 - threadIdx.x + curDiagIdx; 
		idxSeqA = (idxSeqA < 65) ? idxSeqA : 0; //!!!!!!!!!!!!!!!!!!!!!!!!!!!

		int idxSeqB = sizeB - 64 + threadIdx.x;///!!!!!!!!!!!!!!!!!!!va bene solo per una stringa di 64
		idxSeqB = ( idxSeqB < (sizeB - 63 + curDiagIdx) ) ? 0 : idxSeqB;

		int pos1 = ASEQ64(idxSeqA)-60;
		int pos2 = BSEQ64(idxSeqB)-60;

		int h1 = h_nw + PAM(pos1, pos2); 
		h = (threadIdx.x > curDiagIdx) ? max(0, h1) : 0;
		h = max(h, e);
		h = max(h, f);

		//printf("tid=%u diag=%u %c-%c res=%d\t\t", threadIdx.x, curDiagIdx, ASEQ64(idxSeqA), BSEQ64(idxSeqB), PAM(pos1, pos2));
}

// curDiagIdx va da 64 a 127 compreso per un totale di 64 diagonali
__device__ void swcalc_complete_64(const unsigned &curDiagIdx, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const unsigned int &alpha, const unsigned int &beta, int &h, int &e, int &f) 
{
		// la sequenza B (la verticale) e' considerata per intera 
		// mentre la A e' disponibile solo per una estensione 2*NUM_THREADS
		int tmp1 = h_n - alpha;
		int tmp2 = f_n - beta;
		f = max(tmp1, tmp2);
		
		tmp1 = h_w - alpha;
		tmp2 = e_w - beta;
		e = max(tmp1, tmp2);
		
		int idxSeqA = curDiagIdx - threadIdx.x - 1; 
		int idxSeqB = threadIdx.x;

		tmp1 = ASEQ128(idxSeqA) - 60;
		tmp2 = BSEQ128(idxSeqB) - 60;

		h = h_nw + PAM(tmp1, tmp2); 
		h = max(0, h); h = max(e, h); h = max(f, h); 
	
// 		printf("tid=%u diag=%u %c-%c res=%d\t\t", threadIdx.x, curDiagIdx, ASEQ128(idxSeqA), BSEQ128(idxSeqB), PAM(pos1, pos2));
}


__device__ void loadPAM_64threads(const int &tid)
{
	// read the substitution matrix
	unsigned idxp = tid/32, restp = tid%32;

	PAM(idxp, restp)					= blosum50[idxp][restp];		//riga 0,1
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 2,3
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 4,5
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 6,7
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 8,9
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 10,11
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 12,13
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 14,15
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 16,17
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 18,19
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 20,21
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 22,23
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 24,25
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 26,27
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 28,29
	idxp += 2; PAM(idxp, restp)         = blosum50[idxp][restp];		//riga 30,31
}


// finds the max of 128 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 127)
__device__ void max128( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		__syncthreads();

		//if tid % 4 == 0 => tid*2
		//if tid % 4 != 0 => (tid/4)*8 + tid%4
		bool bpv = tid % 4; int ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 4) * 8 + tid % 4);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+4));
		__syncthreads();

		bpv = tid % 8; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 8) * 16 + tid % 8);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+8));
		__syncthreads();

		bpv = tid % 16; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 16) * 32 + tid % 16);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+16));
		__syncthreads();

		bpv = tid % 32; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 32) * 64 + tid % 32);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+32));
		__syncthreads();

// 		bpv = tid % 64; ipv = bpv;
// 		idx = (1 - ipv) * tid * 2    +   (ipv)*( (tid / 64) * 128 + tid % 64);
// 		SCORES(idx) = max( SCORES(idx), SCORES(idx+64));
// 		__syncthreads();

		result = max( result, SCORES(64));
		result = max( result, SCORES(0));
}

// finds the max of 64 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 63)
// needed 128 elements in SCORES(i) but only the meaningful must be initialised
__device__ void max64( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		__syncthreads();

		//if tid % 4 == 0 => tid*2
		//if tid % 4 != 0 => (tid/4)*8 + tid%4
		bool bpv = tid % 4; int ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 4) * 8 + tid % 4);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+4));
		__syncthreads();

//printf("tid=%u ipv=%d idx=%d,\t", tid, ipv, idx);

		bpv = tid % 8; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 8) * 16 + tid % 8);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+8));
		__syncthreads();

		bpv = tid % 16; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 16) * 32 + tid % 16);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+16));
		__syncthreads();

		//bpv = tid % 32; ipv = bpv;
		//idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 32) * 64 + tid % 32);
		//SCORES(idx) = max( SCORES(idx), SCORES(idx+32));
		//__syncthreads();
		result = max( result, SCORES(32));

		result = max( result, SCORES(0));
}

// finds the max of 32 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 31)
// needed 128 elements in SCORES(i) but only the meaningful must be initialised
__device__ void max32( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		__syncthreads();

		//if tid % 4 == 0 => tid*2
		//if tid % 4 != 0 => (tid/4)*8 + tid%4
		bool bpv = tid % 4; int ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 4) * 8 + tid % 4);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+4));
		__syncthreads();

		bpv = tid % 8; ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 8) * 16 + tid % 8);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+8));
		__syncthreads();

		//bpv = tid % 16; ipv = bpv;
		//idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 16) * 32 + tid % 16);
		//SCORES(idx) = max( SCORES(idx), SCORES(idx+16));
		//__syncthreads();

		result = max( result, SCORES(16));

		result = max( result, SCORES(0));
}


// finds the max of 16 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 15)
// needed 128 elements in SCORES(i) but only the meaningful must be initialised
__device__ void max16( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		__syncthreads();

		//if tid % 4 == 0 => tid*2
		//if tid % 4 != 0 => (tid/4)*8 + tid%4
		bool bpv = tid % 4; int ipv = bpv;
		idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 4) * 8 + tid % 4);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+4));
		__syncthreads();

		//bpv = tid % 8; ipv = bpv;
		//idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 8) * 16 + tid % 8);
		//SCORES(idx) = max( SCORES(idx), SCORES(idx+8));
		//__syncthreads();

		result = max( result, SCORES(8));

		result = max( result, SCORES(0));
}


// finds the max of 8 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 7)
// needed 128 elements in SCORES(i) but only the meaningful must be initialised
__device__ void max8( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		__syncthreads();

		//if tid % 4 == 0 => tid*2
		//if tid % 4 != 0 => (tid/4)*8 + tid%4
		//bool bpv = tid % 4; int ipv = bpv;
		//idx = (1 - ipv) * tid * 2    +    (ipv)*( (tid / 4) * 8 + tid % 4);
		//SCORES(idx) = max( SCORES(idx), SCORES(idx+4));
		//__syncthreads();

		result = max( result, SCORES(4));

		result = max( result, SCORES(0));
}

// finds the max of 4 elements using 64 threads ( 0 <= tid <= 63 )
// the input must be already set into SCORES(0 to 3)
// needed 128 elements in SCORES(i) but only the meaningful must be initialised
__device__ void max4( const int &tid, int &result )
{
		int idx = tid*2;
		SCORES(idx) = max( SCORES(idx), SCORES(idx+1));
		__syncthreads();

		// if tid is even => tid *2 
		// if tid is odd => (tid-1)*2+1
		//idx = (1 - tid % 2) * tid * 2    +    (tid % 2)*( (tid-1) * 2 + 1);
		//SCORES(idx) = max( SCORES(idx), SCORES(idx+2));
		//__syncthreads();

		result = max( result, SCORES(2));

		result = max( result, SCORES(0));
}

#endif


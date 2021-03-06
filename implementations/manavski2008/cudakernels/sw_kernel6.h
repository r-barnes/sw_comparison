
#ifndef _TEMPLATE_KERNEL6_H_
#define _TEMPLATE_KERNEL6_H_

#define HSH( index )		CUT_BANK_CHECKER(Hsh, (index))
#define FSH( index )		CUT_BANK_CHECKER(Fsh, (index))
#define ASEQ_PIECE( index )	CUT_BANK_CHECKER(Aseq_Piece, (index))

extern  __shared__  short int int_sharedspace_6[];

#define A_SEQ_CHAR 8
#define LOCAL_MEMORY 2052

// contiene: int pam[32][32]
#define DIMSHAREDSPACE_6		((1024)*sizeof(short int))

#define PAM_6( mac_i, mac_j )	CUT_BANK_CHECKER(int_sharedspace_6, ((mac_i)*32 + (mac_j)))

texture<unsigned char, 1, cudaReadModeElementType> texB6;


__device__ void loadPAM_6_64threads(const int &tid)
{
	// read the substitution matrix
	unsigned idxp = tid/32, restp = tid%32;

	PAM_6(idxp, restp)				= blosum50_6[(idxp*32+restp)];	//riga 0,1
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 2,3
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 4,5
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 6,7
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 8,9
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 10,11
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 12,13
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 14,15
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 16,17
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 18,19
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 20,21
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 22,23
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 24,25
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 26,27
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 28,29
	idxp += 2; PAM_6(idxp, restp)	= blosum50_6[(idxp*32+restp)];	//riga 30,31
}

__device__ void swcalc_kernel_6(const char &aElem, const char &bElem, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const unsigned int &alpha, const unsigned int &beta, int &h, int &e, int &f)
{
	int tmp1 = h_n - alpha;
	int tmp2 = f_n - beta;
	f = max(tmp1, tmp2);
	
	tmp1 = h_w - alpha;
	tmp2 = e_w - beta;
	e = max(tmp1, tmp2);
	
	tmp1 = aElem - 60;
	tmp2 = bElem - 60;

	h = h_nw + PAM_6(tmp1, tmp2);
	h = max(0, h);
	h = max(e, h);
	h = max(f, h);
}


__global__ void sw_kernel6( const char* g_strToAlign, const unsigned sizeNotPad, const char* g_seqlib, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_scores) {

	const unsigned int MAX_NUM_THREADS = 32;

	const unsigned numThreads = blockDim.x;

	// shared memory
	//ho bisogno di una riga di E e di una riga di H per ogni sequenza
	__shared__  short int	Hsh[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];
	__shared__  short int	Fsh[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];
	__shared__  char		Aseq_Piece[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// read the substitution matrix
	unsigned elemPerThread = MAX_NUM_THREADS / numThreads;
	for (unsigned cnt=0; cnt<elemPerThread; cnt++) {
		loadPAM_6_64threads(cnt + (tid*elemPerThread));
		loadPAM_6_64threads(cnt + (tid*elemPerThread) + MAX_NUM_THREADS);
	}

	//numero di sequenza
	unsigned seqNum = (blid*MAX_NUM_THREADS) + tid + seqOffset;

	//offset di tale sequenza nel DB
	unsigned libOffset = g_offsets[seqNum];

	unsigned sizeA = g_sizes[seqNum];
	unsigned sizeB = sizeNotPad;

	__syncthreads();

	unsigned rowOffset = tid * (A_SEQ_CHAR+2);

	unsigned qt = sizeA / A_SEQ_CHAR;
	unsigned rem = sizeA % A_SEQ_CHAR;

	//variabili locali
	int score = 0;
	short int H[LOCAL_MEMORY];
	short int E[LOCAL_MEMORY];
	short int F[LOCAL_MEMORY];

	int f=0, e=0, h=0;
	int h_jprev=0;
	int e_jprev=0;
	int f_jprev=0;
	char bElem;
	unsigned tempVar=0;


	//calcolo per i gruppi da A_SEQ_CHAR caratteri
	///il primo ciclo va all'esterno
	{
		for (unsigned i=0; i<(A_SEQ_CHAR); i++) {
			tempVar = rowOffset + i;
			HSH(tempVar) 			= 0;
			FSH(tempVar) 			= 0;
			ASEQ_PIECE(tempVar)		= g_seqlib[ (libOffset + i) ];
		}
	
		unsigned maxFirstStep = ( sizeA > A_SEQ_CHAR ) ? A_SEQ_CHAR : sizeA;
		for (unsigned i=1; i<sizeB; i++) { 
			
			//taking the single element of the compared sequence
			int x = i;
			bElem = tex1Dfetch(texB6, x);

			h_jprev = 0;
			e_jprev = 0;
			f_jprev = 0;

			unsigned j = 1;
			for (; j<maxFirstStep; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
				score = max(score, h);
			
				tempVar--;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				h_jprev = h;
				f_jprev = f;
				e_jprev = e;
	
			}
			tempVar = rowOffset + j - 1;
			HSH(tempVar) = h_jprev;
			FSH(tempVar) = f_jprev;
			H[i] = h_jprev;
			F[i] = f_jprev;
			E[i] = e_jprev;

		}
	}

	if ( sizeA > A_SEQ_CHAR ) {

		///DAL SECONDO CICLO IN POI
		for (unsigned cnt=1; cnt < qt; cnt++) {
	
			for (unsigned i=0; i<(A_SEQ_CHAR+1); i++) {
				tempVar = rowOffset + i;
				HSH(tempVar) 			= 0;
				FSH(tempVar) 			= 0;
				ASEQ_PIECE(tempVar)	= g_seqlib[ (libOffset + (cnt*(A_SEQ_CHAR)-1) + i) ];
			}
		
			for (unsigned i=1; i<sizeB; i++) { 
				
				//taking the single element of the compared sequence
				int x = i;
				bElem = tex1Dfetch(texB6, x);
		
				h_jprev = H[i];
				e_jprev = E[i];
				f_jprev = F[i];
	
				unsigned j = 1;
				for (; j<A_SEQ_CHAR+1; j++ ) {
					
					tempVar = rowOffset + j;
					
					swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
		
					score = max(score, h);
				
					tempVar--;
					HSH(tempVar) = h_jprev;
					FSH(tempVar) = f_jprev;
					h_jprev = h;
					f_jprev = f;
					e_jprev = e;
		
				}
				tempVar = rowOffset + j - 1;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				H[i] = h_jprev;
				F[i] = f_jprev;
				E[i] = e_jprev;
			}
		}
	
		///calcolo per il residuo
		bool off = qt;
		for (unsigned i=0; i<rem+off; i++) {
			tempVar = rowOffset + i;
			HSH(tempVar) 			= 0;
			FSH(tempVar) 			= 0;
			ASEQ_PIECE(tempVar)	= g_seqlib[ (libOffset + (qt*(A_SEQ_CHAR)-off) + i) ];
		}
		
		for (unsigned i=1; i<sizeB; i++) { 
			
			//taking the single element of the compared sequence
			int x = i;
			bElem = tex1Dfetch(texB6, x);
			
			h_jprev = H[i];
			e_jprev = E[i];
			f_jprev = F[i];
	
			unsigned j = 1;
			for (; j<rem+off; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
				score = max(score, h);
			
				tempVar--;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				h_jprev = h;
				f_jprev = f;
				e_jprev = e;
	
			}
			tempVar = rowOffset + j - 1;
			HSH(tempVar) = h_jprev;
			FSH(tempVar) = f_jprev;
			H[i] = h_jprev;
			F[i] = f_jprev;
			E[i] = e_jprev;
		}

	}

	// write data to global memory
	g_scores[ blid*MAX_NUM_THREADS + seqOffset + tid ] = score;
}


__global__ void sw_kernel6_with_endpos( const char* g_strToAlign, const unsigned sizeNotPad, const char* g_seqlib, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_scores, unsigned *dev_endpos) {

	const unsigned int MAX_NUM_THREADS = 32;

	const unsigned numThreads = blockDim.x;

	// shared memory
	//ho bisogno di una riga di E e di una riga di H per ogni sequenza
	__shared__  short int	Hsh[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];
	__shared__  short int	Fsh[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];
	__shared__  char		Aseq_Piece[MAX_NUM_THREADS*(A_SEQ_CHAR+2)];

	// access thread id and block id
	const unsigned int tid = threadIdx.x;
	const unsigned int blid = blockIdx.x;

	// read the substitution matrix
	unsigned elemPerThread = MAX_NUM_THREADS / numThreads;
	for (unsigned cnt=0; cnt<elemPerThread; cnt++) {
		loadPAM_6_64threads(cnt + (tid*elemPerThread));
		loadPAM_6_64threads(cnt + (tid*elemPerThread) + MAX_NUM_THREADS);
	}

	//numero di sequenza
	unsigned seqNum = (blid*MAX_NUM_THREADS) + tid + seqOffset;

	//offset di tale sequenza nel DB
	unsigned libOffset = g_offsets[seqNum];

	unsigned sizeA = g_sizes[seqNum];
	unsigned sizeB = sizeNotPad;

	__syncthreads();

	unsigned rowOffset = tid * (A_SEQ_CHAR+2);

	unsigned qt = sizeA / A_SEQ_CHAR;
	unsigned rem = sizeA % A_SEQ_CHAR;

	//variabili locali
	int score = 0;
	unsigned short q_endpos = 0;
	unsigned short s_endpos = 0;

	short int H[LOCAL_MEMORY];
	short int E[LOCAL_MEMORY];
	short int F[LOCAL_MEMORY];

	int f=0, e=0, h=0;
	int h_jprev=0;
	int e_jprev=0;
	int f_jprev=0;
	char bElem;
	unsigned tempVar=0;


	//calcolo per i gruppi da A_SEQ_CHAR caratteri
	///il primo ciclo va all'esterno
	{
		for (unsigned i=0; i<(A_SEQ_CHAR); i++) {
			tempVar = rowOffset + i;
			HSH(tempVar) 			= 0;
			FSH(tempVar) 			= 0;
			ASEQ_PIECE(tempVar)		= g_seqlib[ (libOffset + i) ];
		}
	
		unsigned maxFirstStep = ( sizeA > A_SEQ_CHAR ) ? A_SEQ_CHAR : sizeA;
		for (unsigned i=1; i<sizeB; i++) { 
			
			//taking the single element of the compared sequence
			int x = i;
			bElem = tex1Dfetch(texB6, x);

			h_jprev = 0;
			e_jprev = 0;
			f_jprev = 0;

			unsigned j = 1;
			for (; j<maxFirstStep; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
				if (h>score) {
					score = h;
					q_endpos = x;
					s_endpos = j;
				}
			
				tempVar--;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				h_jprev = h;
				f_jprev = f;
				e_jprev = e;
	
			}
			tempVar = rowOffset + j - 1;
			HSH(tempVar) = h_jprev;
			FSH(tempVar) = f_jprev;
			H[i] = h_jprev;
			F[i] = f_jprev;
			E[i] = e_jprev;

		}
	}

	if ( sizeA > A_SEQ_CHAR ) {

		///DAL SECONDO CICLO IN POI
		for (unsigned cnt=1; cnt < qt; cnt++) {
	
			for (unsigned i=0; i<(A_SEQ_CHAR+1); i++) {
				tempVar = rowOffset + i;
				HSH(tempVar) 			= 0;
				FSH(tempVar) 			= 0;
				ASEQ_PIECE(tempVar)	= g_seqlib[ (libOffset + (cnt*(A_SEQ_CHAR)-1) + i) ];
			}
		
			for (unsigned i=1; i<sizeB; i++) { 
				
				//taking the single element of the compared sequence
				int x = i;
				bElem = tex1Dfetch(texB6, x);
		
				h_jprev = H[i];
				e_jprev = E[i];
				f_jprev = F[i];
	
				unsigned j = 1;
				for (; j<A_SEQ_CHAR+1; j++ ) {
					
					tempVar = rowOffset + j;
					
					swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
		
					if (h>score) {
						score = h;
						q_endpos = x;
						s_endpos = (cnt*(A_SEQ_CHAR)-1) + j;
					}
				
					tempVar--;
					HSH(tempVar) = h_jprev;
					FSH(tempVar) = f_jprev;
					h_jprev = h;
					f_jprev = f;
					e_jprev = e;
		
				}
				tempVar = rowOffset + j - 1;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				H[i] = h_jprev;
				F[i] = f_jprev;
				E[i] = e_jprev;
			}
		}
	
		///calcolo per il residuo
		bool off = qt;
		for (unsigned i=0; i<rem+off; i++) {
			tempVar = rowOffset + i;
			HSH(tempVar) 			= 0;
			FSH(tempVar) 			= 0;
			ASEQ_PIECE(tempVar)	= g_seqlib[ (libOffset + (qt*(A_SEQ_CHAR)-off) + i) ];
		}
		
		for (unsigned i=1; i<sizeB; i++) { 
			
			//taking the single element of the compared sequence
			int x = i;
			bElem = tex1Dfetch(texB6, x);
			
			h_jprev = H[i];
			e_jprev = E[i];
			f_jprev = F[i];
	
			unsigned j = 1;
			for (; j<rem+off; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_6(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
				if (h>score) {
					score = h;
					q_endpos = x;
					s_endpos = qt*A_SEQ_CHAR-off + j;
				}

				tempVar--;
				HSH(tempVar) = h_jprev;
				FSH(tempVar) = f_jprev;
				h_jprev = h;
				f_jprev = f;
				e_jprev = e;
	
			}
			tempVar = rowOffset + j - 1;
			HSH(tempVar) = h_jprev;
			FSH(tempVar) = f_jprev;
			H[i] = h_jprev;
			F[i] = f_jprev;
			E[i] = e_jprev;
		}

	}

	unsigned endpos = s_endpos;
	endpos = (endpos<<16) + q_endpos;
	
	// write data to global memory
	g_scores[ blid*MAX_NUM_THREADS + seqOffset + tid ] = score;
	dev_endpos[ blid*MAX_NUM_THREADS + seqOffset + tid ] = endpos;
}



#endif

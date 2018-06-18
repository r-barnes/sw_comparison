
#ifndef _TEMPLATE_KERNEL7_GLOBAL_H_
#define _TEMPLATE_KERNEL7_GLOBAL_H_

#define HSH( index )		CUT_BANK_CHECKER(Hsh, (index))
#define FSH( index )		CUT_BANK_CHECKER(Fsh, (index))
#define ASEQ_PIECE( index )	CUT_BANK_CHECKER(Aseq_Piece, (index))

extern  __shared__  short int int_sharedspace_7[];

#define A_SEQ_CHAR 8

// contiene: int pam[32][32]
#define DIMSHAREDSPACE_7		((1024)*sizeof(short int))

#define PAM_7( mac_i, mac_j )	CUT_BANK_CHECKER(int_sharedspace_7, ((mac_i)*32 + (mac_j)))

texture<unsigned char, 1, cudaReadModeElementType> texB7;


__device__ void loadPAM_7_64threads(const int &tid)
{
	// read the substitution matrix
	unsigned idxp = tid/32, restp = tid%32;

	PAM_7(idxp, restp)				= blosum50[idxp][restp];	//riga 0,1
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 2,3
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 4,5
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 6,7
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 8,9
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 10,11
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 12,13
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 14,15
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 16,17
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 18,19
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 20,21
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 22,23
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 24,25
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 26,27
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 28,29
	idxp += 2; PAM_7(idxp, restp)	= blosum50[idxp][restp];	//riga 30,31
}

__device__ void swcalc_kernel_7(const char &aElem, const char &bElem, const int &h_n, const int &f_n, const int &h_w, const int &e_w, const int &h_nw, const unsigned int &alpha, const unsigned int &beta, int &h, int &e, int &f)
{
	int tmp1 = h_n - alpha;
	int tmp2 = f_n - beta;
	f = max(tmp1, tmp2);
	
	tmp1 = h_w - alpha;
	tmp2 = e_w - beta;
	e = max(tmp1, tmp2);
	
	tmp1 = aElem - 60;
	tmp2 = bElem - 60;

	h = h_nw + PAM_7(tmp1, tmp2);
	h = max(0, h);
	h = max(e, h);
	h = max(f, h);
}


__global__ void sw_kernel7_global( const char* g_strToAlign, const unsigned sizeNotPad, const char* g_seqlib, unsigned seqOffset, unsigned *g_offsets, unsigned *g_sizes, unsigned alpha, unsigned beta, int *g_scores, int *d_colMemory) {

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
	if (tid<2) {
		for (unsigned cnt=0; cnt<MAX_NUM_THREADS; cnt++) {
			loadPAM_7_64threads(cnt + (MAX_NUM_THREADS*tid));
		}
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

	int f=0, e=0, h=0;
	int h_jprev=0;
	int e_jprev=0;
	int f_jprev=0;
	char bElem;
	unsigned tempVar=0;
	unsigned memOffset = ( blid*MAX_NUM_THREADS + tid ) * 6150;

	//calcolo per i gruppi da A_SEQ_CHAR caratteri
	///il primo ciclo va all'esterno
	{
		for (unsigned i=0; i<(A_SEQ_CHAR); i++) {
			tempVar = rowOffset + i;
			HSH(tempVar) 			= 0;
			FSH(tempVar) 			= 0;
			ASEQ_PIECE(tempVar)		= g_seqlib[ (libOffset + i) ];
		}
	
		for (unsigned i=1; i<sizeB; i++) { 
			
			//taking the single element of the compared sequence
			int x = i;
			bElem = texfetch(texB7, x);
	
			h_jprev = 0;
			e_jprev = 0;
			f_jprev = 0;

			unsigned j = 1;
			for (; j<A_SEQ_CHAR; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_7(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
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

			//H
			d_colMemory[memOffset + i] = h_jprev;
			//F
			d_colMemory[memOffset + 2050 + i] = f_jprev;
			//E
			d_colMemory[memOffset + 4100 + i] = e_jprev;
		}
	}
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
			bElem = texfetch(texB7, x);
	
			h_jprev = d_colMemory[memOffset + i];
			f_jprev = d_colMemory[memOffset + 2050 + i];
			e_jprev = d_colMemory[memOffset + 4100 + i];

			unsigned j = 1;
			for (; j<A_SEQ_CHAR+1; j++ ) {
				
				tempVar = rowOffset + j;
				
				swcalc_kernel_7(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);
	
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
			d_colMemory[memOffset + i] = h_jprev;
			d_colMemory[memOffset + 2050 + i] = f_jprev;
			d_colMemory[memOffset + 4100 + i] = e_jprev;
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
		bElem = texfetch(texB7, x);
		
		h_jprev = d_colMemory[memOffset + i];
		f_jprev = d_colMemory[memOffset + 2050 + i];
		e_jprev = d_colMemory[memOffset + 4100 + i];

		unsigned j = 1;
		for (; j<rem+off; j++ ) {
			
			tempVar = rowOffset + j;
			
			swcalc_kernel_7(ASEQ_PIECE(tempVar), bElem, HSH(tempVar), FSH(tempVar), h_jprev, e_jprev, HSH(tempVar-1), alpha, beta, h, e, f);

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
		d_colMemory[memOffset + i] = h_jprev;
		d_colMemory[memOffset + 2050 + i] = f_jprev;
		d_colMemory[memOffset + 4100 + i] = e_jprev;
	}

	// write data to global memory
	g_scores[ blid*MAX_NUM_THREADS + seqOffset + tid ] = score;
}

#endif

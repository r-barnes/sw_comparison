


/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <smithwaterman_kernel.cu>
#include <sw_kernel2.cu>
#include <sw_kernel3.cu>


#define ALLDB_MAX_DIAG_LEN 64
#define MIN_BLOCK_SIZE 64

////////////////////////////////////////////////////////////////////////////////
// seqlib DEVE essere ordinata per la lunghezza delle sequenze
// il handler ammette che sizes[i]<=sizes[i+1]
////////////////////////////////////////////////////////////////////////////////
extern "C" double smithWatermanCuda( const char* strToAlign, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, int* h_scores) 
{

	CUT_CHECK_DEVICE();

	// allocate device memory
	char* d_strToAlign;
	unsigned strToAlignSize = strlen(strToAlign);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_strToAlign, strToAlignSize+1 ) );
	char* d_seqlib;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_seqlib, totBytesUsed));
	unsigned *d_offsets;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_offsets, numSeqs*sizeof(unsigned)) );
	unsigned *d_sizes;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_sizes, numSeqs*sizeof(unsigned)) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( d_strToAlign, strToAlign, strToAlignSize+1, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( d_seqlib, seqlib, totBytesUsed, cudaMemcpyHostToDevice) );
	unsigned arrSizes = numSeqs*sizeof(unsigned);
	CUDA_SAFE_CALL( cudaMemcpy( d_offsets, offsets, arrSizes, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( d_sizes, sizes, arrSizes, cudaMemcpyHostToDevice) );

	// allocate device memory for result
	int* d_Hdata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Hdata, numSeqs*sizeof(int)*ALLDB_MAX_DIAG_LEN*2 ));
	int* d_Edata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Edata, numSeqs*sizeof(int)*ALLDB_MAX_DIAG_LEN ));
	int* d_Fdata;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_Fdata, numSeqs*sizeof(int)*ALLDB_MAX_DIAG_LEN ));
	int* d_scores;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_scores, numSeqs*sizeof(int) ));


	unsigned int timer = 0, timerK1=0, timerK2=0, timerK3=0;
	CUT_SAFE_CALL( cutCreateTimer( &timer));
	CUT_SAFE_CALL( cutStartTimer( timer));

	const unsigned GRID_SIZE = 1000;
	unsigned numK1=0, numK2=0, numK3=0;
	
	CUT_SAFE_CALL( cutCreateTimer( &timerK1));
	CUT_SAFE_CALL( cutStartTimer( timerK1));

	// run del kernel per trovare le prime ~64 antidiagonali per TUTTE le sequenze in seqlib
	for ( unsigned j=startPos; j <= stopPos; ) {
		unsigned nextBlocks = ( j+GRID_SIZE > stopPos + 1 ) ? (stopPos - j + 1) : GRID_SIZE;
		
		dim3  grid( nextBlocks, 1, 1);
		dim3  threads( MIN_BLOCK_SIZE, 1, 1);

		smithwatermanKernel_first65antidiag<<< grid, threads, DIMSHAREDSPACE1_2 >>>( d_strToAlign, d_seqlib, totBytesUsed, numSeqs, j, d_offsets, d_sizes, alpha, beta, d_Hdata, d_Edata, d_Fdata, d_scores); 
		CUT_CHECK_ERROR("Kernel execution failed");

		j += nextBlocks;
		numK1++;
	}

	CUT_SAFE_CALL( cutStopTimer( timerK1));

	// indice delle sequenze già pronte a partire da 0 ( = "nessuna pronta" 
	// si intende per pronte "quelle finite escludendo le ultime ~MIN_BLOCK_SIZE diagonali"
	int lastReadyIndex = startPos - 1;

	int stopPosInt = stopPos;

	for (;lastReadyIndex < stopPosInt && sizes[lastReadyIndex+1] <= MIN_BLOCK_SIZE+1;) 
		++lastReadyIndex;

	CUT_SAFE_CALL( cutCreateTimer( &timerK3));
	CUT_SAFE_CALL( cutStartTimer( timerK3));

	while ( lastReadyIndex < stopPosInt ) {	
		// segue elaborazione di tutte le sequenze del DB più lunghe di MIN_BLOCK_SIZE+1

		if (strlen(strToAlign) <= (MIN_BLOCK_SIZE+1) ) {
			// la sequenza cercata e' di MIN_BLOCK_SIZE caratteri effettivi
			
			//questo contatore permette di ottenere l'offset usato dal 3 kernel per la lettura di A
			unsigned cicleCnt = 1;

			unsigned residueBlockSize = GRID_SIZE;
			unsigned runSteps = 2;

			if (lastReadyIndex + residueBlockSize > stopPos + 1) residueBlockSize = stopPos - lastReadyIndex;

			while ( residueBlockSize > 0) {

				unsigned forwardRunSteps = ( (sizes[(lastReadyIndex + 1)] - 1) / MIN_BLOCK_SIZE ) - runSteps + 1;

				unsigned dimSteps = (sizes[(lastReadyIndex + 1)] - 1) / MIN_BLOCK_SIZE;

				dim3  grid( residueBlockSize, 1, 1);
				dim3  threads( MIN_BLOCK_SIZE, 1, 1);
				smithwatermanKernel_midantidiag_64threads<<< grid, threads, DIMSHAREDSPACE3 >>>( d_strToAlign, strlen(strToAlign), d_seqlib, totBytesUsed, numSeqs, lastReadyIndex+1, d_offsets, d_sizes, alpha, beta, d_Hdata, d_Edata, d_Fdata, d_scores, cicleCnt, forwardRunSteps);
				CUT_CHECK_ERROR("Kernel execution failed");

				for (;residueBlockSize>0 && sizes[lastReadyIndex+1] <= ( dimSteps*MIN_BLOCK_SIZE + 1) ;) {
					++lastReadyIndex; --residueBlockSize;
				}

				cicleCnt += forwardRunSteps;
				
				runSteps += forwardRunSteps;

				numK3++;
			}
		}

	}

	CUT_SAFE_CALL( cutStopTimer( timerK3));

	CUT_SAFE_CALL( cutCreateTimer( &timerK2));
	CUT_SAFE_CALL( cutStartTimer( timerK2));

	// run del kernel per trovare le ultime MIN_BLOCK_SIZE-1 antidiagonali per TUTTE le sequenze in seqlib
	for ( unsigned j=startPos; j <= stopPos; ) {
		unsigned nextBlocks = ( j+GRID_SIZE > stopPos + 1 ) ? (stopPos - j + 1) : GRID_SIZE;
		
		dim3  grid( nextBlocks, 1, 1);
		dim3  threads( MIN_BLOCK_SIZE, 1, 1);
		smithwatermanKernel_last63antidiag<<< grid, threads, DIMSHAREDSPACE1_2 >>>( d_strToAlign, d_seqlib, totBytesUsed, numSeqs, j, d_offsets, d_sizes, alpha, beta, d_Hdata, d_Edata, d_Fdata, d_scores);
	
		CUT_CHECK_ERROR("Kernel execution failed");

		j += nextBlocks;
		numK2++;
	}

	CUT_SAFE_CALL( cutStopTimer( timerK2));

	CUT_SAFE_CALL( cutStopTimer( timer));
	printf( "\nTotal processing time: %f (ms)\n", cutGetTimerValue( timer ));

	double timerExt = cutGetTimerValue( timer );

	CUT_SAFE_CALL( cutDeleteTimer( timer));
	printf( "\nProcessing time Kernel 1: %f (ms)\n", cutGetTimerValue( timerK1 ));
	CUT_SAFE_CALL( cutDeleteTimer( timerK1));
	printf( "\nProcessing time Kernel 2: %f (ms)\n", cutGetTimerValue( timerK2 ));
	CUT_SAFE_CALL( cutDeleteTimer( timerK2));
	printf( "\nProcessing time Kernel 3: %f (ms)\n", cutGetTimerValue( timerK3 ));
	CUT_SAFE_CALL( cutDeleteTimer( timerK3));
	printf("Numero di chiamate al Kernel 1 = %u\n", numK1);
	printf("Numero di chiamate al Kernel 3 = %u\n", numK3);
	printf("Numero di chiamate al Kernel 2 = %u\n", numK2);

	unsigned numSeqsEff = stopPos - startPos + 1;

	// allocate mem for the result on host side
	int *h_Hdata = (int*) malloc( numSeqs*ALLDB_MAX_DIAG_LEN*sizeof(int)*2 );
	int *h_Edata = (int*) malloc( numSeqs*ALLDB_MAX_DIAG_LEN*sizeof(int) );
	int *h_Fdata = (int*) malloc( numSeqs*ALLDB_MAX_DIAG_LEN*sizeof(int) );
	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( h_Hdata, d_Hdata, numSeqs*sizeof( int ) * ALLDB_MAX_DIAG_LEN * 2, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy( h_Edata, d_Edata, numSeqs*sizeof( int ) * ALLDB_MAX_DIAG_LEN, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy( h_Fdata, d_Fdata, numSeqs*sizeof( int ) * ALLDB_MAX_DIAG_LEN, cudaMemcpyDeviceToHost) );
	//CUDA_SAFE_CALL( cudaMemcpy( h_scores, d_scores, numSeqs*sizeof( int ) , cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );


/*
	for (unsigned cnt = 1; cnt < numSeqs; ++cnt) {
			//printf("\n\n#################################Sequence: %u#################################\n", cnt);
			printf("H values:\n");
			for(unsigned j=0; j<ALLDB_MAX_DIAG_LEN; ++j){
				printf("H[%u]=%d, " , j, h_Hdata[ (cnt*2)*ALLDB_MAX_DIAG_LEN + j]);
				//printf("H[%u]=%d, " , j, h_Hdata[ ((numSeqs-1)*2)*ALLDB_MAX_DIAG_LEN + j]);
			}
		
			printf("\nHprev values:\n");
			for(unsigned j=0; j<ALLDB_MAX_DIAG_LEN; ++j){
				printf("Hprev[%u]=%d, " , j, h_Hdata[ (cnt*2)*ALLDB_MAX_DIAG_LEN + ALLDB_MAX_DIAG_LEN + j]);
			}
		
			printf("\nE values:\n");
			for(unsigned j=0; j<ALLDB_MAX_DIAG_LEN; ++j)
				printf("E[%u]=%d, " , j, h_Edata[ cnt*ALLDB_MAX_DIAG_LEN + j]);
				//printf("E[%u]=%d, " , j, h_Edata[ (numSeqs-1)*ALLDB_MAX_DIAG_LEN + j]);
			printf("\nF values:\n");
			for(unsigned j=0; j<ALLDB_MAX_DIAG_LEN; ++j)
				printf("F[%u]=%d, " , j, h_Fdata[ cnt*ALLDB_MAX_DIAG_LEN + j]);
				//printf("F[%u]=%d, " , j, h_Fdata[ (numSeqs-1)*ALLDB_MAX_DIAG_LEN + j]);
			
			printf("\nscores[%u]:\n%d, " , cnt, h_scores[cnt]);
			//printf("\nscores[%u]:\n%d, " , (numSeqs-1), h_scores[(numSeqs-1)]);	
	}


	FILE* outStream2 = fopen("gpuTest.dat", "w");
	for (unsigned cnt=1; cnt<numSeqs; ++cnt) {
			fprintf(outStream2, "seq %u SCORE = %u LAST ELEM = %d\n", cnt, h_scores[cnt], h_Hdata[ (cnt*2)*ALLDB_MAX_DIAG_LEN + 63]);
	}
	fclose(outStream2);

*/

	// cleanup memory
	free(h_Fdata);
	free(h_Edata);
	free(h_Hdata);
	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_Fdata));
	CUDA_SAFE_CALL(cudaFree(d_Edata));
	CUDA_SAFE_CALL(cudaFree(d_Hdata));
	CUDA_SAFE_CALL(cudaFree(d_sizes));
	CUDA_SAFE_CALL(cudaFree(d_offsets));
	CUDA_SAFE_CALL(cudaFree(d_seqlib));
	CUDA_SAFE_CALL(cudaFree(d_strToAlign));

	return timerExt;
}

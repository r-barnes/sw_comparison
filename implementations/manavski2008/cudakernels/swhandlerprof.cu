/***************************************************************************
 *   Copyright (C) 2006                                                    *
 *                                                                         *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <memory.h>

// includes, project
#include <cutil.h>
#include "sbtmatrix.h"
#include "swkernelprof1.h"

#include <QtCore/QTime>


extern "C" void swInitMemProf( const char *seqlib, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, int* h_scores, char **d_seqlib, unsigned **d_offsets, unsigned **d_sizes, int **d_scores) {

	CUDA_SAFE_CALL( cudaMalloc( (void**) d_seqlib, totBytesUsed));
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_offsets, numSeqs*sizeof(unsigned)) );
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_sizes, numSeqs*sizeof(unsigned)) );

	// copy host memory to device
	CUDA_SAFE_CALL( cudaMemcpy( *d_seqlib, seqlib, totBytesUsed, cudaMemcpyHostToDevice) );
	unsigned arrSizes = numSeqs*sizeof(unsigned);
	CUDA_SAFE_CALL( cudaMemcpy( *d_offsets, offsets, arrSizes, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( *d_sizes, sizes, arrSizes, cudaMemcpyHostToDevice) );

	// allocate device memory for result
	CUDA_SAFE_CALL( cudaMalloc( (void**) d_scores, numSeqs*sizeof(int) ));

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( *d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );
}

extern "C" void swCleanMemProf( char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int* d_scores ) {

	CUDA_SAFE_CALL(cudaFree(d_scores));
	CUDA_SAFE_CALL(cudaFree(d_sizes));
	CUDA_SAFE_CALL(cudaFree(d_offsets));
	CUDA_SAFE_CALL(cudaFree(d_seqlib));
}

extern "C" double smithWatermanCudaProf(const char* strToAlign, const unsigned sizeNotPad, const char *seqlib, unsigned startPos, unsigned stopPos, const unsigned numSeqs, const unsigned totBytesUsed, unsigned *offsets, unsigned *sizes, const unsigned alpha, const unsigned beta, const char* subMat, const char *lastSubMat, int* h_scores, char *d_seqlib, unsigned *d_offsets, unsigned *d_sizes, int * &d_scores, unsigned debug) {

//########################################profiling############################################

	const int *blosum;

	//caricamento matrice
	if ( strcmp (subMat,"BL50") == 0 )
		blosum = cpu_abl50;
	else if ( strcmp (subMat,"BL62") == 0 )
		blosum = cpu_abl62;
	else if ( strcmp (subMat,"BL90") == 0 )
		blosum = cpu_abl90;
	else
		blosum = cpu_dna1;

	//gestione padding
	unsigned queryRealSize = sizeNotPad-1;
	unsigned queryMod = queryRealSize%PADDING_FACT;
	unsigned queryPadSize = (queryMod!=0) ? queryRealSize  + (PADDING_FACT-queryMod) : queryRealSize;

	//il 4 che comparirà da qui in avanti è legato al fatto che accorpiamo 4 diversi valori della substitution matrix in un unico unsigned
	unsigned profileSize = 32*(queryPadSize/4);
	
	unsigned *queryProf = (unsigned *) malloc(profileSize*sizeof(unsigned));
	memset(queryProf, 'd', profileSize*sizeof(unsigned));

	unsigned i, j;

	char a, b;
	unsigned pos;

	//profiling
	for (unsigned i=0; i<ALPHA_SIZE; ++i) {

		a = AMINO_ACIDS[i];
		pos = (unsigned)a - 60;
		int temp=0, temp2=0;

		for (unsigned j=0; j<queryPadSize; ++j) {

			if (j > queryRealSize)
				temp = (BIAS) & 0xff;
			else {
				b = strToAlign[j+1];
				temp = (blosum[pos*32 + (b-60)] + BIAS) & 0xff;
			}

			//in questo modo il numero letto per primo finisce nel byte meno significativo
			temp2 += temp << (8*(j%4));

			//if (i==0 && j<4)
				//printf("%d %d\n", temp, temp2);

			if ( (j+1)%4==0 ) {
				queryProf[pos*(queryPadSize/4) + ((j+1)/4 - 1)] = temp2;
				temp2 = 0;
			}
		}
	}

printf("trace\n");
/*
	for (unsigned cnt=0; cnt<profileSize; ++cnt) {
		if (cnt!=0 && cnt%(queryPadSize/4)==0)
			printf("\n");
		printf("%d ", queryProf[cnt]);
	}
*/
//########################################caricamento_profiling############################################

	short unsigned* d_queryProf;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &d_queryProf, profileSize*sizeof(unsigned) ) );
	CUDA_SAFE_CALL( cudaMemcpy( d_queryProf, queryProf, profileSize*sizeof(unsigned), cudaMemcpyHostToDevice) );

	cudaChannelFormatDesc chDesc;
	chDesc.x = 32;
	chDesc.y = 0;
	chDesc.z = 0;
	chDesc.w = 0;
	chDesc.f = cudaChannelFormatKindUnsigned;
	texProf.normalized = false;

	// working well on CUDA 0.8
	//CUDA_SAFE_CALL( cudaBindTexture( "texProf", d_queryProf, &chDesc, (size_t)(profileSize*sizeof(unsigned)), (size_t)0 ) );

	CUDA_SAFE_CALL( cudaBindTexture(0, &texProf, d_queryProf, &chDesc, (size_t)(profileSize*sizeof(unsigned)) ) );

//########################################calcolo_effettivo############################################
	
	unsigned strToAlignSize = strlen(strToAlign);

	//settaggio GRID_SIZE sulla base della sequenza di ingresso
	//per le sequenze più lunghe di 579 in realtà non sono stati condotti test. Perciò si è scelto una soglia di sicurezza pari a 50
	//450 è un valore che per 579 può provocare problemi
	unsigned GRID_SIZE;
	if (strToAlignSize<50)
		GRID_SIZE = 800;
	else if (strToAlignSize<580)
		GRID_SIZE = 450;
	else 
		GRID_SIZE = 50;

	double max_timer_value=0;

	unsigned arrSizes = numSeqs*sizeof(int);

	//inizializzazione di d_scores fatta per evitare confusione nella lettura dei risultati
	CUDA_SAFE_CALL( cudaMemcpy( d_scores, offsets, arrSizes, cudaMemcpyHostToDevice) );

	//numero sequenze effettive
	unsigned numSeqsEff = stopPos - startPos + 1;

	unsigned numTotBlocks = numSeqsEff / MAX_BLOCK_SIZE;
	unsigned residueThreads = numSeqsEff % MAX_BLOCK_SIZE;

	//chiamata per il residuo

	if ( residueThreads != 0 ) {

		dim3  grid( 1, 1, 1);
		dim3  threads( residueThreads, 1, 1);

		QTime timer_krl; timer_krl.start();

		sw_kernelprof1<<<grid, threads>>>(queryPadSize, d_seqlib, startPos, d_offsets, d_sizes, alpha, beta, d_scores);

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;
	}

	//chiamata per multipli di MAX_BLOCK_SIZE
	
	unsigned newStartPos = startPos + residueThreads;
	
	unsigned cnt;
	for (cnt=0; cnt<numTotBlocks;) {

		unsigned numBlocks = (cnt + GRID_SIZE > numTotBlocks) ? (numTotBlocks - cnt) : GRID_SIZE;

		dim3  grid( numBlocks, 1, 1);
		dim3  threads( MAX_BLOCK_SIZE, 1, 1);

		QTime timer_krl; timer_krl.start();

		sw_kernelprof1<<<grid, threads>>>(queryPadSize, d_seqlib, newStartPos, d_offsets, d_sizes, alpha, beta, d_scores);

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;

		cnt += numBlocks;
		newStartPos += numBlocks * MAX_BLOCK_SIZE;
	}

	if (debug != 0)
		printf( "\nMAX TIMER VALUE: %f (ms), NUM BLOCKS: %d\n", max_timer_value, cnt);

	// copy result from device to host
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );

	// cleanup memory
	CUDA_SAFE_CALL(cudaFree(d_queryProf));
	free(queryProf);

	return 0;
}


/***************************************************************************
 *   Copyright (C) 2008                                                    *
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
/**
	@author Svetlin Manavski <svetlin@manavski.com>
*/

#include "handler_solexa.h"
#include "e2g_consts.h"

#include <string>
#include <memory.h>
#include <iostream>
using namespace std;

#include <cutil.h>
#include <cuda_runtime_api.h>

#include <QtCore/QTime>


/** This package alligns short Solexa sequences ( 35 bp ) to genomic data considering splice sites
    The required input is a list of pairs: 1 solexa query and 1 cut of the chromosome to be aligned to
*/

/*
extern "C" void solexa_handler_1( const unsigned gridSize, const unsigned numThreads, const char* d_strToAlign, const char *d_seqlib, unsigned newStartPos, unsigned *d_offsets, unsigned *d_sizes, const char *d_splice_sites, const unsigned dbsize, const short unsigned first_gap_penalty, const short unsigned next_gap_penalty, const short unsigned splice_penalty, const short unsigned intron_penalty, int* d_scores);
*/

GPUHandlerSolexa::GPUHandlerSolexa(unsigned device, const unsigned maxSeqs, const unsigned dbbytes) : myDevice(device), maxSeqsNumber(maxSeqs), maxDBBytes(dbbytes), d_queries(NULL), d_seqlib(NULL), d_splice_sites(NULL), d_offsets(NULL), d_sizes(NULL), d_scores(NULL), h_queries(NULL), h_subjects(NULL), h_splice_sites(NULL), h_offsets(NULL), h_sizes(NULL)
{
	allocMem();
}

GPUHandlerSolexa::~GPUHandlerSolexa()
{
	solexaCleanMem();
}


void GPUHandlerSolexa::allocMem()
{
	h_queries = new char[SOLEXA_QUERY_SIZE*maxSeqsNumber];
	h_subjects = new char[maxDBBytes];
	h_splice_sites = new char[maxDBBytes];
	h_offsets = new unsigned[maxSeqsNumber];
	h_sizes = new unsigned[maxSeqsNumber];
	
	cudaError_t err = cudaMalloc( (void**) &d_queries, SOLEXA_QUERY_SIZE*maxSeqsNumber);
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_queries on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	err = cudaMalloc( (void**) &d_seqlib, maxDBBytes);
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_seqlib on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	err = cudaMalloc( (void**) &d_offsets, maxSeqsNumber*sizeof(unsigned)) ;
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_offsets on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}
	
	err = cudaMalloc( (void**) &d_sizes, maxSeqsNumber*sizeof(unsigned)) ;
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_sizes on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	err = cudaMalloc( (void**) &d_splice_sites, maxDBBytes);
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_splice_sites on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}


	// allocate device memory for result
	err = cudaMalloc( (void**) &d_scores, maxSeqsNumber*sizeof(int) );
	if ( err != cudaSuccess ) {
		string ser =  "cannot allocate enough memory for d_scores on device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

}


void GPUHandlerSolexa::setMem( const unsigned numPairs, const unsigned totBytesUsed )
{
	// copy host memory to device
	cudaError_t err = cudaMemcpy( d_seqlib, h_subjects, totBytesUsed, cudaMemcpyHostToDevice);
	if ( err != cudaSuccess ) {
		string ser =  "cannot copy memory form host to device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	unsigned arrSizes = numPairs*sizeof(unsigned);
	err = cudaMemcpy( d_offsets, h_offsets, arrSizes, cudaMemcpyHostToDevice);
	if ( err != cudaSuccess ) {
		string ser =  "cannot copy memory form host to device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	err = cudaMemcpy( d_sizes, h_sizes, arrSizes, cudaMemcpyHostToDevice);
	if ( err != cudaSuccess ) {
		string ser =  "cannot copy memory form host to device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	// copy host memory to device
	err = cudaMemcpy( d_splice_sites, h_splice_sites, totBytesUsed, cudaMemcpyHostToDevice);
	if ( err != cudaSuccess ) {
		string ser =  "cannot copy splice_sites form host to device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	// copy host memory to device
	err = cudaMemcpy( d_queries, h_queries, numPairs*SOLEXA_QUERY_SIZE, cudaMemcpyHostToDevice);
	if ( err != cudaSuccess ) {
		string ser =  "cannot copy memory form host to device. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

	//init of d_scores to avoid confusing results
	err = cudaMemset( d_scores, 0, arrSizes );
	if ( err != cudaSuccess ) {
		string ser =  "cannot set device memory. " ; ser += cudaGetErrorString(err);
		throw ser;
	}

}

void GPUHandlerSolexa::solexaCleanMem()
{
	delete []h_queries;
	delete []h_subjects;
	delete []h_splice_sites;
	delete []h_offsets;
	delete []h_sizes;

	cudaError_t err = cudaFree(d_scores);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_scores";
		cerr << ser << endl;
	}
	err = cudaFree(d_splice_sites);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_splice_sites";
		cerr << ser << endl;
	}
	err = cudaFree(d_sizes);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_sizes";
		cerr << ser << endl;
	}
	err = cudaFree(d_offsets);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_offsets";
		cerr << ser << endl;
	}
	err = cudaFree(d_seqlib);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_seqlib";
		cerr << ser << endl;
	}
	err = cudaFree(d_queries);
	if ( err != cudaSuccess ) {
		string ser =  "cannot free gpu memory for d_queries";
		cerr << ser << endl;
	}

}


void GPUHandlerSolexa::est2genome(vector<EstAlignPair *> &input_pairs, const short unsigned first_gap_penalty, const short unsigned next_gap_penalty, const short unsigned splice_penalty, const short unsigned intron_penalty, bool debug)
{
	unsigned totBytesUsed;
		
	convertData(input_pairs, totBytesUsed);
	unsigned numPairs = input_pairs.size();
	

	setMem(numPairs, totBytesUsed);

	
	unsigned startPos = 0; 
	unsigned stopPos = numPairs -1;
 
	double max_timer_value=0;

	//numero sequenze effettive
	unsigned numSeqsEff = stopPos - startPos + 1;

	unsigned numTotBlocks = numSeqsEff / MAX_NUM_THREADS;
	unsigned residueThreads = numSeqsEff % MAX_NUM_THREADS;

	//GRID SIZE must be the number of pairs divided by the number of pairs computed in a single thread block
	unsigned GRID_SIZE = numTotBlocks;
	

	//call to compute the residual threads number
	if ( residueThreads )  {

		dim3  grid( 1, 1, 1);
		dim3  threads( residueThreads, 1, 1);

		QTime timer_krl; timer_krl.start();

		// commented to not require compilation
		//solexa_handler_1(1, residueThreads, d_queries, d_seqlib, startPos, d_offsets, d_sizes, d_splice_sites, totBytesUsed, first_gap_penalty, next_gap_penalty, splice_penalty, intron_penalty, d_scores);

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;

	} 
	cout << "residueThreads = " << residueThreads << endl;
	
	//chiamata per multipli di MAX_NUM_THREADS

	unsigned newStartPos = startPos + residueThreads;

	unsigned cnt;
	for (cnt=0; cnt<numTotBlocks;) {

		unsigned numBlocks = (cnt + GRID_SIZE > numTotBlocks) ? (numTotBlocks - cnt) : GRID_SIZE;

		dim3  grid( numBlocks, 1, 1);
		dim3  threads( MAX_NUM_THREADS, 1, 1);

		QTime timer_krl; timer_krl.start();

		// commented to not require compilation
		//solexa_handler_1(numBlocks, MAX_NUM_THREADS, d_queries, d_seqlib, newStartPos, d_offsets, d_sizes, d_splice_sites, totBytesUsed, first_gap_penalty, next_gap_penalty, splice_penalty, intron_penalty,  d_scores);

		int last_time = timer_krl.elapsed();
		max_timer_value = (max_timer_value > last_time) ? max_timer_value : last_time;

		cnt += numBlocks;
		newStartPos += numBlocks * MAX_NUM_THREADS;
	}

	if (debug)
		cout << "\nMAX TIMER VALUE: " << max_timer_value << " (ms), NUM BLOCKS: " << cnt << endl;

	// copy result from device to host
	int *h_scores = new int[numPairs];
	CUDA_SAFE_CALL( cudaMemcpy( h_scores+startPos, d_scores+startPos, numSeqsEff*sizeof( int ) , cudaMemcpyDeviceToHost) );
	
	for(unsigned j=startPos; j<=stopPos; ++j) {
		input_pairs[j]->max_score = *(h_scores+j);
	}

	delete []h_scores;

}


void GPUHandlerSolexa::convertData(vector<EstAlignPair *> &input_pairs, unsigned &totBytesUsed)
{
	unsigned numQueries = input_pairs.size();
	
	if (numQueries > maxSeqsNumber)
		throw string("GPUHandlerSolexa: the number of queries must be smaller than the allocated ones");
	
	//assumes the real max length of the quey is SOLEXA_QUERY_SIZE - 1 
	for (unsigned j=0; j<numQueries; ++j) {
		strncpy( h_queries+SOLEXA_QUERY_SIZE*j, input_pairs[j]->getQuery()->sequenceData(), SOLEXA_QUERY_SIZE );
		h_queries[(SOLEXA_QUERY_SIZE-1)*(j+1)] = 0;
	}

	unsigned j=0;
	for (; j<numQueries; ++j) {
		h_sizes[j] = input_pairs[j]->getSubject()->getSize();
		h_offsets[j] = (j>0) ? (h_offsets[j-1] + h_sizes[j-1] + 1) : 0;
		if ( (h_offsets[j] + h_sizes[j] + 1) < maxDBBytes )
			strcpy( h_subjects+h_offsets[j], input_pairs[j]->getSubject()->sequenceData() );
		else
			throw("GPUHandlerSolexa: size of subjects is larger than allocated one");
		
		// finds splice sites
		estFindSpliceSites( *input_pairs[j] );
		strcpy(h_splice_sites + h_offsets[j], input_pairs[j]->splice_sites.c_str());
	}

	totBytesUsed = h_offsets[j-1] + h_sizes[j-1] + 1;
}





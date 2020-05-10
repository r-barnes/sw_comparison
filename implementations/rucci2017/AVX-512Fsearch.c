#include "AVX-512Fsearch.h"

// Host search using AVX-512 instrucions and Score Profile technique
void search_avx512f_ap (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp, 
	char profile, unsigned short int query_length_threshold, 
	char * vect_db_sequences, unsigned short int * vect_db_sequences_lengths, unsigned short int * vect_db_nbbs, unsigned long int vect_db_sequences_count, unsigned long int * vect_db_sequences_disp,
	__m512i * submat, int open_gap, int extend_gap, int n_threads, int block_width, __m512i * scores, double * workTime){

	long int i, j, k, qp_count, sp_count;
	double tick;

	char *a;
	unsigned int * a_disp, queryProfiles_length;
	unsigned long int * b_disp = NULL;
	unsigned short int * m, *n, *nbbs, query_sequences_max_length; 
	char  *b;
	__m512i * queryProfiles;

	a = query_sequences;
	m = query_sequences_lengths;
	a_disp = query_disp;

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];

	b = vect_db_sequences;
	n = vect_db_sequences_lengths;
	nbbs = vect_db_nbbs;
	b_disp = vect_db_sequences_disp;

	if (profile == QUERY_PROFILE)
		query_length_threshold = query_sequences_max_length+1;
	else 
		if (profile == SCORE_PROFILE)
			query_length_threshold = 0;

	// calculate number of query sequences that are processed with query and score profile
	i = 0;
	while ((i < query_sequences_count) && (query_sequences_lengths[i] < query_length_threshold))
		i++;
	qp_count = i;
	sp_count = query_sequences_count-qp_count;

	// allocate memory for query profiles (if correspond)
	if (qp_count > 0) 
		queryProfiles = (__m512i *)_mm_malloc((a_disp[qp_count])*2*sizeof(__m512i),MEMALIGN);

	tick = dwalltime();

	#pragma omp parallel default(none) shared(block_width, a, b, n, nbbs, m, a_disp, b_disp, submat, scores, query_sequences_count, \
				vect_db_sequences_count, open_gap, extend_gap, query_sequences_max_length, qp_count, sp_count, \
				queryProfiles, query_length_threshold) num_threads(n_threads) 
	{
			char * ptr_a;
			__m512i *row1, *row2, *maxCol, *maxRow, *lastCol, *tmp, *ptr_scores, *bIndexes, *queryProfile,  * scoreProfile,  *ptr_scoreProfile1,  *ptr_scoreProfile2;
			__declspec(align(16)) __m128i* ptr_b, *ptr_b_block;

			__declspec(align(MEMALIGN)) __m512i vzero = _mm512_setzero_epi32(), score, previous, current1, current2, aux1, auxLastCol;
			__declspec(align(MEMALIGN)) __m512i vextend_gap = _mm512_set1_epi32(extend_gap), vopen_extend_gap = _mm512_set1_epi32(open_gap+extend_gap);
			__declspec(align(MEMALIGN)) __m512i v16 = _mm512_set1_epi32(16), submat_hi1, submat_lo1, submat_hi2, submat_lo2, bValues, maxRow1, maxRow2;
			__mmask16 * masks, mask;

			unsigned int i, j, ii, jj, k, disp, dim1, dim2, nbb;
			unsigned long int t, s, q; 

			// allocate memory for auxiliary buffers
			row1 = (__m512i *) _mm_malloc((block_width+1)*sizeof(__m512i),MEMALIGN);
			row2 = (__m512i *) _mm_malloc((block_width+1)*sizeof(__m512i),MEMALIGN);
			maxCol = (__m512i *) _mm_malloc((block_width+1)*sizeof(__m512i),MEMALIGN);
			maxRow = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i),MEMALIGN);
			lastCol = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i),MEMALIGN);

			// allocate memory for SP (if correspond)
			if (query_sequences_max_length >= query_length_threshold)
				scoreProfile = (__m512i *) _mm_malloc(SUBMAT_ROWS*block_width*sizeof(__m512i), MEMALIGN);

			// build query profiles (if correspond)
			if (qp_count > 0) {
				// alloc memory for indexes
				bIndexes = (__m512i *) _mm_malloc((block_width)*sizeof(__m512i),MEMALIGN);
				masks = (__mmask16 *) _mm_malloc((block_width)*sizeof(__mmask16),MEMALIGN);

				#pragma omp for schedule(dynamic)  
				for (i=0; i< a_disp[qp_count] ; i++) {
					queryProfiles[i*2] = submat[a[i]*2];
					queryProfiles[i*2+1] = submat[a[i]*2+1];
				}
			}

			// calculate chunk alignments using query profile technique
			#pragma omp for schedule(dynamic) nowait
			for (t=0; t< qp_count*vect_db_sequences_count; t++) {
	
				q = (qp_count-1) - (t % qp_count);
				s = (vect_db_sequences_count-1) - (t / qp_count);

				queryProfile = queryProfiles + a_disp[q]*2;
				ptr_b = (__m128i*)(b + b_disp[s]);
				ptr_scores = scores + (q*vect_db_sequences_count+s);

				// init buffers
				#pragma unroll(UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
				#pragma unroll(UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
						
				// set score to 0
				score = _mm512_setzero_epi32();

				// calculate number of blocks
				nbb = nbbs[s];

				for (k=0; k < nbb; k++){

					// calculate dim1
					disp = k*block_width;
					dim1 = (block_width < n[s]-disp ? block_width : n[s]-disp);
					// calculate dim2
					dim2 = dim1 / DB_SEQ_LEN_MULT;

					// init buffers
					#pragma unroll(UNROLL_COUNT)
					for (i=1; i<dim1+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
					#pragma unroll(UNROLL_COUNT)
					for (i=0; i<dim1 ; i++ ) row1[i] = _mm512_setzero_epi32();
					auxLastCol = _mm512_setzero_epi32();

					// get bIndexes
					ptr_b_block = ptr_b + disp;
					#pragma unroll(UNROLL_COUNT)
					for (i=0; i<dim1 ; i++ ) {
						bIndexes[i] = _mm512_cvtepi8_epi32(ptr_b_block[i]);
						masks[i] = _mm512_cmpge_epi32_mask(bIndexes[i],v16);
					}

					for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
						
						// update row[0] with lastCol[i-1]
						row1[0] = lastCol[i];
						previous = lastCol[i+1];
						// load submat values corresponding to current a residue
						submat_lo1 = (queryProfile[i*2]);
						submat_hi1 = (queryProfile[i*2+1]);
						submat_lo2 = (queryProfile[(i+1)*2]);
						submat_hi2 = (queryProfile[(i+1)*2+1]);
						// store maxRow in auxiliars
						maxRow1 = maxRow[i];
						maxRow2 = maxRow[i+1];

						for (ii=0; ii<dim2 ; ii++) {

							#pragma unroll(DB_SEQ_LEN_MULT)
							for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT; jj++, j++) {

								//calcuate the diagonal value
								aux1 = _mm512_permutevar_epi32(bIndexes[j-1], submat_lo1);
								aux1 = _mm512_mask_permutevar_epi32(aux1, masks[j-1], bIndexes[j-1], submat_hi1);
								current1 = _mm512_add_epi32(row1[j-1], aux1);								
								// calculate current1 max value
								current1 = _mm512_max_epi32(current1, maxRow1);
								current1 = _mm512_max_epi32(current1, maxCol[j]);
								current1 = _mm512_max_epi32(current1, vzero);
								// update maxRow and maxCol
								maxRow1 = _mm512_sub_epi32(maxRow1, vextend_gap);
								maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap);
								aux1 = _mm512_sub_epi32(current1, vopen_extend_gap);
								maxRow1 = _mm512_max_epi32(maxRow1, aux1);
								maxCol[j] =  _mm512_max_epi32(maxCol[j], aux1);	
								// update max score
								score = _mm512_max_epi32(score,current1);

								//calcuate the diagonal value
								aux1 = _mm512_permutevar_epi32(bIndexes[j-1], submat_lo2);
								aux1 = _mm512_mask_permutevar_epi32(aux1, masks[j-1], bIndexes[j-1], submat_hi2);
								current2 = _mm512_add_epi32(previous, aux1);								
								// update previous
								previous = current1;
								// calculate current2 max value
								current2 = _mm512_max_epi32(current2, maxRow2);
								current2 = _mm512_max_epi32(current2, maxCol[j]);
								current2 = _mm512_max_epi32(current2, vzero);
								// update maxRow and maxCol
								maxRow2 = _mm512_sub_epi32(maxRow2, vextend_gap);
								maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap);
								aux1 = _mm512_sub_epi32(current2, vopen_extend_gap);
								maxRow2 = _mm512_max_epi32(maxRow2, aux1);
								maxCol[j] =  _mm512_max_epi32(maxCol[j], aux1);	
								// update row buffer
								row2[j] = current2;
								// update max score
								score = _mm512_max_epi32(score,current2);
							}
						
						}
						if (k != nbb-1) {
							// update maxRow
							maxRow[i] = maxRow1;
							maxRow[i+1] = maxRow2;
							// update lastCol
							lastCol[i] = auxLastCol;
							lastCol[i+1] = current1;
							auxLastCol = current2;
						}
						// swap buffers
						tmp = row1;
						row1 = row2;
						row2 = tmp;
					}

				}

				// store max value
				_mm512_store_epi32(ptr_scores, score);
			}

			// calculate chunk alignments using score profile technique
			#pragma omp for schedule(dynamic) nowait
			for (t=0; t< sp_count*vect_db_sequences_count; t++) {

				q = qp_count + (sp_count-1) - (t % sp_count);
				s = (vect_db_sequences_count-1) - (t / sp_count);

				ptr_a = a + a_disp[q];
				ptr_b = (__m128i*)(b + b_disp[s]);
				ptr_scores = scores + (q*vect_db_sequences_count+s);

				// init buffers
				#pragma unroll(UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_epi32(); // index 0 is not used
				#pragma unroll(UNROLL_COUNT)
				for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_epi32();
						
				// set score to 0
				score = _mm512_setzero_epi32();

				// calculate number of blocks
				nbb = nbbs[s];

				for (k=0; k < nbb; k++){

					// calculate dim1
					disp = k*block_width;
					dim1 = (block_width < n[s]-disp ? block_width : n[s]-disp);
					// calculate dim2
					dim2 = dim1 / DB_SEQ_LEN_MULT;

					// init buffers
					#pragma unroll(UNROLL_COUNT)
					for (i=1; i<dim1+1 ; i++ ) maxCol[i] = _mm512_setzero_epi32(); //index 0 is not used
					#pragma unroll(UNROLL_COUNT)
					for (i=0; i<dim1 ; i++ ) row1[i] = _mm512_setzero_epi32();
					auxLastCol = _mm512_setzero_epi32();

					// build score profile
					ptr_b_block = ptr_b + disp;
					for (i=0; i< dim1 ;i++ ) {
						bValues = _mm512_cvtepi8_epi32(ptr_b_block[i]);
						mask = _mm512_cmpge_epi32_mask(bValues,v16);
						ptr_scoreProfile1 = scoreProfile + i;
						#pragma unroll
						for (j=0; j< SUBMAT_ROWS; j++) {
							aux1 = _mm512_permutevar_epi32(bValues, (submat[j*2]));
							ptr_scoreProfile1[j*dim1] = _mm512_mask_permutevar_epi32(aux1, mask, bValues, (submat[j*2+1]));
						}
					}

					for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
					
						// update row[0] with lastCol[i-1]
						row1[0] = lastCol[i];
						previous = lastCol[i+1];
						// calculate score profile displacement
						ptr_scoreProfile1 = scoreProfile+ptr_a[i]*dim1;
						ptr_scoreProfile2 = scoreProfile+ptr_a[i+1]*dim1;
						// store maxRow in auxiliars
						maxRow1 = maxRow[i];
						maxRow2 = maxRow[i+1];

						for (ii=0; ii<dim2 ; ii++) {

							#pragma unroll(DB_SEQ_LEN_MULT)
							for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT; jj++, j++) {
								//calcuate the diagonal value
								current1 = _mm512_add_epi32(row1[j-1], (ptr_scoreProfile1[j-1]));								
								// calculate current1 max value
								current1 = _mm512_max_epi32(current1, maxRow1);
								current1 = _mm512_max_epi32(current1, maxCol[j]);
								current1 = _mm512_max_epi32(current1, vzero);
								// update maxRow and maxCol
								maxRow1 = _mm512_sub_epi32(maxRow1, vextend_gap);
								maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap);
								aux1 = _mm512_sub_epi32(current1, vopen_extend_gap);
								maxRow1 = _mm512_max_epi32(maxRow1, aux1);
								maxCol[j] =  _mm512_max_epi32(maxCol[j], aux1);	
								// update max score
								score = _mm512_max_epi32(score,current1);

								//calcuate the diagonal value
								current2 = _mm512_add_epi32(previous, (ptr_scoreProfile2[j-1]));								
								// update previous
								previous = current1;
								// calculate current2 max value
								current2 = _mm512_max_epi32(current2, maxRow2);
								current2 = _mm512_max_epi32(current2, maxCol[j]);
								current2 = _mm512_max_epi32(current2, vzero);
								// update maxRow and maxCol
								maxRow2 = _mm512_sub_epi32(maxRow2, vextend_gap);
								maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap);
								aux1 = _mm512_sub_epi32(current2, vopen_extend_gap);
								maxRow2 = _mm512_max_epi32(maxRow2, aux1);
								maxCol[j] =  _mm512_max_epi32(maxCol[j], aux1);	
								// update row buffer
								row2[j] = current2;
								// update max score
								score = _mm512_max_epi32(score,current2);
							}
						
						}
						if (k != nbb-1) {
							// update maxRow
							maxRow[i] = maxRow1;
							maxRow[i+1] = maxRow2;
							// update lastCol
							lastCol[i] = auxLastCol;
							lastCol[i+1] = current1;
							auxLastCol = current2;
						}
						// swap buffers
						tmp = row1;
						row1 = row2;
						row2 = tmp;
					}

				}

				// store max value
				_mm512_store_epi32(ptr_scores, score);
			}

			_mm_free(row1);_mm_free(row2); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol);
			if (qp_count > 0) { _mm_free(bIndexes); _mm_free(masks); }
			if (sp_count > 0) _mm_free(scoreProfile);

	}

	*workTime = dwalltime()-tick;	
	
	if (qp_count > 0) _mm_free(queryProfiles);
}


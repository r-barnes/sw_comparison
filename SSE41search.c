#include "SSE41search.h"


// CPU search using SSE instrucions and Score Profile technique
void search_sse41_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime){

	long int i, j, k;
	double tick;

	char *a, * b;
	unsigned int * a_disp;
	unsigned long int * b_disp = NULL;
	unsigned short int * m, *n, *nbbs, sequences_db_max_length, query_sequences_max_length; 

	a = query_sequences;
	m = query_sequences_lengths;
	a_disp = query_disp;

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = vect_sequences_db_lengths[vect_sequences_db_count-1];

	b =  vect_sequences_db;
	n = vect_sequences_db_lengths;
	nbbs =  vect_sequences_db_blocks;
	b_disp = vect_sequences_db_disp;

	tick = dwalltime();
	
	#pragma omp parallel default(none) shared(block_size, a, b, n, nbbs, m, a_disp, b_disp, submat, scores, query_sequences_count, vect_sequences_db_count, open_gap, extend_gap, sequences_db_max_length, query_sequences_max_length) num_threads(n_threads) 
	{

		__m128i  *row1, *row2, *maxCol, *maxRow, *lastCol, * ptr_scores, *tmp;
		__m128i*ptr_scoreProfile1, *ptr_scoreProfile2, *ptr_scoreProfile3, *ptr_scoreProfile4;
		char * ptr_a, * ptr_b, * scoreProfile;

		__declspec(align(MEMALIGN)) __m128i score, auxBlosum[2], auxLastCol, b_values;
		__declspec(align(MEMALIGN)) __m128i current1, current2, current3, current4, previous2, previous3, previous4;
		__declspec(align(MEMALIGN)) __m128i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
		__declspec(align(MEMALIGN)) __m128i vextend_gap_epi8 = _mm_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm_set1_epi8(open_gap+extend_gap), vzero_epi8 = _mm_set1_epi8(0);
		__declspec(align(MEMALIGN)) __m128i vextend_gap_epi16 = _mm_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm_set1_epi16(open_gap+extend_gap), vzero_epi16 = _mm_set1_epi16(0);
		__declspec(align(MEMALIGN)) __m128i vextend_gap_epi32 = _mm_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm_set1_epi32(open_gap+extend_gap), vzero_epi32 = _mm_set1_epi32(0);
		// SP
		__declspec(align(MEMALIGN)) __m128i v15 = _mm_set1_epi8(15), v16 = _mm_set1_epi8(16), vneg32 = _mm_set1_epi8(-32);
		// overflow
		__declspec(align(MEMALIGN)) __m128i v127 = _mm_set1_epi8(127), v32767 = _mm_set1_epi16(32767);
		// bias
		__declspec(align(MEMALIGN)) __m128i v128 = _mm_set1_epi32(128), v32768 = _mm_set1_epi32(32768); 


		unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, disp_5, dim1, dim2, nbb;
		unsigned long int t, s, q; 
		int overflow_flag, bb1, bb1_start, bb1_end, bb2, bb2_start, bb2_end;

		// allocate memory for auxiliary buffers
		row1 = (__m128i *) _mm_malloc((block_size+1)*sizeof(__m128i), MEMALIGN);
		row2 = (__m128i *) _mm_malloc((block_size+1)*sizeof(__m128i), MEMALIGN);
		maxCol = (__m128i *) _mm_malloc((block_size+1)*sizeof(__m128i), MEMALIGN);
		maxRow = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), MEMALIGN);
		lastCol = (__m128i *) _mm_malloc((query_sequences_max_length)*sizeof(__m128i), MEMALIGN);
		scoreProfile = (char *) _mm_malloc((SUBMAT_ROWS_x_SSE_INT8_VECTOR_LENGTH*block_size)*sizeof(char), MEMALIGN);
		
		// calculate alignment score
		#pragma omp for schedule(dynamic) nowait
		for (t=0; t< query_sequences_count*vect_sequences_db_count; t++) {

			q = (query_sequences_count-1) - (t % query_sequences_count);
			s = (vect_sequences_db_count-1) - (t / query_sequences_count);

			ptr_a = a + a_disp[q];
			ptr_b = b + b_disp[s];
			ptr_scores = (__m128i *) (scores + (q*vect_sequences_db_count+s)*SSE_INT8_VECTOR_LENGTH);

			// caluclate number of blocks
			nbb = nbbs[s];

			// init buffers
			#pragma unroll(SSE_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi8(-128);
			#pragma unroll(SSE_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi8(-128);
				
			// set score to 0
			score = _mm_set1_epi8(-128);

			for (k=0; k < nbb; k++){

				// calculate dim1
				disp_4 = k*block_size;
				dim1 = n[s]-disp_4;
				dim1 = (block_size < dim1 ? block_size : dim1);
				// calculate dim2
				dim2 = dim1 / DB_SEQ_LEN_MULT;
		
				// calculate a[i] displacement
				disp_1 = dim1*SSE_INT8_VECTOR_LENGTH;

				// init buffers
				#pragma unroll(SSE_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi8(-128);
				#pragma unroll(SSE_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm_set1_epi8(-128);
				auxLastCol = _mm_set1_epi8(-128);

				// build score profile
				for (i=0; i< dim1 ;i++ ) {
					// indexes
					b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*SSE_INT8_VECTOR_LENGTH));
					// indexes >= 16
					aux1 = _mm_sub_epi8(b_values, v16);
					// indexes < 16
					aux2 = _mm_cmpgt_epi8(b_values,v15);
					aux3 = _mm_and_si128(aux2,vneg32);
					aux4 = _mm_add_epi8(b_values,aux3);
					ptr_scoreProfile1 = (__m128i*)(scoreProfile) + i;
					#pragma unroll
					for (j=0; j< SUBMAT_ROWS-1; j++) {
						tmp = (__m128i *) (submat + j*SUBMAT_COLS);
						auxBlosum[0] = _mm_load_si128(tmp);
						auxBlosum[1] = _mm_load_si128(tmp+1);
						aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
						aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
						aux7 = _mm_add_epi8(aux5,  aux6);
						_mm_store_si128(ptr_scoreProfile1+j*dim1,   aux7);
					}
					_mm_store_si128(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero_epi8);
				}

				for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
				
					// update row[0] with lastCol[i-1]
					row1[0] = lastCol[i];
					previous2 = lastCol[i+1];
					previous3 = lastCol[i+2];
					previous4 = lastCol[i+3];
					// calculate score profile displacement
					ptr_scoreProfile1 = (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1);
					ptr_scoreProfile2 = (__m128i *)(scoreProfile+((int)(ptr_a[i+1]))*disp_1);
					ptr_scoreProfile3 = (__m128i *)(scoreProfile+((int)(ptr_a[i+2]))*disp_1);
					ptr_scoreProfile4 = (__m128i *)(scoreProfile+((int)(ptr_a[i+3]))*disp_1);
					// store maxRow in auxiliars
					aux1 = maxRow[i];
					aux2 = maxRow[i+1];
					aux3 = maxRow[i+2];
					aux4 = maxRow[i+3];

					for (ii=0; ii<dim2 ; ii++) {

						#pragma unroll(DB_SEQ_LEN_MULT)
						for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT; jj++, j++) {
							//calcuate the diagonal value
							current1 = _mm_adds_epi8(row1[j-1], _mm_load_si128(ptr_scoreProfile1+(j-1)));
							// calculate current1 max value
							current1 = _mm_max_epi8(current1, aux1);
							current1 = _mm_max_epi8(current1, maxCol[j]);
							//current1 = _mm_max_epi8(current1, vzero_epi8);
							// update maxRow and maxCol
							aux1 = _mm_subs_epi8(aux1, vextend_gap_epi8);
							maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 = _mm_subs_epi8(current1, vopen_extend_gap_epi8);
							aux1 = _mm_max_epi8(aux1, aux0);
							maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm_max_epi8(score,current1);
							
							//calcuate the diagonal value
							current2 = _mm_adds_epi8(previous2, _mm_load_si128(ptr_scoreProfile2+(j-1)));
							// update previous
							previous2 = current1;
							// calculate current2 max value
							current2 = _mm_max_epi8(current2, aux2);
							current2 = _mm_max_epi8(current2, maxCol[j]);
							//current2 = _mm_max_epi8(current2, vzero_epi8);
							// update maxRow and maxCol
							aux2 = _mm_subs_epi8(aux2, vextend_gap_epi8);
							maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 = _mm_subs_epi8(current2, vopen_extend_gap_epi8);
							aux2 = _mm_max_epi8(aux2, aux0);
							maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm_max_epi8(score,current2);

							//calcuate the diagonal value
							current3 = _mm_adds_epi8(previous3, _mm_load_si128(ptr_scoreProfile3+(j-1)));
							// update previous
							previous3 = current2;
							// calculate current3 max value
							current3 = _mm_max_epi8(current3, aux3);
							current3 = _mm_max_epi8(current3, maxCol[j]);
							//current3 = _mm_max_epi8(current3, vzero_epi8);
							// update maxRow and maxCol
							aux3 = _mm_subs_epi8(aux3, vextend_gap_epi8);
							maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 = _mm_subs_epi8(current3, vopen_extend_gap_epi8);
							aux3 = _mm_max_epi8(aux3, aux0);
							maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm_max_epi8(score,current3);

							//calcuate the diagonal value
							current4 = _mm_adds_epi8(previous4, _mm_load_si128(ptr_scoreProfile4+(j-1)));
							// update previous
							previous4 = current3;
							// calculate current4 max value
							current4 = _mm_max_epi8(current4, aux4);
							current4 = _mm_max_epi8(current4, maxCol[j]);
							//current4 = _mm_max_epi8(current4, vzero_epi8);
							// update maxRow and maxCol
							aux4 = _mm_subs_epi8(aux4, vextend_gap_epi8);
							maxCol[j] = _mm_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 = _mm_subs_epi8(current4, vopen_extend_gap_epi8);
							aux4 = _mm_max_epi8(aux4, aux0);
							maxCol[j] =  _mm_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm_max_epi8(score,current4);
							// update row buffer
							row2[j] = current4;
						}
					}
					// update maxRow
					maxRow[i] = aux1;
					maxRow[i+1] = aux2;
					maxRow[i+2] = aux3;
					maxRow[i+3] = aux4;
					// update lastCol
					lastCol[i] = auxLastCol;
					lastCol[i+1] = current1;
					lastCol[i+2] = current2;
					lastCol[i+3] = current3;
					auxLastCol = current4;
					// swap buffers
					tmp = row1;
					row1 = row2;
					row2 = tmp;
				
				}
			}

			// store max value
			aux1 = _mm_add_epi32(_mm_cvtepi8_epi32(score),v128);
			_mm_store_si128 (ptr_scores,aux1);
			aux1 = _mm_add_epi32(_mm_cvtepi8_epi32(_mm_srli_si128(score,4)),v128);
			_mm_store_si128 (ptr_scores+1,aux1);
			aux1 = _mm_add_epi32(_mm_cvtepi8_epi32(_mm_srli_si128(score,8)),v128);
			_mm_store_si128 (ptr_scores+2,aux1);
			aux1 = _mm_add_epi32(_mm_cvtepi8_epi32(_mm_srli_si128(score,12)),v128);
			_mm_store_si128 (ptr_scores+3,aux1);

			// overflow detection
			aux1 = _mm_cmpeq_epi8(score,v127);
			overflow_flag = _mm_test_all_zeros(aux1,v127); 

			// if overflow
			if (overflow_flag == 0){

				// detect if overflow occurred in low-half, high-half or both halves
				aux1 = _mm_cmpeq_epi8(_mm_slli_si128(score,8),v127);
				bb1_start = _mm_test_all_zeros(aux1,v127);
				aux1 = _mm_cmpeq_epi8(_mm_srli_si128(score,8),v127);
				bb1_end = 2 - _mm_test_all_zeros(aux1,v127);

				// recalculate using 16-bit signed integer precision
				for (bb1=bb1_start; bb1<bb1_end ; bb1++){

					// init buffers
					#pragma unroll(SSE_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi16(-32768);
					#pragma unroll(SSE_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi16(-32768);
						
					// set score to 0
					score = _mm_set1_epi16(-32768);

					disp_2 = bb1*SSE_INT16_VECTOR_LENGTH;

					for (k=0; k < nbb; k++){

						// calculate dim1
						disp_4 = k*block_size;
						dim1 = n[s]-disp_4;
						dim1 = (block_size < dim1 ? block_size : dim1);
						// calculate dim2
						dim2 = dim1 / DB_SEQ_LEN_MULT;

						// calculate a[i] displacement
						disp_1 = dim1*SSE_INT8_VECTOR_LENGTH;

						// init buffers
						#pragma unroll(SSE_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi16(-32768);
						#pragma unroll(SSE_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm_set1_epi16(-32768);
						auxLastCol = _mm_set1_epi16(-32768);

						// build score profile
						for (i=0; i< dim1 ;i++ ) {
							// indexes
							b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*SSE_INT8_VECTOR_LENGTH));
							// indexes >= 16
							aux1 = _mm_sub_epi8(b_values, v16);
							// indexes < 16
							aux2 = _mm_cmpgt_epi8(b_values,v15);
							aux3 = _mm_and_si128(aux2,vneg32);
							aux4 = _mm_add_epi8(b_values,aux3);
							ptr_scoreProfile1 = (__m128i*)(scoreProfile) + i;
							#pragma unroll
							for (j=0; j< SUBMAT_ROWS-1; j++) {
								tmp = (__m128i *) (submat + j*SUBMAT_COLS);
								auxBlosum[0] = _mm_load_si128(tmp);
								auxBlosum[1] = _mm_load_si128(tmp+1);
								aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
								aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
								aux7 = _mm_add_epi8(aux5,  aux6);
								_mm_store_si128(ptr_scoreProfile1+j*dim1,   aux7);
							}
							_mm_store_si128(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero_epi8);
						}

						for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
						
							// update row[0] with lastCol[i-1]
							row1[0] = lastCol[i];
							previous2 = lastCol[i+1];
							previous3 = lastCol[i+2];
							previous4 = lastCol[i+3];
							// calculate score profile displacement
							ptr_scoreProfile1 = (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2);
							ptr_scoreProfile2 = (__m128i *)(scoreProfile+((int)(ptr_a[i+1]))*disp_1+disp_2);
							ptr_scoreProfile3 = (__m128i *)(scoreProfile+((int)(ptr_a[i+2]))*disp_1+disp_2);
							ptr_scoreProfile4 = (__m128i *)(scoreProfile+((int)(ptr_a[i+3]))*disp_1+disp_2);
							// store maxRow in auxiliars
							aux1 = maxRow[i];
							aux2 = maxRow[i+1];
							aux3 = maxRow[i+2];
							aux4 = maxRow[i+3];

							for (ii=0; ii<dim2 ; ii++) {
									
								#pragma unroll(DB_SEQ_LEN_MULT)
								for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
									//calcuate the diagonal value
									current1 = _mm_adds_epi16(row1[j-1], _mm_cvtepi8_epi16(_mm_loadu_si128(ptr_scoreProfile1+(j-1))));
									// calculate current1 max value
									current1 = _mm_max_epi16(current1, aux1);
									current1 = _mm_max_epi16(current1, maxCol[j]);
									//current1 = _mm_max_epi16(current1, vzero_epi16);
									// update maxRow and maxCol
									aux1 = _mm_subs_epi16(aux1, vextend_gap_epi16);
									maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm_subs_epi16(current1, vopen_extend_gap_epi16);
									aux1 = _mm_max_epi16(aux1, aux0);
									maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm_max_epi16(score,current1);

									//calcuate the diagonal value
									current2 = _mm_adds_epi16(previous2, _mm_cvtepi8_epi16(_mm_loadu_si128(ptr_scoreProfile2+(j-1))));
									// update previous
									previous2 = current1;
									// calculate current2 max value
									current2 = _mm_max_epi16(current2, aux2);
									current2 = _mm_max_epi16(current2, maxCol[j]);
									//current2 = _mm_max_epi16(current2, vzero_epi16);
									// update maxRow and maxCol
									aux2 = _mm_subs_epi16(aux2, vextend_gap_epi16);
									maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm_subs_epi16(current2, vopen_extend_gap_epi16);
									aux2 = _mm_max_epi16(aux2, aux0);
									maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm_max_epi16(score,current2);

									//calcuate the diagonal value
									current3 = _mm_adds_epi16(previous3, _mm_cvtepi8_epi16(_mm_loadu_si128(ptr_scoreProfile3+(j-1))));
									// update previous
									previous3 = current2;
									// calculate current3 max value
									current3 = _mm_max_epi16(current3, aux3);
									current3 = _mm_max_epi16(current3, maxCol[j]);
									//current3 = _mm_max_epi16(current3, vzero_epi16);
									// update maxRow and maxCol
									aux3 = _mm_subs_epi16(aux3, vextend_gap_epi16);
									maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm_subs_epi16(current3, vopen_extend_gap_epi16);
									aux3 = _mm_max_epi16(aux3, aux0);
									maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm_max_epi16(score,current3);

									//calcuate the diagonal value
									current4 = _mm_adds_epi16(previous4, _mm_cvtepi8_epi16(_mm_loadu_si128(ptr_scoreProfile4+(j-1))));
									// update previous
									previous4 = current3;
									// calculate current4 max value
									current4 = _mm_max_epi16(current4, aux4);
									current4 = _mm_max_epi16(current4, maxCol[j]);
									//current4 = _mm_max_epi16(current4, vzero_epi16);
									// update maxRow and maxCol
									aux4 = _mm_subs_epi16(aux4, vextend_gap_epi16);
									maxCol[j] = _mm_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm_subs_epi16(current4, vopen_extend_gap_epi16);
									aux4 = _mm_max_epi16(aux4, aux0);
									maxCol[j] =  _mm_max_epi16(maxCol[j], aux0);	
									// update row buffer
									row2[j] = current4;
									// update max score
									score = _mm_max_epi16(score,current4);
								}
							}
							// update maxRow
							maxRow[i] = aux1;
							maxRow[i+1] = aux2;
							maxRow[i+2] = aux3;
							maxRow[i+3] = aux4;
							// update lastCol
							lastCol[i] = auxLastCol;
							lastCol[i+1] = current1;
							lastCol[i+2] = current2;
							lastCol[i+3] = current3;
							auxLastCol = current4;
							// swap buffers
							tmp = row1;
							row1 = row2;
							row2 = tmp;
						}

					}

					// store max value
					aux1 = _mm_add_epi32(_mm_cvtepi16_epi32(score),v32768);
					_mm_store_si128 (ptr_scores+bb1*2,aux1);
					aux1 = _mm_add_epi32(_mm_cvtepi16_epi32(_mm_srli_si128(score,8)),v32768);
					_mm_store_si128 (ptr_scores+bb1*2+1,aux1);

					// overflow detection
					aux1 = _mm_cmpeq_epi16(score,v32767);
					overflow_flag = _mm_test_all_zeros(aux1,v32767); 

					// if overflow
					if (overflow_flag == 0){

						// detect if overflow occurred in low-half, high-half or both halves
						aux1 = _mm_cmpeq_epi16(_mm_slli_si128(score,8),v32767);
						bb2_start = _mm_test_all_zeros(aux1,v32767);
						aux1 = _mm_cmpeq_epi16(_mm_srli_si128(score,8),v32767);
						bb2_end = 2 - _mm_test_all_zeros(aux1,v32767);

						// recalculate using 32-bit signed integer precision
						for (bb2=bb2_start; bb2<bb2_end ; bb2++){

							// init buffers
							#pragma unroll(SSE_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm_set1_epi32(0);
							#pragma unroll(SSE_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm_set1_epi32(0);
								
							// set score to 0
							score = _mm_set1_epi32(0);

							disp_3 = disp_2 + bb2*SSE_INT32_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								disp_4 = k*block_size;
								dim1 = n[s]-disp_4;
								dim1 = (block_size < dim1 ? block_size : dim1);
								// calculate dim2
								dim2 = dim1 / DB_SEQ_LEN_MULT;

								// calculate a[i] displacement
								disp_1 = dim1*SSE_INT8_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm_set1_epi32(0);
								#pragma unroll(SSE_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm_set1_epi32(0);
								auxLastCol = _mm_set1_epi32(0);

								// build score profile
								for (i=0; i< dim1 ;i++ ) {
									// indexes
									b_values = _mm_loadu_si128((__m128i *) (ptr_b + (disp_4+i)*SSE_INT8_VECTOR_LENGTH));
									// indexes >= 16
									aux1 = _mm_sub_epi8(b_values, v16);
									// indexes < 16
									aux2 = _mm_cmpgt_epi8(b_values,v15);
									aux3 = _mm_and_si128(aux2,vneg32);
									aux4 = _mm_add_epi8(b_values,aux3);
									ptr_scoreProfile1 = (__m128i*)(scoreProfile) + i;
									#pragma unroll
									for (j=0; j< SUBMAT_ROWS-1; j++) {
										tmp = (__m128i *) (submat + j*SUBMAT_COLS);
										auxBlosum[0] = _mm_load_si128(tmp);
										auxBlosum[1] = _mm_load_si128(tmp+1);
										aux5  = _mm_shuffle_epi8(auxBlosum[0], aux4);			
										aux6  = _mm_shuffle_epi8(auxBlosum[1], aux1);			
										aux7 = _mm_add_epi8(aux5,  aux6);
										_mm_store_si128(ptr_scoreProfile1+j*dim1,   aux7);
									}
									_mm_store_si128(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero_epi8);
								}

								for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
								
									// update row[0] with lastCol[i-1]
									row1[0] = lastCol[i];
									previous2 = lastCol[i+1];
									previous3 = lastCol[i+2];
									previous4 = lastCol[i+3];
									// calculate score profile displacement
									ptr_scoreProfile1 =  (__m128i *)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_3);
									ptr_scoreProfile2 =  (__m128i *)(scoreProfile+((int)(ptr_a[i+1]))*disp_1+disp_3);
									ptr_scoreProfile3 =  (__m128i *)(scoreProfile+((int)(ptr_a[i+2]))*disp_1+disp_3);
									ptr_scoreProfile4 =  (__m128i *)(scoreProfile+((int)(ptr_a[i+3]))*disp_1+disp_3);
									// store maxRow in auxiliars
									aux1 = maxRow[i];
									aux2 = maxRow[i+1];
									aux3 = maxRow[i+2];
									aux4 = maxRow[i+3];

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(DB_SEQ_LEN_MULT)
										for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
											//calcuate the diagonal value
											current1 = _mm_add_epi32(row1[j-1], _mm_cvtepi8_epi32(_mm_loadu_si128(ptr_scoreProfile1+(j-1))));
											// calculate current1 max value
											current1 = _mm_max_epi32(current1, aux1);
											current1 = _mm_max_epi32(current1, maxCol[j]);
											current1 = _mm_max_epi32(current1, vzero_epi32);
											// update maxRow and maxCol
											aux1 = _mm_sub_epi32(aux1, vextend_gap_epi32);
											maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm_sub_epi32(current1, vopen_extend_gap_epi32);
											aux1 = _mm_max_epi32(aux1, aux0);
											maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm_max_epi32(score,current1);

											//calcuate the diagonal value
											current2 = _mm_add_epi32(previous2, _mm_cvtepi8_epi32(_mm_loadu_si128(ptr_scoreProfile2+(j-1))));
											// update previous
											previous2 = current1;
											// calculate current2 max value
											current2 = _mm_max_epi32(current2, aux2);
											current2 = _mm_max_epi32(current2, maxCol[j]);
											current2 = _mm_max_epi32(current2, vzero_epi32);
											// update maxRow and maxCol
											aux2 = _mm_sub_epi32(aux2, vextend_gap_epi32);
											maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm_sub_epi32(current2, vopen_extend_gap_epi32);
											aux2 = _mm_max_epi32(aux2, aux0);
											maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm_max_epi32(score,current2);

											//calcuate the diagonal value
											current3 = _mm_add_epi32(previous3, _mm_cvtepi8_epi32(_mm_loadu_si128(ptr_scoreProfile3+(j-1))));
											// update previous
											previous3 = current2;
											// calculate current3 max value
											current3 = _mm_max_epi32(current3, aux3);
											current3 = _mm_max_epi32(current3, maxCol[j]);
											current3 = _mm_max_epi32(current3, vzero_epi32);
											// update maxRow and maxCol
											aux3 = _mm_sub_epi32(aux3, vextend_gap_epi32);
											maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm_sub_epi32(current3, vopen_extend_gap_epi32);
											aux3 = _mm_max_epi32(aux3, aux0);
											maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm_max_epi32(score,current3);
											
											//calcuate the diagonal value
											current4 = _mm_add_epi32(previous4, _mm_cvtepi8_epi32(_mm_loadu_si128(ptr_scoreProfile4+(j-1))));
											// update previous
											previous4 = current3;
											// calculate current4 max value
											current4 = _mm_max_epi32(current4, aux4);
											current4 = _mm_max_epi32(current4, maxCol[j]);
											current4 = _mm_max_epi32(current4, vzero_epi32);
											// update maxRow and maxCol
											aux4 = _mm_sub_epi32(aux4, vextend_gap_epi32);
											maxCol[j] = _mm_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm_sub_epi32(current4, vopen_extend_gap_epi32);
											aux4 = _mm_max_epi32(aux4, aux0);
											maxCol[j] =  _mm_max_epi32(maxCol[j], aux0);	
											// update row buffer
											row2[j] = current4;
											// update max score
											score = _mm_max_epi32(score,current4);											}
									}
									// update maxRow
									maxRow[i] = aux1;
									maxRow[i+1] = aux2;
									maxRow[i+2] = aux3;
									maxRow[i+3] = aux4;
									// update lastCol
									lastCol[i] = auxLastCol;
									lastCol[i+1] = current1;
									lastCol[i+2] = current2;
									lastCol[i+3] = current3;
									auxLastCol = current4;
									// swap buffers
									tmp = row1;
									row1 = row2;
									row2 = tmp;
								}
							}
							// store max value
							_mm_store_si128 (ptr_scores+bb1*2+bb2,score);
						}
					}
				}
			}
		}

		 _mm_free(row1); _mm_free(row2); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol); _mm_free(scoreProfile);
	}

	*workTime = dwalltime()-tick;	
}


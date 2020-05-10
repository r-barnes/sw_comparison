#include "charSubmat.h"
#include "AVX-512BWsearch.h"
#include "utils.h"

// search using AVX512BW instructions and Score Profile technique
void search_avx512bw_sp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime){

	long int i, j, k;
	double tick;

	char *a, * b;
	unsigned int  * a_disp;
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

		__m512i *row1, *row2, *tmp_row, *maxCol, *maxRow, *lastCol, * ptr_scores;
		__m512i *ptr_scoreProfile1, *ptr_scoreProfile2, *ptr_scoreProfile3, *ptr_scoreProfile4;
		char * ptr_a, * ptr_b, * scoreProfile;


		__declspec(align(MEMALIGN)) __m512i score, auxLastCol, b_values, submat_lo, submat_hi;
		__declspec(align(MEMALIGN)) __m512i current1, current2, current3, current4, previous2, previous3, previous4;
		__declspec(align(MEMALIGN)) __m512i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7, aux8;
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi8 = _mm512_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm512_set1_epi8(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi16 = _mm512_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm512_set1_epi16(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi32 = _mm512_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm512_set1_epi32(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vzero = _mm512_setzero_si512();
		// SP
		__declspec(align(MEMALIGN)) __m512i v15 = _mm512_set1_epi8(15), vneg32 = _mm512_set1_epi8(-32), v16 = _mm512_set1_epi8(16);
		// overflow detection
		__declspec(align(MEMALIGN)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767), aux256[2];
		// bias
		__declspec(align(MEMALIGN)) __m512i v128 = _mm512_set1_epi32(128), v32768 = _mm512_set1_epi32(32768);
		__declspec(align(MEMALIGN)) __m128i aux128, auxBlosum[2], *tmp;
		__mmask64 mask;

		unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim1, dim2, nbb;
		unsigned long int t, s, q; 
		int overflow_low_flag, overflow_high_flag, bb1, bb2, bb1_start, bb1_end, bb2_start, bb2_end;

		// allocate memory for auxiliary buffers
		row1 = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		row2 = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		maxCol = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		maxRow = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), MEMALIGN);
		lastCol = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), MEMALIGN);
		scoreProfile = (char *) _mm_malloc((SUBMAT_ROWS_x_AVX512BW_INT8_VECTOR_LENGTH*block_size)*sizeof(char), MEMALIGN);

		// calculate alignment score
		#pragma omp for schedule(dynamic) nowait
		for (t=0; t< query_sequences_count*vect_sequences_db_count; t++) {

			q = (query_sequences_count-1) - (t % query_sequences_count);
			s = (vect_sequences_db_count-1) - (t / query_sequences_count);

			ptr_a = a + a_disp[q];
			ptr_b = b + b_disp[s];
			ptr_scores = (__m512i *) (scores + (q*vect_sequences_db_count+s)*AVX512BW_INT8_VECTOR_LENGTH);

			// calculate number of blocks
			nbb = nbbs[s];

			// init buffers
			#pragma unroll(AVX512BW_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) maxRow[i] =  _mm512_set1_epi8(-128);
			#pragma unroll(AVX512BW_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) lastCol[i] =  _mm512_set1_epi8(-128);
				
			// set score to 0
			score =  _mm512_set1_epi8(-128);

			for (k=0; k < nbb; k++){

				// calculate dim1
				disp_4 = k*block_size;
				dim1 = n[s]-disp_4;
				dim1 = (block_size < dim1 ? block_size : dim1);
				// calculate dim2
				dim2 = dim1 / DB_SEQ_LEN_MULT;

				// calculate SP sub-block length
				disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

				// init buffers
				#pragma unroll(AVX512BW_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_set1_epi8(-128);
				#pragma unroll(AVX512BW_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_set1_epi8(-128);
				auxLastCol = _mm512_set1_epi8(-128);

				// build score profile
				for (i=0; i< dim1 ;i++ ) {
					// indexes
					b_values =  _mm512_loadu_si512((__m512i *) (ptr_b + (disp_4+i)*AVX512BW_INT8_VECTOR_LENGTH));
					// indexes >= 16
					aux1 = _mm512_sub_epi8(b_values, v16);
					// indexes < 16
					mask = _mm512_cmplt_epi8_mask(b_values,v16);
					aux4 = _mm512_mask_mov_epi8(vneg32, mask, b_values);
					ptr_scoreProfile1 = (__m512i *)(scoreProfile) + i;
					#pragma unroll
					for (j=0; j< SUBMAT_ROWS-1; j++) {
						tmp = (__m128i*) (submat + j*SUBMAT_COLS);
						auxBlosum[0] = _mm_load_si128(tmp);
						auxBlosum[1] = _mm_load_si128(tmp+1);
						submat_lo = _mm512_broadcast_i64x2(auxBlosum[0]);
						submat_hi = _mm512_broadcast_i64x2(auxBlosum[1]);
						aux5 = _mm512_shuffle_epi8(submat_lo,aux4);
						aux6 = _mm512_shuffle_epi8(submat_hi,aux1);
						_mm512_store_si512(ptr_scoreProfile1+j*dim1,_mm512_or_si512(aux5,aux6));
					}
					_mm512_store_si512(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero);
				}

				for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
				
					// update row[0] with lastCol[i-1]
					row1[0] = lastCol[i];
					previous2 = lastCol[i+1];
					previous3 = lastCol[i+2];
					previous4 = lastCol[i+3];
					// calculate score profile displacement
					ptr_scoreProfile1 = (__m512i*)(scoreProfile+((unsigned int)(ptr_a[i]))*disp_1);
					ptr_scoreProfile2 = (__m512i*)(scoreProfile+((unsigned int)(ptr_a[i+1]))*disp_1);
					ptr_scoreProfile3 = (__m512i*)(scoreProfile+((unsigned int)(ptr_a[i+2]))*disp_1);
					ptr_scoreProfile4 = (__m512i*)(scoreProfile+((unsigned int)(ptr_a[i+3]))*disp_1);
					// store maxRow in auxiliars
					aux1 = maxRow[i];
					aux2 = maxRow[i+1];
					aux3 = maxRow[i+2];
					aux4 = maxRow[i+3];

					for (ii=0; ii<dim2 ; ii++) {

						#pragma unroll(DB_SEQ_LEN_MULT)
						for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
							//calcuate the diagonal value
							current1 =  _mm512_adds_epi8(row1[j-1], _mm512_load_si512(ptr_scoreProfile1+(j-1)));
							// calculate current1 max value
							current1 = _mm512_max_epi8(current1, aux1);
							current1 = _mm512_max_epi8(current1, maxCol[j]);
							//current1 = _mm512_max_epi8(current1, vzero);
							// update maxRow and maxCol
							aux1 =  _mm512_subs_epi8(aux1, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current1, vopen_extend_gap_epi8);
							aux1 = _mm512_max_epi8(aux1, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current1);

							//calcuate the diagonal value
							current2 =  _mm512_adds_epi8(previous2, _mm512_load_si512(ptr_scoreProfile2+(j-1)));
							// update previous
							previous2 = current1;
							// calculate current2 max value
							current2 = _mm512_max_epi8(current2, aux2);
							current2 = _mm512_max_epi8(current2, maxCol[j]);
							//current2 = _mm512_max_epi8(current2, vzero);
							// update maxRow and maxCol
							aux2 =  _mm512_subs_epi8(aux2, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current2, vopen_extend_gap_epi8);
							aux2 = _mm512_max_epi8(aux2, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current2);							

							//calcuate the diagonal value
							current3 =  _mm512_adds_epi8(previous3, _mm512_load_si512(ptr_scoreProfile3+(j-1)));
							// update previous
							previous3 = current2;
							// calculate current3 max value
							current3 = _mm512_max_epi8(current3, aux3);
							current3 = _mm512_max_epi8(current3, maxCol[j]);
							//current3 = _mm512_max_epi8(current3, vzero);
							// update maxRow and maxCol
							aux3 =  _mm512_subs_epi8(aux3, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current3, vopen_extend_gap_epi8);
							aux3 = _mm512_max_epi8(aux3, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current3);							

							//calcuate the diagonal value
							current4 =  _mm512_adds_epi8(previous4, _mm512_load_si512(ptr_scoreProfile4+(j-1)));
							// update previous
							previous4 = current3;
							// calculate current4 max value
							current4 = _mm512_max_epi8(current4, aux4);
							current4 = _mm512_max_epi8(current4, maxCol[j]);
							//current4 = _mm512_max_epi8(current4, vzero);
							// update maxRow and maxCol
							aux4 =  _mm512_subs_epi8(aux4, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current4, vopen_extend_gap_epi8);
							aux4 = _mm512_max_epi8(aux4, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update row buffer
							row2[j] = current4;
							// update max score
							score = _mm512_max_epi8(score,current4);							
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
					tmp_row = row1;
					row1 = row2;
					row2 = tmp_row;
				}
			}

			// store max value
			#pragma unroll
			for (i=0; i < 4; i++){
				aux128 = _mm512_extracti32x4_epi32  (score, i);
				aux1 = _mm512_add_epi32(_mm512_cvtepi8_epi32(aux128),v128);
				_mm512_store_si512 (ptr_scores+i, aux1);
			}
			
			// overflow detection
			// low 
			aux256[0] = _mm512_extracti32x8_epi32 (score,0);
			aux256[1] = _mm256_cmpeq_epi8(aux256[0],v127);
			overflow_low_flag = _mm256_testz_si256(aux256[1],v127); 
			// high
			aux256[0] = _mm512_extracti32x8_epi32 (score,1);
			aux256[1] = _mm256_cmpeq_epi8(aux256[0],v127);
			overflow_high_flag = _mm256_testz_si256(aux256[1],v127); 

			// if overflow
			if ((overflow_low_flag == 0) || (overflow_high_flag == 0)){

				bb1_start = overflow_low_flag;
				bb1_end = 2-overflow_high_flag;

				// recalculate using 16-bit signed integer precision
				for (bb1=bb1_start; bb1<bb1_end ; bb1++){

					// init buffers
					#pragma unroll(AVX512BW_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_set1_epi16(-32768);
					#pragma unroll(AVX512BW_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_set1_epi16(-32768);
					
					// set score to 0
					score = _mm512_set1_epi16(-32768);

					disp_2 = bb1*AVX512BW_INT16_VECTOR_LENGTH;

					for (k=0; k < nbb; k++){

						// calculate dim1
						disp_4 = k*block_size;
						dim1 = n[s]-disp_4;
						dim1 = (block_size < dim1 ? block_size : dim1);
						// calculate dim2
						dim2 = dim1 / DB_SEQ_LEN_MULT;

						// calculate SP sub-block length
						disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

						// init buffers
						#pragma unroll(AVX512BW_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_set1_epi16(-32768);
						#pragma unroll(AVX512BW_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_set1_epi16(-32768);
						auxLastCol = _mm512_set1_epi16(-32768);

						// build score profile
						for (i=0; i< dim1 ;i++ ) {
							// indexes
							b_values =  _mm512_loadu_si512((__m512i *) (ptr_b + (disp_4+i)*AVX512BW_INT8_VECTOR_LENGTH));
							// indexes >= 16
							aux1 = _mm512_sub_epi8(b_values, v16);
							// indexes < 16
							mask = _mm512_cmplt_epi8_mask(b_values,v16);
							aux4 = _mm512_mask_mov_epi8(vneg32, mask, b_values);
							ptr_scoreProfile1 = (__m512i *)(scoreProfile) + i;
							#pragma unroll
							for (j=0; j< SUBMAT_ROWS-1; j++) {
								tmp = (__m128i*) (submat + j*SUBMAT_COLS);
								auxBlosum[0] = _mm_load_si128(tmp);
								auxBlosum[1] = _mm_load_si128(tmp+1);
								submat_lo = _mm512_broadcast_i64x2(auxBlosum[0]);
								submat_hi = _mm512_broadcast_i64x2(auxBlosum[1]);
								aux5 = _mm512_shuffle_epi8(submat_lo,aux4);
								aux6 = _mm512_shuffle_epi8(submat_hi,aux1);
								_mm512_store_si512(ptr_scoreProfile1+j*dim1,_mm512_or_si512(aux5,aux6));
							}
							_mm512_store_si512(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero);
						}

						for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
						
							// update row[0] with lastCol[i-1]
							row1[0] = lastCol[i];
							previous2 = lastCol[i+1];
							previous3 = lastCol[i+2];
							previous4 = lastCol[i+3];
							// calculate score profile displacement
							ptr_scoreProfile1 = (__m512i *)(scoreProfile+((int)(ptr_a[i]))*disp_1+disp_2);
							ptr_scoreProfile2 = (__m512i *)(scoreProfile+((int)(ptr_a[i+1]))*disp_1+disp_2);
							ptr_scoreProfile3 = (__m512i *)(scoreProfile+((int)(ptr_a[i+2]))*disp_1+disp_2);
							ptr_scoreProfile4 = (__m512i *)(scoreProfile+((int)(ptr_a[i+3]))*disp_1+disp_2);
							// store maxRow in auxiliars
							aux1 = maxRow[i];
							aux2 = maxRow[i+1];
							aux3 = maxRow[i+2];
							aux4 = maxRow[i+3];

							for (ii=0; ii<dim2 ; ii++) {

								#pragma unroll(DB_SEQ_LEN_MULT)
								for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
									//calcuate the diagonal value
									current1 = _mm512_adds_epi16(row1[j-1], _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (ptr_scoreProfile1+(j-1)))));
									// calculate current1 max value
									current1 = _mm512_max_epi16(current1, aux1);
									current1 = _mm512_max_epi16(current1, maxCol[j]);
									//current1 = _mm512_max_epi16(current1, vzero);
									// update maxRow and maxCol
									aux1 = _mm512_subs_epi16(aux1, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current1, vopen_extend_gap_epi16);
									aux1 = _mm512_max_epi16(aux1, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current1);

									//calcuate the diagonal value
									current2 = _mm512_adds_epi16(previous2, _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (ptr_scoreProfile2+(j-1)))));
									// update previous
									previous2 = current1;
									// calculate current2 max value
									current2 = _mm512_max_epi16(current2, aux2);
									current2 = _mm512_max_epi16(current2, maxCol[j]);
									//current2 = _mm512_max_epi16(current2, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_subs_epi16(aux2, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current2, vopen_extend_gap_epi16);
									aux2 = _mm512_max_epi16(aux2, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current2);

									//calcuate the diagonal value
									current3 = _mm512_adds_epi16(previous3, _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (ptr_scoreProfile3+(j-1)))));
									// update previous
									previous3 = current2;
									// calculate current3 max value
									current3 = _mm512_max_epi16(current3, aux3);
									current3 = _mm512_max_epi16(current3, maxCol[j]);
									//current3 = _mm512_max_epi16(current3, vzero);
									// update maxRow and maxCol
									aux3 = _mm512_subs_epi16(aux3, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current3, vopen_extend_gap_epi16);
									aux3 = _mm512_max_epi16(aux3, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current3);

									//calcuate the diagonal value
									current4 = _mm512_adds_epi16(previous4, _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (ptr_scoreProfile4+(j-1)))));
									// update previous
									previous4 = current3;
									// calculate current4 max value
									current4 = _mm512_max_epi16(current4, aux4);
									current4 = _mm512_max_epi16(current4, maxCol[j]);
									//current4 = _mm512_max_epi16(current4, vzero);
									// update maxRow and maxCol
									aux4 = _mm512_subs_epi16(aux4, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current4, vopen_extend_gap_epi16);
									aux4 = _mm512_max_epi16(aux4, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update row buffer
									row2[j] = current4;
									// update max score
									score = _mm512_max_epi16(score,current4);
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
							tmp_row = row1;
							row1 = row2;
							row2 = tmp_row;
						}
					}
					// store max value (low)
					aux256[0] = _mm512_extracti32x8_epi32 (score,0);
					aux1 = _mm512_add_epi32(_mm512_cvtepi16_epi32(aux256[0]),v32768);
					_mm512_store_si512 (ptr_scores+bb1*2,aux1);
					//  check overflow (low)
					aux256[1] = _mm256_cmpeq_epi16(aux256[0],v32767);
					overflow_low_flag = _mm256_testz_si256(aux256[1],v127); 
					// store max value (high)
					aux256[0] = _mm512_extracti32x8_epi32 (score,1);
					aux1 = _mm512_add_epi32(_mm512_cvtepi16_epi32(aux256[0]),v32768);
					_mm512_store_si512 (ptr_scores+bb1*2+1,aux1);
					//  check overflow (high)
					aux256[1] = _mm256_cmpeq_epi16(aux256[0],v32767);
					overflow_high_flag = _mm256_testz_si256(aux256[1],v32767); 

					// if overflow
					if ((overflow_low_flag == 0) || (overflow_high_flag == 0)){

						// check overflow in low 16 bits
						bb2_start = overflow_low_flag;
						// check overflow in high 16 bits
						bb2_end = 2 - overflow_high_flag;

						// recalculate using 32-bit signed integer precision
						for (bb2=bb2_start; bb2<bb2_end ; bb2++){

							// init buffers
							#pragma unroll(AVX512BW_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_si512();
							#pragma unroll(AVX512BW_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_si512();
							
							// set score to 0
							score = _mm512_setzero_si512();

							disp_3 = disp_2 + bb2*AVX512BW_INT32_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								disp_4 = k*block_size;
								dim1 = n[s]-disp_4;
								dim1 = (block_size < dim1 ? block_size : dim1);
								// calculate dim2
								dim2 = dim1 / DB_SEQ_LEN_MULT;

								// calculate SP sub-block length
								disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(AVX512BW_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_setzero_si512();
								#pragma unroll(AVX512BW_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_setzero_si512();
								auxLastCol = _mm512_setzero_si512();

								// build score profile
								for (i=0; i< dim1 ;i++ ) {
									// indexes
									b_values =  _mm512_loadu_si512((__m512i *) (ptr_b + (disp_4+i)*AVX512BW_INT8_VECTOR_LENGTH));
									// indexes >= 16
									aux1 = _mm512_sub_epi8(b_values, v16);
									// indexes < 16
									mask = _mm512_cmplt_epi8_mask(b_values,v16);
									aux4 = _mm512_mask_mov_epi8(vneg32, mask, b_values);
									ptr_scoreProfile1 = (__m512i *)(scoreProfile) + i;
									#pragma unroll
									for (j=0; j< SUBMAT_ROWS-1; j++) {
										tmp = (__m128i*) (submat + j*SUBMAT_COLS);
										auxBlosum[0] = _mm_load_si128(tmp);
										auxBlosum[1] = _mm_load_si128(tmp+1);
										submat_lo = _mm512_broadcast_i64x2(auxBlosum[0]);
										submat_hi = _mm512_broadcast_i64x2(auxBlosum[1]);
										aux5 = _mm512_shuffle_epi8(submat_lo,aux4);
										aux6 = _mm512_shuffle_epi8(submat_hi,aux1);
										_mm512_store_si512(ptr_scoreProfile1+j*dim1,_mm512_or_si512(aux5,aux6));
									}
									_mm512_store_si512(ptr_scoreProfile1+(SUBMAT_ROWS-1)*dim1,  vzero);
								}


								for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
								
									// update row[0] with lastCol[i-1]
									row1[0] = lastCol[i];
									previous2 = lastCol[i+1];
									previous3 = lastCol[i+2];
									previous4 = lastCol[i+3];
									// calculate score profile displacement
									ptr_scoreProfile1 = (__m512i *)(scoreProfile+((unsigned int)(ptr_a[i]))*disp_1+disp_3);
									ptr_scoreProfile2 = (__m512i *)(scoreProfile+((unsigned int)(ptr_a[i+1]))*disp_1+disp_3);
									ptr_scoreProfile3 = (__m512i *)(scoreProfile+((unsigned int)(ptr_a[i+2]))*disp_1+disp_3);
									ptr_scoreProfile4 = (__m512i *)(scoreProfile+((unsigned int)(ptr_a[i+3]))*disp_1+disp_3);
									// store maxRow in auxiliars
									aux1 = maxRow[i];
									aux2 = maxRow[i+1];
									aux3 = maxRow[i+2];
									aux4 = maxRow[i+3];

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(DB_SEQ_LEN_MULT)
										for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
											//calcuate the diagonal value
											current1 = _mm512_add_epi32(row1[j-1], _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile1+(j-1)))));
											// calculate current1 max value
											current1 = _mm512_max_epi32(current1, aux1);
											current1 = _mm512_max_epi32(current1, maxCol[j]);
											current1 = _mm512_max_epi32(current1, vzero);
											// update maxRow and maxCol
											aux1 = _mm512_sub_epi32(aux1, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current1, vopen_extend_gap_epi32);
											aux1 = _mm512_max_epi32(aux1, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current1);

											//calcuate the diagonal value
											current2 = _mm512_add_epi32(previous2, _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile2+(j-1)))));
											// update previous
											previous2 = current1;
											// calculate current2 max value
											current2 = _mm512_max_epi32(current2, aux2);
											current2 = _mm512_max_epi32(current2, maxCol[j]);
											current2 = _mm512_max_epi32(current2, vzero);
											// update maxRow and maxCol
											aux2 = _mm512_sub_epi32(aux2, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current2, vopen_extend_gap_epi32);
											aux2 = _mm512_max_epi32(aux2, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current2);

											//calcuate the diagonal value
											current3 = _mm512_add_epi32(previous3, _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile3+(j-1)))));
											// update previous
											previous3 = current2;
											// calculate current3 max value
											current3 = _mm512_max_epi32(current3, aux3);
											current3 = _mm512_max_epi32(current3, maxCol[j]);
											current3 = _mm512_max_epi32(current3, vzero);
											// update maxRow and maxCol
											aux3 = _mm512_sub_epi32(aux3, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current3, vopen_extend_gap_epi32);
											aux3 = _mm512_max_epi32(aux3, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current3);

											//calcuate the diagonal value
											current4 = _mm512_add_epi32(previous4, _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (ptr_scoreProfile4+(j-1)))));
											// update previous
											previous4 = current3;
											// calculate current4 max value
											current4 = _mm512_max_epi32(current4, aux4);
											current4 = _mm512_max_epi32(current4, maxCol[j]);
											current4 = _mm512_max_epi32(current4, vzero);
											// update maxRow and maxCol
											aux4 = _mm512_sub_epi32(aux4, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current4, vopen_extend_gap_epi32);
											aux4 = _mm512_max_epi32(aux4, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update row buffer
											row2[j] = current4;
											// update max score
											score = _mm512_max_epi32(score,current4);

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
									tmp_row = row1;
									row1 = row2;
									row2 = tmp_row;
								}
							}
							// store max value
							_mm512_store_si512 (ptr_scores+bb1*2+bb2,score);
						}
					}
				}
			}
		}

		 _mm_free(row1);  _mm_free(row2);  _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol); _mm_free(scoreProfile);
	}

	*workTime = dwalltime()-tick;

}

// search using AVX512BW instructions and Query Profile technique
void search_avx512bw_qp (char * query_sequences, unsigned short int * query_sequences_lengths, unsigned long int query_sequences_count, unsigned int * query_disp,
	char * vect_sequences_db, unsigned short int * vect_sequences_db_lengths, unsigned short int * vect_sequences_db_blocks, unsigned long int vect_sequences_db_count, 
	unsigned long int * vect_sequences_db_disp,	char * submat, int open_gap, int extend_gap, int n_threads, int block_size, int * scores, double * workTime){

	long int i, j, k;
	double tick;

	char *a, * b;
	unsigned int  * a_disp;
	unsigned long int * b_disp = NULL;
	unsigned short int * m, *n, *nbbs, sequences_db_max_length, query_sequences_max_length; 
	__m512i * queryProfiles;

	a = query_sequences;
	m = query_sequences_lengths;
	a_disp = query_disp;

	query_sequences_max_length = query_sequences_lengths[query_sequences_count-1];
	sequences_db_max_length = vect_sequences_db_lengths[vect_sequences_db_count-1];

	b =  vect_sequences_db;
	n = vect_sequences_db_lengths;
	nbbs =  vect_sequences_db_blocks;
	b_disp = vect_sequences_db_disp;

	// allocate memory for query profiles (if correspond)
	queryProfiles = (__m512i *)_mm_malloc((a_disp[query_sequences_count])*2*sizeof(__m512i), MEMALIGN);

	tick = dwalltime();

	#pragma omp parallel default(none) shared(block_size, a, b, n, nbbs, m, a_disp, b_disp, submat, scores, query_sequences_count, vect_sequences_db_count, \
		open_gap, extend_gap, sequences_db_max_length, query_sequences_max_length, queryProfiles) num_threads(n_threads) 
	{

		__m512i *row1, *row2, *tmp_row, *maxCol, *maxRow, *lastCol, * ptr_scores, * ptr_b_block, *queryProfile;
		__declspec(align(MEMALIGN)) char * ptr_a, * ptr_b, submatValues[AVX512BW_INT8_VECTOR_LENGTH];


		__declspec(align(MEMALIGN)) __m512i score, auxLastCol, *bIndexes_lo,  *bIndexes_hi;
		__declspec(align(MEMALIGN)) __m512i aux0, aux1, aux2, aux3, aux4, aux5, aux6, aux7;
		__declspec(align(MEMALIGN)) __m512i current1, current2, current3, current4, previous2, previous3, previous4;
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi8 = _mm512_set1_epi8(extend_gap), vopen_extend_gap_epi8 = _mm512_set1_epi8(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi16 = _mm512_set1_epi16(extend_gap), vopen_extend_gap_epi16 = _mm512_set1_epi16(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vextend_gap_epi32 = _mm512_set1_epi32(extend_gap), vopen_extend_gap_epi32 = _mm512_set1_epi32(open_gap+extend_gap);
		__declspec(align(MEMALIGN)) __m512i vzero = _mm512_setzero_si512();
		// QP
		__declspec(align(MEMALIGN)) __m512i v15 = _mm512_set1_epi8(15), vneg32 = _mm512_set1_epi8(-32), v16 = _mm512_set1_epi8(16);
		// overflow detection
		__declspec(align(MEMALIGN)) __m256i v127 = _mm256_set1_epi8(127), v32767 = _mm256_set1_epi16(32767), aux256[2];
		// bias
		__declspec(align(MEMALIGN)) __m512i v128 = _mm512_set1_epi32(128), v32768 = _mm512_set1_epi32(32768);
		__declspec(align(MEMALIGN)) __m128i aux128, auxBlosum[2], *tmp;
		__mmask64 mask;


		#pragma omp for schedule(dynamic)  
		for (i=0; i< a_disp[query_sequences_count] ; i++) {
			queryProfiles[i*2] = _mm512_broadcast_i64x2(_mm_loadu_si128((__m128i*)(submat)+a[i]*2));
			queryProfiles[i*2+1] = _mm512_broadcast_i64x2(_mm_loadu_si128((__m128i*)(submat)+a[i]*2+1));
		}


		unsigned int i, j, ii, jj, k, disp_1, disp_2, disp_3, disp_4, dim1, dim2, nbb;
		unsigned long int t, s, q; 
		int overflow_low_flag, overflow_high_flag, bb1, bb2, bb1_start, bb1_end, bb2_start, bb2_end;

		// allocate memory for auxiliary buffers
		row1 = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		row2 = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		maxCol = (__m512i *) _mm_malloc((block_size+1)*sizeof(__m512i), MEMALIGN);
		maxRow = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), MEMALIGN);
		lastCol = (__m512i *) _mm_malloc((query_sequences_max_length)*sizeof(__m512i), MEMALIGN);
		// alloc memory for indexes
		bIndexes_lo = (__m512i *) _mm_malloc((block_size)*sizeof(__m512i), MEMALIGN);
		bIndexes_hi = (__m512i *) _mm_malloc((block_size)*sizeof(__m512i), MEMALIGN);


		// calculate alignment score
		#pragma omp for schedule(dynamic) nowait
		for (t=0; t< query_sequences_count*vect_sequences_db_count; t++) {

			q = (query_sequences_count-1) - (t % query_sequences_count);
			s = (vect_sequences_db_count-1) - (t / query_sequences_count);

			ptr_a = a + a_disp[q];
			ptr_b = b + b_disp[s];
			ptr_scores = (__m512i *) (scores + (q*vect_sequences_db_count+s)*AVX512BW_INT8_VECTOR_LENGTH);
			queryProfile = queryProfiles + a_disp[q]*2;


			// calculate number of blocks
			nbb = nbbs[s];

			// init buffers
			#pragma unroll(AVX512BW_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_set1_epi8(-128);
			#pragma unroll(AVX512BW_UNROLL_COUNT)
			for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_set1_epi8(-128);
				
			// set score to 0
			score = _mm512_set1_epi8(-128);

			for (k=0; k < nbb; k++){

				// calculate dim1
				disp_4 = k*block_size;
				dim1 = n[s]-disp_4;
				dim1 = (block_size < dim1 ? block_size : dim1);
				// calculate dim2
				dim2 = dim1 / DB_SEQ_LEN_MULT;

				// calculate SP sub-block length
				disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

				// init buffers
				#pragma unroll(AVX512BW_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_set1_epi8(-128);
				#pragma unroll(AVX512BW_UNROLL_COUNT)
				for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_set1_epi8(-128);
				auxLastCol = _mm512_set1_epi8(-128);

				// get bIndexes
				ptr_b_block = (__m512i*)(ptr_b) + disp_4;
				#pragma unroll(AVX512BW_UNROLL_COUNT)
				for (i=0; i<dim1 ; i++ ) {
					// indexes >= 16
					bIndexes_hi[i] = _mm512_sub_epi8(ptr_b_block[i], v16);
					// indexes < 16
					mask = _mm512_cmplt_epi8_mask(ptr_b_block[i],v16);
					bIndexes_lo[i] = _mm512_mask_mov_epi8(vneg32, mask, ptr_b_block[i]);
				}

				for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
				
					// update row[0] with lastCol[i-1]
					row1[0] = lastCol[i];
					previous2 = lastCol[i+1];
					previous3 = lastCol[i+2];
					previous4 = lastCol[i+3];
					// store maxRow in auxiliars
					aux1 = maxRow[i];
					aux2 = maxRow[i+1];
					aux3 = maxRow[i+2];
					aux4 = maxRow[i+3];

					for (ii=0; ii<dim2 ; ii++) {

						#pragma unroll(DB_SEQ_LEN_MULT)
						for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
							//calcuate the scoring matrix value
							aux5 = _mm512_shuffle_epi8(queryProfile[i*2],bIndexes_lo[j-1]);
							aux6 = _mm512_shuffle_epi8(queryProfile[i*2+1],bIndexes_hi[j-1]);
							//calcuate the diagonal value
							current1 =  _mm512_adds_epi8(row1[j-1], _mm512_or_si512(aux5,aux6));
							// calculate current1 max value
							current1 = _mm512_max_epi8(current1, aux1);
							current1 = _mm512_max_epi8(current1, maxCol[j]);
							//current1 = _mm512_max_epi8(current1, vzero);
							// update maxRow and maxCol
							aux1 =  _mm512_subs_epi8(aux1, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current1, vopen_extend_gap_epi8);
							aux1 = _mm512_max_epi8(aux1, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current1);

							//calcuate the scoring matrix value
							aux5 = _mm512_shuffle_epi8(queryProfile[(i+1)*2],bIndexes_lo[j-1]);
							aux6 = _mm512_shuffle_epi8(queryProfile[(i+1)*2+1],bIndexes_hi[j-1]);
							//calcuate the diagonal value
							current2 =  _mm512_adds_epi8(previous2, _mm512_or_si512(aux5,aux6));
							// update previous
							previous2 = current1;
							// calculate current2 max value
							current2 = _mm512_max_epi8(current2, aux2);
							current2 = _mm512_max_epi8(current2, maxCol[j]);
							//current2 = _mm512_max_epi8(current2, vzero);
							// update maxRow and maxCol
							aux2 =  _mm512_subs_epi8(aux2, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current2, vopen_extend_gap_epi8);
							aux2 = _mm512_max_epi8(aux2, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current2);							

							//calcuate the scoring matrix value
							aux5 = _mm512_shuffle_epi8(queryProfile[(i+2)*2],bIndexes_lo[j-1]);
							aux6 = _mm512_shuffle_epi8(queryProfile[(i+2)*2+1],bIndexes_hi[j-1]);
							//calcuate the diagonal value
							current3 =  _mm512_adds_epi8(previous3, _mm512_or_si512(aux5,aux6));
							// update previous
							previous3 = current2;
							// calculate current3 max value
							current3 = _mm512_max_epi8(current3, aux3);
							current3 = _mm512_max_epi8(current3, maxCol[j]);
							//current3 = _mm512_max_epi8(current3, vzero);
							// update maxRow and maxCol
							aux3 =  _mm512_subs_epi8(aux3, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current3, vopen_extend_gap_epi8);
							aux3 = _mm512_max_epi8(aux3, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update max score
							score = _mm512_max_epi8(score,current3);							

							//calcuate the scoring matrix value
							aux5 = _mm512_shuffle_epi8(queryProfile[(i+3)*2],bIndexes_lo[j-1]);
							aux6 = _mm512_shuffle_epi8(queryProfile[(i+3)*2+1],bIndexes_hi[j-1]);
							//calcuate the diagonal value
							current4 =  _mm512_adds_epi8(previous4, _mm512_or_si512(aux5,aux6));
							// update previous
							previous4 = current3;
							// calculate current4 max value
							current4 = _mm512_max_epi8(current4, aux4);
							current4 = _mm512_max_epi8(current4, maxCol[j]);
							//current4 = _mm512_max_epi8(current4, vzero);
							// update maxRow and maxCol
							aux4 =  _mm512_subs_epi8(aux4, vextend_gap_epi8);
							maxCol[j] = _mm512_subs_epi8(maxCol[j], vextend_gap_epi8);
							aux0 =  _mm512_subs_epi8(current4, vopen_extend_gap_epi8);
							aux4 = _mm512_max_epi8(aux4, aux0);
							maxCol[j] =  _mm512_max_epi8(maxCol[j], aux0);	
							// update row buffer
							row2[j] = current4;
							// update max score
							score = _mm512_max_epi8(score,current4);							
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
					tmp_row = row1;
					row1 = row2;
					row2 = tmp_row;
				}
			}

			// store max value
			#pragma unroll
			for (i=0; i < 4; i++){
				aux128 = _mm512_extracti32x4_epi32  (score, i);
				aux1 = _mm512_add_epi32(_mm512_cvtepi8_epi32(aux128),v128);
				_mm512_store_si512 (ptr_scores+i, aux1);
			}
			
			// overflow detection
			// low 
			aux256[0] = _mm512_extracti32x8_epi32 (score,0);
			aux256[1] = _mm256_cmpeq_epi8(aux256[0],v127);
			overflow_low_flag = _mm256_testz_si256(aux256[1],v127); 
			// high
			aux256[0] = _mm512_extracti32x8_epi32 (score,1);
			aux256[1] = _mm256_cmpeq_epi8(aux256[0],v127);
			overflow_high_flag = _mm256_testz_si256(aux256[1],v127); 

			// if overflow
			if ((overflow_low_flag == 0) || (overflow_high_flag == 0)){

				bb1_start = overflow_low_flag;
				bb1_end = 2-overflow_high_flag;

				// recalculate using 16-bit signed integer precision
				for (bb1=bb1_start; bb1<bb1_end ; bb1++){

					// init buffers
					#pragma unroll(AVX512BW_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_set1_epi16(-32768);
					#pragma unroll(AVX512BW_UNROLL_COUNT)
					for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_set1_epi16(-32768);
					
					// set score to 0
					score = _mm512_set1_epi16(-32768);

					disp_2 = bb1*AVX512BW_INT16_VECTOR_LENGTH;

					for (k=0; k < nbb; k++){

						// calculate dim1
						disp_4 = k*block_size;
						dim1 = n[s]-disp_4;
						dim1 = (block_size < dim1 ? block_size : dim1);
						// calculate dim2
						dim2 = dim1 / DB_SEQ_LEN_MULT;

						// calculate SP sub-block length
						disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

						// init buffers
						#pragma unroll(AVX512BW_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_set1_epi16(-32768);
						#pragma unroll(AVX512BW_UNROLL_COUNT)
						for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_set1_epi16(-32768);
						auxLastCol = _mm512_set1_epi16(-32768);

						// get bIndexes
						ptr_b_block = (__m512i*)(ptr_b) + disp_4;
						#pragma unroll(AVX512BW_UNROLL_COUNT)
						for (i=0; i<dim1 ; i++ ) {
							// indexes >= 16
							bIndexes_hi[i] = _mm512_sub_epi8(ptr_b_block[i], v16);
							// indexes < 16
							mask = _mm512_cmplt_epi8_mask(ptr_b_block[i],v16);
							bIndexes_lo[i] = _mm512_mask_mov_epi8(vneg32, mask, ptr_b_block[i]);
						}

						for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
						
							// update row[0] with lastCol[i-1]
							row1[0] = lastCol[i];
							previous2 = lastCol[i+1];
							previous3 = lastCol[i+2];
							previous4 = lastCol[i+3];
							// store maxRow in auxiliars
							aux1 = maxRow[i];
							aux2 = maxRow[i+1];
							aux3 = maxRow[i+2];
							aux4 = maxRow[i+3];

							for (ii=0; ii<dim2 ; ii++) {

								#pragma unroll(DB_SEQ_LEN_MULT)
								for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {
									//calcuate the scoring matrix value
									aux5 = _mm512_shuffle_epi8(queryProfile[i*2],bIndexes_lo[j-1]);
									aux6 = _mm512_shuffle_epi8(queryProfile[i*2+1],bIndexes_hi[j-1]);
									_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
									//calcuate the diagonal value
									current1 =  _mm512_adds_epi16(row1[j-1],  _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (submatValues+disp_2))));
									// calculate current1 max value
									current1 = _mm512_max_epi16(current1, aux1);
									current1 = _mm512_max_epi16(current1, maxCol[j]);
									//current1 = _mm512_max_epi16(current1, vzero);
									// update maxRow and maxCol
									aux1 = _mm512_subs_epi16(aux1, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current1, vopen_extend_gap_epi16);
									aux1 = _mm512_max_epi16(aux1, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current1);

									//calcuate the scoring matrix value
									aux5 = _mm512_shuffle_epi8(queryProfile[(i+1)*2],bIndexes_lo[j-1]);
									aux6 = _mm512_shuffle_epi8(queryProfile[(i+1)*2+1],bIndexes_hi[j-1]);
									_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
									//calcuate the diagonal value
									current2 =  _mm512_adds_epi16(previous2,  _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (submatValues+disp_2))));
									// update previous
									previous2 = current1;
									// calculate current2 max value
									current2 = _mm512_max_epi16(current2, aux2);
									current2 = _mm512_max_epi16(current2, maxCol[j]);
									//current2 = _mm512_max_epi16(current2, vzero);
									// update maxRow and maxCol
									aux2 = _mm512_subs_epi16(aux2, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current2, vopen_extend_gap_epi16);
									aux2 = _mm512_max_epi16(aux2, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current2);

									//calcuate the scoring matrix value
									aux5 = _mm512_shuffle_epi8(queryProfile[(i+2)*2],bIndexes_lo[j-1]);
									aux6 = _mm512_shuffle_epi8(queryProfile[(i+2)*2+1],bIndexes_hi[j-1]);
									_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
									//calcuate the diagonal value
									current3 =  _mm512_adds_epi16(previous3,  _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (submatValues+disp_2))));
									// update previous
									previous3 = current2;
									// calculate current3 max value
									current3 = _mm512_max_epi16(current3, aux3);
									current3 = _mm512_max_epi16(current3, maxCol[j]);
									//current3 = _mm512_max_epi16(current3, vzero);
									// update maxRow and maxCol
									aux3 = _mm512_subs_epi16(aux3, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current3, vopen_extend_gap_epi16);
									aux3 = _mm512_max_epi16(aux3, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update max score
									score = _mm512_max_epi16(score,current3);

									//calcuate the scoring matrix value
									aux5 = _mm512_shuffle_epi8(queryProfile[(i+3)*2],bIndexes_lo[j-1]);
									aux6 = _mm512_shuffle_epi8(queryProfile[(i+3)*2+1],bIndexes_hi[j-1]);
									_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
									//calcuate the diagonal value
									current4 =  _mm512_adds_epi16(previous4,  _mm512_cvtepi8_epi16(_mm256_loadu_si256((__m256i *) (submatValues+disp_2))));
									// update previous
									previous4 = current3;
									// calculate current4 max value
									current4 = _mm512_max_epi16(current4, aux4);
									current4 = _mm512_max_epi16(current4, maxCol[j]);
									//current4 = _mm512_max_epi16(current4, vzero);
									// update maxRow and maxCol
									aux4 = _mm512_subs_epi16(aux4, vextend_gap_epi16);
									maxCol[j] = _mm512_subs_epi16(maxCol[j], vextend_gap_epi16);
									aux0 = _mm512_subs_epi16(current4, vopen_extend_gap_epi16);
									aux4 = _mm512_max_epi16(aux4, aux0);
									maxCol[j] =  _mm512_max_epi16(maxCol[j], aux0);	
									// update row buffer
									row2[j] = current4;
									// update max score
									score = _mm512_max_epi16(score,current4);
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
							tmp_row = row1;
							row1 = row2;
							row2 = tmp_row;
						}
					}
					// store max value (low)
					aux256[0] = _mm512_extracti32x8_epi32 (score,0);
					aux1 = _mm512_add_epi32(_mm512_cvtepi16_epi32(aux256[0]),v32768);
					_mm512_store_si512 (ptr_scores+bb1*2,aux1);
					// check overflow (low)
					aux256[1] = _mm256_cmpeq_epi16(aux256[0],v32767);
					overflow_low_flag = _mm256_testz_si256(aux256[1],v127); 
					// store max value (high)
					aux256[0] = _mm512_extracti32x8_epi32 (score,1);
					aux1 = _mm512_add_epi32(_mm512_cvtepi16_epi32(aux256[0]),v32768);
					_mm512_store_si512 (ptr_scores+bb1*2+1,aux1);
					// check overflow (high)
					aux256[1] = _mm256_cmpeq_epi16(aux256[0],v32767);
					overflow_high_flag = _mm256_testz_si256(aux256[1],v32767); 

					// if overflow
					if ((overflow_low_flag == 0) || (overflow_high_flag == 0)){

						// check overflow in low 16 bits
						bb2_start = overflow_low_flag;
						// check overflow in high 16 bits
						bb2_end = 2 - overflow_high_flag;

						// recalculate using 32-bit signed integer precision
						for (bb2=bb2_start; bb2<bb2_end ; bb2++){

							// init buffers
							#pragma unroll(AVX512BW_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) maxRow[i] = _mm512_setzero_si512();
							#pragma unroll(AVX512BW_UNROLL_COUNT)
							for (i=0; i<m[q] ; i++ ) lastCol[i] = _mm512_setzero_si512();
							
							// set score to 0
							score = _mm512_setzero_si512();

							disp_3 = disp_2 + bb2*AVX512BW_INT32_VECTOR_LENGTH;

							for (k=0; k < nbb; k++){

								// calculate dim1
								disp_4 = k*block_size;
								dim1 = n[s]-disp_4;
								dim1 = (block_size < dim1 ? block_size : dim1);
								// calculate dim2
								dim2 = dim1 / DB_SEQ_LEN_MULT;

								// calculate SP sub-block length
								disp_1 = dim1*AVX512BW_INT8_VECTOR_LENGTH;

								// init buffers
								#pragma unroll(AVX512BW_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) maxCol[i] = _mm512_setzero_si512();
								#pragma unroll(AVX512BW_UNROLL_COUNT)
								for (i=0; i<dim1+1 ; i++ ) row1[i] = _mm512_setzero_si512();
								auxLastCol = _mm512_setzero_si512();

								// get bIndexes
								ptr_b_block = (__m512i*)(ptr_b) + disp_4;
								#pragma unroll(AVX512BW_UNROLL_COUNT)
								for (i=0; i<dim1 ; i++ ) {
									// indexes >= 16
									bIndexes_hi[i] = _mm512_sub_epi8(ptr_b_block[i], v16);
									// indexes < 16
									mask = _mm512_cmplt_epi8_mask(ptr_b_block[i],v16);
									bIndexes_lo[i] = _mm512_mask_mov_epi8(vneg32, mask, ptr_b_block[i]);
								}


								for( i = 0; i < m[q]; i+=QUERY_SEQ_LEN_MULT){
								
									// update row[0] with lastCol[i-1]
									row1[0] = lastCol[i];
									previous2 = lastCol[i+1];
									previous3 = lastCol[i+2];
									previous4 = lastCol[i+3];
									// store maxRow in auxiliars
									aux1 = maxRow[i];
									aux2 = maxRow[i+1];
									aux3 = maxRow[i+2];
									aux4 = maxRow[i+3];

									for (ii=0; ii<dim2 ; ii++) {

										#pragma unroll(DB_SEQ_LEN_MULT)
										for( j=ii*DB_SEQ_LEN_MULT+1, jj=0; jj < DB_SEQ_LEN_MULT;  jj++, j++) {

											//calcuate the scoring matrix value
											aux5 = _mm512_shuffle_epi8(queryProfile[(i)*2],bIndexes_lo[j-1]);
											aux6 = _mm512_shuffle_epi8(queryProfile[(i)*2+1],bIndexes_hi[j-1]);
											_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
											//calcuate the diagonal value
											current1 =  _mm512_add_epi32(row1[j-1],  _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (submatValues+disp_3))));
											// update previous
											previous2 = current1;
											// calculate current1 max value
											current1 = _mm512_max_epi32(current1, aux1);
											current1 = _mm512_max_epi32(current1, maxCol[j]);
											current1 = _mm512_max_epi32(current1, vzero);
											// update maxRow and maxCol
											aux1 = _mm512_sub_epi32(aux1, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current1, vopen_extend_gap_epi32);
											aux1 = _mm512_max_epi32(aux1, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current1);

											//calcuate the scoring matrix value
											aux5 = _mm512_shuffle_epi8(queryProfile[(i+1)*2],bIndexes_lo[j-1]);
											aux6 = _mm512_shuffle_epi8(queryProfile[(i+1)*2+1],bIndexes_hi[j-1]);
											_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
											//calcuate the diagonal value
											current2 =  _mm512_add_epi32(previous2,  _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (submatValues+disp_3))));
											// update previous
											previous2 = current1;
											// calculate current2 max value
											current2 = _mm512_max_epi32(current2, aux2);
											current2 = _mm512_max_epi32(current2, maxCol[j]);
											current2 = _mm512_max_epi32(current2, vzero);
											// update maxRow and maxCol
											aux2 = _mm512_sub_epi32(aux2, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current2, vopen_extend_gap_epi32);
											aux2 = _mm512_max_epi32(aux2, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current2);

											//calcuate the scoring matrix value
											aux5 = _mm512_shuffle_epi8(queryProfile[(i+2)*2],bIndexes_lo[j-1]);
											aux6 = _mm512_shuffle_epi8(queryProfile[(i+2)*2+1],bIndexes_hi[j-1]);
											_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
											//calcuate the diagonal value
											current3 =  _mm512_add_epi32(previous3,  _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (submatValues+disp_3))));
											// update previous
											previous3 = current2;
											// calculate current3 max value
											current3 = _mm512_max_epi32(current3, aux3);
											current3 = _mm512_max_epi32(current3, maxCol[j]);
											current3 = _mm512_max_epi32(current3, vzero);
											// update maxRow and maxCol
											aux3 = _mm512_sub_epi32(aux3, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current3, vopen_extend_gap_epi32);
											aux3 = _mm512_max_epi32(aux3, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update max score
											score = _mm512_max_epi32(score,current3);

											//calcuate the scoring matrix value
											aux5 = _mm512_shuffle_epi8(queryProfile[(i+3)*2],bIndexes_lo[j-1]);
											aux6 = _mm512_shuffle_epi8(queryProfile[(i+3)*2+1],bIndexes_hi[j-1]);
											_mm512_store_si512(submatValues, _mm512_or_si512(aux5,aux6));
											//calcuate the diagonal value
											current4 =  _mm512_add_epi32(previous4,  _mm512_cvtepi8_epi32(_mm_loadu_si128((__m128i *) (submatValues+disp_3))));
											// update previous
											previous4 = current3;
											// calculate current4 max value
											current4 = _mm512_max_epi32(current4, aux4);
											current4 = _mm512_max_epi32(current4, maxCol[j]);
											current4 = _mm512_max_epi32(current4, vzero);
											// update maxRow and maxCol
											aux4 = _mm512_sub_epi32(aux4, vextend_gap_epi32);
											maxCol[j] = _mm512_sub_epi32(maxCol[j], vextend_gap_epi32);
											aux0 = _mm512_sub_epi32(current4, vopen_extend_gap_epi32);
											aux4 = _mm512_max_epi32(aux4, aux0);
											maxCol[j] =  _mm512_max_epi32(maxCol[j], aux0);	
											// update row buffer
											row2[j] = current4;
											// update max score
											score = _mm512_max_epi32(score,current4);

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
									tmp_row = row1;
									row1 = row2;
									row2 = tmp_row;
								}
							}
							// store max value
							_mm512_store_si512 (ptr_scores+bb1*2+bb2,score);
						}
					}
				}
			}
		}

		 _mm_free(row1);  _mm_free(row2); _mm_free(maxCol); _mm_free(maxRow); _mm_free(lastCol); _mm_free(bIndexes_lo);_mm_free(bIndexes_hi);
	}

	_mm_free(queryProfiles);

	*workTime = dwalltime()-tick;

}

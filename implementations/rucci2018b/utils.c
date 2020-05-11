#include "utils.h"

// tune number of threads and block size considering processor
void tuning (int * threads, int * block_size, unsigned short int * vect_db_sequences_lengths,
	unsigned short int * vect_db_sequences_blocks, unsigned long int vect_db_sequences_count){

	#if AVX512F
	int * submat = iBlosum62;
	#else
	char * submat = cBlosum62;
	#endif
	int open_gap=OPEN_GAP, extend_gap=EXTEND_GAP, b1, b2, qp_block_size=0, sp_block_size=0;
	double workTime, totalTime=0, bestTime=0;
	unsigned long int i;
	char tuning_filename[30]=".swimm2_sse41_tuning";

	#if AVX512F && KNL
		strcpy(tuning_filename,".swimm2_knl_avx512f_tuning");
	#elif AVX512F
		strcpy(tuning_filename,".swimm2_avx512f_tuning");
	#elif AVX2 && KNL
		strcpy(tuning_filename,".swimm2_knl_avx2_tuning");
	#elif AVX2
		strcpy(tuning_filename,".swimm2_avx2_tuning");
	#elif AVX512BW
		strcpy(tuning_filename,".swimm2_avx512bw_tuning");
	#endif


	// tune number of threads (only if the user did not set it)
	if (*threads == 0) 
		*threads =  sysconf (_SC_NPROCESSORS_ONLN);


	// open tuning filen
	FILE * tuning_file = fopen(tuning_filename,"r");

	if (tuning_file == NULL)	{

		printf("\nAuto-tuning. This step may take some minutes but is executed only once after build... ");
		fflush(stdout);


		char * tun_vect_db_sequences, * tun_query_sequence;
		unsigned short int * tun_vect_db_sequence_lengths, * tun_vect_db_sequence_blocks, * tun_query_sequence_lengths;
		unsigned long int tun_vect_db_sequences_count, * tun_vect_db_sequence_disp;
		unsigned int  * tun_query_sequence_disps;
		int * scores;

		// load synthetic query sequence
		load_tuning_query_sequence (&tun_query_sequence, &tun_query_sequence_lengths, &tun_query_sequence_disps);
		
		// load synthetic database sequence
		assemble_tuning_chunk_db (&tun_vect_db_sequences, &tun_vect_db_sequence_lengths, &tun_vect_db_sequence_blocks, 
								&tun_vect_db_sequence_disp, &tun_vect_db_sequences_count) ;

		// alloc memory for scores buffer
		scores = _mm_malloc (TUNING_QUERY_COUNT*tun_vect_db_sequences_count*sizeof(int)*VECTOR_LENGTH, MEMALIGN);

		// configure block size for data locality (Score Profile)
		for (b1=TUNING_MIN_BLOCK_SIZE; b1<=TUNING_MAX_BLOCK_SIZE ; b1+=TUNING_BLOCK_SIZE_STEP){

			// adapt block size
			b2 = (b1 / DB_SEQ_LEN_MULT) * DB_SEQ_LEN_MULT;

			// re-calculate number of blocks
			for (i=0; i< tun_vect_db_sequences_count; i++ ) 
				tun_vect_db_sequence_blocks[i] = ceil( (double) tun_vect_db_sequence_lengths[i] / (double) b2);

			// set accumulator
			totalTime=0;
			// repeat test
			for (i=0; i< TUNING_REPEAT_TIMES; i++ )  {

				workTime = dwalltime();

				#if AVX512F
					// search using AVX512F instrucions and Adaptive Profile technique
					search_avx512f_ap (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, SCORE_PROFILE, query_length_threshold, 
							tun_vect_db_sequences, tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, (__m512i*)intSubmat, 
							open_gap, extend_gap, *threads, b2, (__m512i*)scores, 	&workTime);

				#elif AVX512BW
						// search using AVX512BW instrucions and Score Profile technique
						search_avx512bw_sp (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, tun_vect_db_sequences,
							tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, charSubmat, open_gap, extend_gap,
							*threads, b2, scores, &workTime);
				#elif SSE41
					// search using SSE4.1 instrucions and Score Profile technique
					search_sse41_sp (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, tun_vect_db_sequences,
						tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, charSubmat, open_gap, extend_gap,
						*threads, b2, scores, &workTime);
				#else
					// database search using AVX2 instrucions and Score Profile technique
					search_avx2_sp (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, tun_vect_db_sequences,
							tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, charSubmat, open_gap, extend_gap,
						*threads, b2, scores, &workTime);
				#endif

				totalTime += workTime;
			}

			if (bestTime == 0){
				bestTime = totalTime;
				sp_block_size = b2;
			} else {
				if (totalTime < bestTime) {
					bestTime = totalTime;
					sp_block_size = b2;
				}
			}
		}

		#if AVX512BW || AVX512F
		bestTime=0;

			// configure block size for data locality (Query profile)
			for (b1=TUNING_MIN_BLOCK_SIZE; b1<=TUNING_MAX_BLOCK_SIZE ; b1+=TUNING_BLOCK_SIZE_STEP){

				// adapt block size
				b2 = (b1 / DB_SEQ_LEN_MULT) * DB_SEQ_LEN_MULT;

				// re-calculate number of blocks
				for (i=0; i< tun_vect_db_sequences_count; i++ ) 
					tun_vect_db_sequence_blocks[i] = ceil( (double) tun_vect_db_sequence_lengths[i] / (double) b2);

				// set accumulator
				totalTime=0;
				// repeat test
				for (i=0; i< TUNING_REPEAT_TIMES; i++ )  {

					workTime = dwalltime();

					#if AVX512F
						// search using AVX512F instrucions and Adaptive Profile technique
						search_avx512f_ap (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, QUERY_PROFILE, query_length_threshold, 
								tun_vect_db_sequences, tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, (__m512i*)intSubmat, 
								open_gap, extend_gap, *threads, b2, (__m512i*)scores, 	&workTime);

					#elif AVX512BW
						// search using AVX512bw instrucions and Query Profile technique
						search_avx512bw_qp (tun_query_sequence, tun_query_sequence_lengths, TUNING_QUERY_COUNT, tun_query_sequence_disps, tun_vect_db_sequences,
							tun_vect_db_sequence_lengths, tun_vect_db_sequence_blocks, tun_vect_db_sequences_count, tun_vect_db_sequence_disp, charSubmat, open_gap, extend_gap, 
							*threads, b2, scores, &workTime);
					#endif

					totalTime += workTime;
				}

				if (bestTime == 0){
					bestTime = totalTime;
					qp_block_size = b2;
				} else {
					if (totalTime < bestTime) {
						bestTime = totalTime;
						qp_block_size = b2;
					}
				}
			}
		#endif

		// create file, save block size, close file
		tuning_file = fopen(tuning_filename,"w");
		fprintf(tuning_file,"%d %d",sp_block_size,qp_block_size);
		fclose(tuning_file);

		printf("Done.\n");

		*block_size = (profile == QUERY_PROFILE ? qp_block_size : sp_block_size);


		// re-calculate number of blocks
		for (i=0; i< vect_db_sequences_count; i++ ) 
			vect_db_sequences_blocks[i] = ceil( (double) vect_db_sequences_lengths[i] / (double) (*block_size));

		_mm_free(tun_query_sequence);
		_mm_free(tun_query_sequence_lengths);
		_mm_free(tun_query_sequence_disps);
		_mm_free(tun_vect_db_sequences);
		_mm_free(tun_vect_db_sequence_lengths);
		_mm_free(tun_vect_db_sequence_blocks);
		_mm_free(tun_vect_db_sequence_disp);
		_mm_free(scores);

	} else {

		printf("\nAuto-tuning... Skipped.\n");

		// retrieve block size, close file
		fscanf(tuning_file,"%d %d",&sp_block_size,&qp_block_size);
		fclose(tuning_file);

		*block_size = (profile == SCORE_PROFILE ? sp_block_size : qp_block_size);

		// re-calculate number of blocks
		for (i=0; i< vect_db_sequences_count; i++ ) 
			vect_db_sequences_blocks[i] = ceil( (double) vect_db_sequences_lengths[i] / (double) (*block_size));

	}

}

void merge_scores(int * scores, char ** titles, unsigned long int size) {
	unsigned long int i1 = 0;
	unsigned long int i2 = size / 2;
	unsigned long int it = 0;
	// allocate memory for temporary buffers
	char ** tmp2 = (char **) malloc(size*sizeof(char *));
	int * tmp3 = (int *) malloc (size*sizeof(int));

	while(i1 < size/2 && i2 < size) {
		if (scores[i1] > scores[i2]) {
			tmp2[it] = titles[i1];
			tmp3[it] = scores[i1];
			i1++;
		}
		else {
			tmp2[it] = titles[i2];
			tmp3[it] = scores[i2];
			i2 ++;
		}
		it ++;
	}

	while (i1 < size/2) {
		tmp2[it] = titles[i1];
		tmp3[it] = scores[i1];
	    i1++;
	    it++;
	}
	while (i2 < size) {
		tmp2[it] = titles[i2];
		tmp3[it] = scores[i2];
	    i2++;
	    it++;
	}

	memcpy(titles, tmp2, size*sizeof(char *));
	memcpy(scores, tmp3, size*sizeof(int));

	free(tmp2);
	free(tmp3);

}


void mergesort_scores_serial(int * scores, char ** titles, unsigned long int size) {
	int tmp_score;
	char * tmp_seq;

	if (size == 2) { 
		if (scores[0] <= scores[1]) {
			// swap scores
			tmp_score = scores[0];
			scores[0] = scores[1];
			scores[1] = tmp_score;
			// swap titles
			tmp_seq = titles[0];
			titles[0] = titles[1];
			titles[1] = tmp_seq;
		}
	} else {
		if (size > 2){
			mergesort_scores_serial(scores, titles, size/2);
			mergesort_scores_serial(scores + size/2, titles + size/2, size - size/2);
			merge_scores(scores, titles, size);
		}
	}
}

void sort_scores (int * scores, char ** titles, unsigned long int size, int threads) {
    if ( threads == 1) {
	      mergesort_scores_serial(scores, titles, size);
    }
    else if (threads > 1) {
        #pragma omp parallel sections num_threads(threads)
        {
            #pragma omp section
            sort_scores(scores, titles, size/2, threads/2);
            #pragma omp section
            sort_scores(scores + size/2, titles  + size/2, size-size/2, threads-threads/2);
        }

        merge_scores(scores, titles, size);
    } // threads > 1
}

// Wall time
double dwalltime()
{
	double sec;
	struct timeval tv;

	gettimeofday(&tv,NULL);
	sec = tv.tv_sec + tv.tv_usec/1000000.0;
	return sec;
}
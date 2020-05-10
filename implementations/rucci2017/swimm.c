#include "swimm.h"

// Global options
char *sequences_filename=NULL, * queries_filename=NULL, *input_filename=NULL, * output_filename=NULL, *op=NULL, submat_name[]="BLOSUM62", profile=SCORE_PROFILE;
int threads=0, block_size=BLOCK_SIZE, open_gap=OPEN_GAP, extend_gap=EXTEND_GAP, vector_length=VECTOR_LENGTH;
unsigned short int query_length_threshold = QUERY_LENGTH_THRESHOLD;
unsigned long int top=TOP;
#if AVX512F
	int *intSubmat=iBlosum62;
#else
	char* charSubmat= cBlosum62;
#endif

int main(int argc, char *argv[]) {

	unsigned long int i, j, sequences_count, D, vect_db_sequences_count, vD, * chunk_vD, * vect_db_sequences_disp, query_sequences_count, Q;
	unsigned int chunk_count, * chunk_vect_db_sequences_count, ** chunk_vect_db_sequences_disp, * query_sequences_disp;
	int max_title_length, *scores;
	unsigned short int ** chunk_vect_db_sequences_lengths, * vect_db_sequences_lengths, * vect_db_sequences_blocks, sequences_db_max_length, * query_sequences_lengths, *m;
	char ** chunk_vect_db_sequences, * vect_db_sequences, *query_sequences, ** query_headers, ** sequence_db_headers, ** tmp_sequence_db_headers, simd_set[20]="SSE4.1";
    time_t current_time = time(NULL);
	double workTime, tick;
	
	/* Process program arguments */
	program_arguments_processing(argc,argv);

	/* Database preprocessing */
	if (strcmp(op,"preprocess") == 0)
		preprocess_db (input_filename,output_filename,threads); 
	else {
		/* Database search */
		// Print database search information
		printf("\nSWIMM v%s \n\n",VERSION);
		printf("Database file:\t\t\t%s\n",sequences_filename);

		// Load query sequence from file in a
		load_query_sequences(queries_filename,&query_sequences,&query_headers,&query_sequences_lengths,&m,&query_sequences_count,&Q,&query_sequences_disp,threads);

		// Assemble database 
		assemble_single_chunk_db (sequences_filename, vector_length, &sequences_count, &D, &sequences_db_max_length, &max_title_length, &vect_db_sequences_count, &vD, 
			&vect_db_sequences,	&vect_db_sequences_lengths, &vect_db_sequences_blocks, &vect_db_sequences_disp, threads, block_size);

		// Allocate buffers 
		top = (sequences_count < top ? sequences_count : top);
		scores = (int*) _mm_malloc(query_sequences_count*(vect_db_sequences_count*vector_length)*sizeof(int), MEMALIGN);
		tmp_sequence_db_headers = (char**) malloc(sequences_count*sizeof(char *));

		// Print database search information
		printf("Database size:\t\t\t%ld sequences (%ld residues) \n",sequences_count,D);
		printf("Longest database sequence: \t%d residues\n",sequences_db_max_length);
		printf("Substitution matrix:\t\t%s\n",submat_name);
		printf("Gap open penalty:\t\t%d\n",open_gap);
		printf("Gap extend penalty:\t\t%d\n",extend_gap);
		printf("Query filename:\t\t\t%s\n",queries_filename);

		// retrieve number of threads and block size. re-calculate block sizes
		tuning (&threads, &block_size, vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count);

		printf("\nSearching... ");
		fflush(stdout);

		workTime = dwalltime();

		#if AVX512F
			// search using AVX512F instrucions and Adaptive Profile technique
			search_avx512f_ap (query_sequences, m, query_sequences_count, query_sequences_disp, profile, query_length_threshold, 
					vect_db_sequences, vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count, vect_db_sequences_disp, (__m512i*)intSubmat, 
					open_gap, extend_gap, threads, block_size, (__m512i*)scores, 	&workTime);

		#elif AVX512BW
			if (profile == QUERY_PROFILE)
				// search using AVX512bw instrucions and Query Profile technique
				search_avx512bw_qp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_db_sequences,
					vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count, vect_db_sequences_disp, charSubmat, open_gap, extend_gap, threads, block_size, scores,
					&workTime);
			else 
				// search using AVX512BW instrucions and Score Profile technique
				search_avx512bw_sp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_db_sequences,
					vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count, vect_db_sequences_disp, charSubmat, open_gap, extend_gap, threads, block_size, scores,
					&workTime);
		#elif SSE41
			// search using SSE4.1 instrucions and Score Profile technique
			search_sse41_sp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_db_sequences,
				vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count, vect_db_sequences_disp, charSubmat, open_gap, extend_gap, threads, block_size, scores,
				&workTime);
		#else
			// database search using AVX2 instrucions and Score Profile technique
			search_avx2_sp (query_sequences, m, query_sequences_count, query_sequences_disp, vect_db_sequences,
					vect_db_sequences_lengths, vect_db_sequences_blocks, vect_db_sequences_count, vect_db_sequences_disp, charSubmat, open_gap, extend_gap, threads, block_size, scores,
					&workTime);
		#endif

		printf("Done.\n");


		// Free allocated memory
		_mm_free(query_sequences);
		_mm_free(query_sequences_disp);
		_mm_free(m);
		_mm_free(vect_db_sequences);
		_mm_free(vect_db_sequences_lengths);
		_mm_free(vect_db_sequences_disp);

		// Load database headers
		load_database_headers (sequences_filename, sequences_count, max_title_length, &sequence_db_headers);

		// allow nested paralelism
		omp_set_nested(1);

		#if AVX512F
			strcpy(simd_set,"AVX-512F");
		#elif AVX512BW
			strcpy(simd_set,"AVX-512BW");
		#elif AVX2
			strcpy(simd_set,"AVX2");
		#endif

		// Print top scores
		for (i=0; i<query_sequences_count ; i++ ) {
			memcpy(tmp_sequence_db_headers,sequence_db_headers,sequences_count*sizeof(char *));
			sort_scores(scores+i*vect_db_sequences_count*vector_length,tmp_sequence_db_headers,sequences_count,threads);
			printf("\nQuery no.\t\t\t%d\n",i+1);
			printf("Query description: \t\t%s\n",query_headers[i]+1);
			printf("Query length:\t\t\t%d residues\n",query_sequences_lengths[i]);
			printf("\nScore\tSequence description\n");
			for (j=0; j<top; j++) 
				printf("%d\t%s",scores[i*vect_db_sequences_count*vector_length+j],tmp_sequence_db_headers[j]+1);
		}
		printf("\nSearch date:\t\t\t%s",ctime(&current_time));
		printf("Search time:\t\t\t%lf seconds\n",workTime);
		printf("Search speed:\t\t\t%.2lf GCUPS\n",(Q*D) / (workTime*1000000000));
		printf("Execution mode:\t\t\t%d threads, %s instructions\n",threads,simd_set);
		printf("Profile technique:\t\t%s\n",(profile == SCORE_PROFILE ? "Score" : (profile == QUERY_PROFILE ? "Query" : "Adaptive")));

		// Free allocated memory
		_mm_free(query_sequences_lengths);
		_mm_free(scores); 	
		for (i=0; i<query_sequences_count ; i++ ) 
			free(query_headers[i]);
		free(query_headers);
		for (i=0; i<sequences_count ; i++ ) 
			free(sequence_db_headers[i]);
		free(sequence_db_headers);
		free(tmp_sequence_db_headers);

	}

	return 0;
}


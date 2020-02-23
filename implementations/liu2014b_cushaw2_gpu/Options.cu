/*
 * Options.cpp
 *
 *  Created on: Jan 10, 2012
 *      Author: yongchao
 */

#include "Options.h"
#include "Utils.h"
#include "GPUMacros.h"

Options::Options() {
	/*set defaults*/
	_setDefaults();
}
Options::Options(int argc, char* argv[]) {
	/*set defaults*/
	_setDefaults();

	/*parse the command line*/
	parse(argc, argv);
}
Options::~Options() {
	_readsFileNames.clear();
	delete[] _numErrorTable;
	delete[] _minSeedSizeTable;
	pthread_mutex_destroy(&globalMutex);
}
void Options::printUsage() {
	fprintf(stderr,
			"--------------------------------------------------------------------------------\n\n");
#ifdef HAVE_WRAPPER
	fprintf(stderr,
			"CUSHAW2-GPU-WRAPPER (v%s) is a program to launch CUSHAW2-GPU for multiple GPUs sharing a host",
			CUSHAW2_VERSION);
	fprintf(stderr,
			"--------------------------------------------------------------------------------\n\n");

	fprintf(stderr, "Usage: cushaw2-gpu-wrapper -r bwt [options]\n");
#else
	fprintf(stderr,
			"CUSHAW2-GPU (v%s) is a GPU-based long-read aligner based on Burrows-Wheeler transform\n",
			CUSHAW2_VERSION);

	fprintf(stderr,
			"--------------------------------------------------------------------------------\n\n");

	fprintf(stderr, "Usage: cushaw2-gpu -r bwt [options]\n");
#endif

	fprintf(stderr, "\tbwt: the file name base for the reference genome\n");

	/*the file input options*/
	fprintf(stderr, "Input:\n");
	fprintf(stderr,
			"\t-r <string> (the file name base for the reference genome)\n");

	/*single-end reads*/
	fprintf(stderr,
			"\t-f <string> file1 [file2] (single-end sequence files in FASTA/FASTQ format)\n");
	fprintf(stderr,
			"\t-b <string> file1 [file2] (single-end sequence files in BAM format)\n");
	fprintf(stderr,
			"\t-s <string> file1 [file2] (single-end sequence files in SAM format)\n");

	/*paired-end reads*/
	fprintf(stderr,
			"\t-q <string> file1_1 file1_2  [file2_1 file2_2] (paired-end sequence files in FASTA/FASTQ format)\n");

	/*output*/
	fprintf(stderr, "Output:\n");
#ifdef HAVE_WRAPPER
	fprintf(stderr, "\t-o <string> (SAM output file path prefix, required. Each GPU process has its own output file with prefix $o)\n");
#else
	fprintf(stderr, "\t-o <string> (SAM output file path, default = %s)\n",
			_samFileName.length() > 0 ? _samFileName.c_str() : "STDOUT");
#endif
	fprintf(stderr,
			"\t-unique <bool> (only output unique alignments [with mapping quality scores > 0], default = %d)\n",
			_onlyUnique);
	fprintf(stderr,
			"\t-multi <int> (output <= #int [1, %d] equivalent best alignments, default = %d)\n",
			MAX_MULTI_ALIGN, _maxMultiAligns);
	/*read group information*/
	fprintf(stderr,
			"\n\t/*read group information in SAM header as follows*/\n");
	fprintf(stderr, "\t-rgid <string> (read group identifier [tag RG:ID])\n");
	fprintf(stderr,
			"\t-rgsm <string> (read group sample name [tag RG:SM], required if #rgid is given)\n");
	fprintf(stderr,
			"\t-rglb <string> (read group library [tag RG:LD], ineffective if #rgid is not given)\n");
	fprintf(stderr,
			"\t-rgpl <string> (read group platform/technology [tag RG:PL], ineffective if #rigid is not given)\n\t\tsupported values: capillary, ls454, illumina, solid, helicos, iontorrent, and pacbio\n");
	fprintf(stderr,
			"\t-rgpu <string> (read group platform unit identifier [tag RG:PU], ineffective if #rgid is not given)\n");
	fprintf(stderr,
			"\t-rgcn <string> (name of sequencing center produced the read [tag RG:CN], ineffiective if #rgid is not given)\n");
	fprintf(stderr,
			"\t-rgds <string> (description about the reads [tag RG:DS], ineffiective if #rgid is not given)\n");
	fprintf(stderr,
			"\t-rgdt <string> (date on which the run was produced [tag RG:DT], ineffiective if #rgid is not given)\n");
	fprintf(stderr,
			"\t-rgpi <string> (predicated median insert size [tag RG:PI], ineffiective if #rgid is not given)\n");

	/*alignment options*/
	fprintf(stderr, "Scoring:\n");
	fprintf(stderr, "\t-match <int> (score for a match, default = %d)\n",
			_match);
	fprintf(stderr,
			"\t-mismatch <int> (penalty for a mismatch, default = %d)\n",
			_mismatch);
	fprintf(stderr, "\t-gopen <int> (gap open penalty, default = %d)\n",
			_gapOpen);
	fprintf(stderr, "\t-gext <int> (gap extension penalty, default = %d)\n",
			_gapExtend);

	fprintf(stderr, "Align:\n");
	fprintf(stderr,
			"\t-sensitive (concerned more about the sensitivity, only using min_score)\n");
	fprintf(stderr,
			"\t-min_score <int> (minimal optimal local alignment score divided by matching score, default = %d)\n",
			_minAlginScore);
	fprintf(stderr,
			"\t-min_id <float> (minimal identity of  optical local alignments, default = %.2f)\n",
			_minIdentity);
	fprintf(stderr,
			"\t-min_ratio <float> (minimal ratio of reads in optimal local alignments, default = %.2f)\n",
			_minRatio);

	fprintf(stderr, "Seed:\n");
	fprintf(stderr,
			"\t-min_seed <int> (lower bound of minimal seed size, default = %d)\n",
			_lowerMinSeedSize);
	fprintf(stderr,
			"\t-max_seed <int> (upper bound of minimal seed size, default = %d)\n",
			_upperMinSeedSize);
	fprintf(stderr,
			"\t-miss_prob <float> (missing probability to estimate the seed sizes, default = %.2f)\n",
			_missProb);
	fprintf(stderr,
			"\t-max_occ <int> (maximal number of occurrences per seed, default = %d)\n",
			_maxSeedOcc);

	fprintf(stderr, "Pairing:\n");

	/*output insert size information*/
	if (_estInsertSize) {
		fprintf(stderr,
				"\t-avg_ins <int> (insert size for paired-end reads, estimated from input if not specified)\n");

		fprintf(stderr,
				"\t-ins_std <int> (standard deviation of insert size, estimated from input if not specified)\n");
		fprintf(stderr,
				"\t-ins_npairs <int> (top number of read pairs for insert size estimation [#int times 0x%x], default = %d)\n",
				INS_SIZE_EST_MULTIPLE, _topReadsEstIns / INS_SIZE_EST_MULTIPLE);
		fprintf(stderr,
				"\t-ins_mapq <int> (minimal mapping quality score of a SE alignment for insert size estimation, default = %d)\n",
				_mapQualReliable);
	} else {
		fprintf(stderr,
				"\t-avg_ins <int> (insert size for paired-end reads, default = %d)\n",
				_insertSize);

		fprintf(stderr,
				"\t-ins_std <int> (standard deviation of insert size for paired-end reads, default = %d)\n",
				_stdInsertsize);
	}
	fprintf(stderr,
			"\t-no_rescue (do not rescue the mate using Smith-Waterman for an un-paired read)\n");
#ifdef HAVE_TWICE_RESCUE
	fprintf(stderr, "\t-no_rescue_twice (do not attempt the second rescuing for an un-paired read)\n");
#endif

	/*compute*/
	fprintf(stderr, "Compute:\n");
#ifdef HAVE_WRAPPER
	fprintf(stderr, "\t-t <int> (number of extra CPU threads, default = %d. Will be evenly distributed over all GPU processes)\n",
			_numCPUs);
	fprintf(stderr,
			"\t-g <int> (number of GPUs, default = %d)\n",
			_numGPUs);
#else
	fprintf(stderr, "\t-t <int> (number of extra CPU threads, default = %d)\n",
			_numCPUs);
	fprintf(stderr,
				"\t-g <int> (index of the GPU, default = %d, -1 to disable the use of GPUs)\n",
				_gpuIndex);
#endif
	fprintf(stderr, "Others:\n");
	fprintf(stderr, "\t-h <print out the usage of the program)\n");
}
void Options::_setDefaults() {
	/*parameters for file input*/
	_bwtFileBase = "";

	/*empty string means outputing to STDOUT*/
	_samFileName = "";
	_readsFileNames.clear();

	/*parameters for alignment*/
	_minRatio = DEFAULT_SEQ_LENGTH_RATIO; /*minimal portion for bases in a short read*/
	_minIdentity = DEFAULT_MIN_IDENTITY; /*minimal identity*/
	_numCPUs = 0;
	_numGPUs = 1;
	_gpuIndex = 0;
	_numThreads = _numCPUs + _numGPUs; /*the number of threads used for the alignment*/
	_estInsertSize = true; /*estimate the insert size from input paired-rend reads*/
	_insertSize = 500; /*the insert size for paired-end reads*/
	_stdInsertsize = (int) (_insertSize * 0.1); /*the standard deviation of insert size for paired-end reads*/
	_topReadsEstIns = 0x10000;
	_mapQualReliable = 20;
	_maxMultiAligns = 1;
	_missProb = 0.04;
	_onlyUnique = false;
	_paired = false;
	_viaFifo = false;
	_lowerMinSeedSize = 13;
	_upperMinSeedSize = GLOBAL_MAX_SEED_SIZE;
	_maxSeedOcc = GLOBAL_MAX_NUM_SEED_REPEATS;
	_rescueMate = true;
#ifdef HAVE_TWICE_RESCUE
	_rescueTwice = true;
#endif

	/*penalties*/
	/*scoring scheme 1*/
	_match = 1; /*score for matching*/
	_mismatch = 3; /*penalty for mismatching*/
	_gapOpen = 5; /*penalty for gap open*/
	_gapExtend = 2; /*penalty for gap extension*/

	/*minial local alignment score*/
	_minAlginScore = 30 * _match;

	/*estimate errors*/
	_numErrorTable = new int[MAX_SEQ_LENGTH + 1];
	if (_numErrorTable == NULL) {
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}
	_minSeedSizeTable = new int[MAX_SEQ_LENGTH + 1];
	if (_minSeedSizeTable == NULL) {
		Utils::exit("Memory allocation failed in function %s line %d\n",
				__FUNCTION__, __LINE__);
	}

	/*global lock*/
	pthread_mutex_init(&globalMutex, NULL);
}
bool Options::parse(int argc, char* argv[]) {
	bool sensitivity = false;
	bool newInsertSize = false;
	bool newStdInsertSize = false;
	int intVal;
	double floatVal;
	int argind = 1;

	if (argc < 2) {
		return false;
	}

	/*check the help*/
	if (!strcmp(argv[argind], "-h") || !strcmp(argv[argind], "-?")) {
		return false;
	}

	/*print out the command line*/
	fprintf(stderr, "Command: ");
	for (int i = 0; i < argc; ++i) {
		fprintf(stderr, "%s ", argv[i]);
	}
	fputc('\n', stderr);

	/*check the availability of GPUs*/
	GPUInfo *gpuInfo = GPUInfo::getGPUInfo();
	if (gpuInfo->getNumGPUs() == 0) {
		Utils::exit("No compatible GPUs are available in your machine\n");
	}

	/*for the other options*/
	bool first = true;
	bool done = false;
	while (argind < argc) {
		/*single-end sequence files*/
		if (!strcmp(argv[argind], "-r")) {
			argind++;
			if (argind < argc) {
				_bwtFileBase = argv[argind];
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

		} else if (!strcmp(argv[argind], "-f")) {
			/*specify the inputs are single-end*/
			if (first) {
				first = false;
				setPaired(false);
			} else if (isPaired()) {
				Utils::log(
						"Cannot specify single-end and paired-end reads together\n");
				return false;
			}

			/*increase the argument index*/
			argind++;
			done = false;
			while (argind < argc
					&& (argv[argind][0] != '-'
							|| (argv[argind][0] == '-'
									&& argv[argind][1] == '\0'))) {
				//test file
				if (strcmp(argv[argind], "-") && !Utils::exists(argv[argind])) {
					Utils::log("The file %s does not exist\n", argv[argind]);
					return false;
				}
				//get the file name
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_FASTX));

				//increase the argument index
				argind++;
				done = true;
			}
			if (done == false) {
				Utils::log("At least one input file must be specified\n");
				return false;
			}
		} else if (!strcmp(argv[argind], "-b")) {
			/*specify the inputs are single-end*/
			if (first) {
				first = false;
				setPaired(false);
			} else if (isPaired()) {
				Utils::log(
						"Cannot specify single-end and paired-end reads together\n");
				return false;
			}

			/*increase the argument index*/
			argind++;
			done = false;
			while (argind < argc
					&& (argv[argind][0] != '-'
							|| (argv[argind][0] == '-'
									&& argv[argind][1] == '\0'))) {
				//test file
				if (strcmp(argv[argind], "-") && !Utils::exists(argv[argind])) {
					Utils::log("The file %s does not exist\n", argv[argind]);
					return false;
				}
				//get the file name
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_BAM));

				//increase the argument index
				argind++;
				done = true;
			}
			if (done == false) {
				Utils::log("At least one input file must be specified\n");
				return false;
			}
		} else if (!strcmp(argv[argind], "-s")) {
			/*specify the inputs are single-end*/
			if (first) {
				first = false;
				setPaired(false);
			} else if (isPaired()) {
				Utils::log(
						"Cannot specify single-end and paired-end reads together\n");
				return false;
			}

			/*increase the argument index*/
			argind++;
			done = false;
			while (argind < argc
					&& (argv[argind][0] != '-'
							|| (argv[argind][0] == '-'
									&& argv[argind][1] == '\0'))) {
				//test file
				if (strcmp(argv[argind], "-") && !Utils::exists(argv[argind])) {
					Utils::log("The file %s does not exist\n", argv[argind]);
					return false;
				}
				//get the file name
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_SAM));

				//increase the argument index
				argind++;
				done = true;
			}
			if (done == false) {
				Utils::log("At least one input file must be specified\n");
				return false;
			}

			//input files are paired in FASTA format
		} else if (!strcmp(argv[argind], "-q")) {
			if (first) {
				first = false;
				setPaired(1);
			} else if (!isPaired()) {
				Utils::log(
						"Cannot specify single-end and paired-end reads together\n");
				return false;
			}

			++argind;
			done = false;
			if (argind + 1 < argc && argv[argind][0] != '-'
					&& (argv[argind + 1][0] != '-'
							|| (argv[argind + 1][0] == '-'
									&& argv[argind + 1][1] == '\0'))) {
				//save the two files
				//test file
				if (strcmp(argv[argind], "-") && !Utils::exists(argv[argind])) {
					Utils::log("The file %s does not exist\n", argv[argind]);
					return false;
				}
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_FASTX));
				argind++;

				//test file
				if (strcmp(argv[argind], "-") && !Utils::exists(argv[argind])) {
					Utils::log("The file %s does not exist\n", argv[argind]);
					return false;
				}
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_FASTX));
				argind++;

				done = true;
			}
			if (!done) {
				Utils::log(
						"Two paired input files should be specified for -fq\n");
				return false;
			}
		} else if (!strcmp(argv[argind], "-fifo")) {
			/*FIFO file for inter-process communcations. Users cannot directly use this parameter*/
			/*increase the argument index*/
			++argind;
			if(argind < argc){
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_FIFO));
			}else{
				Utils::log("At least one FIFO input file must be specified\n");
				return false;
			}
			++argind;
			_viaFifo = true;
		} else if (!strcmp(argv[argind], "-fifope")) {
			/*FIFO file for inter-process communcations. Users cannot directly use this parameter*/
			/*increase the argument index*/
			++argind;
			if(argind < argc){
				_readsFileNames.push_back(
						make_pair(string(argv[argind]), FILE_TYPE_FIFO));
			}else{
				Utils::log("At least one FIFO input file must be specified\n");
				return false;
			}
			++argind;
			_paired = true;
			_viaFifo = true;
		} else if (!strcmp(argv[argind], "-o")) {
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_samFileName = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-multi")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 1) {
					intVal = 1;
				}
				if (intVal > MAX_MULTI_ALIGN) {
					intVal = MAX_MULTI_ALIGN;
				}
				_maxMultiAligns = intVal;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-unique")) { /*unique alignments*/
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				_onlyUnique = intVal > 0 ? true : false;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

			/*read group information*/
		} else if (!strcmp(argv[argind], "-rgid")) { /*read group identifier*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgID = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgsm")) { /*read group sample name*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgSM = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rglb")) { /*read group library*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgLB = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgpl")) { /*read group platform*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgPL = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgpu")) { /*read group platform unit*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgPU = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgcn")) { /*name of sequencing center producing the read*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgCN = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgds")) { /*description about the reads*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgDS = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgdt")) { /*date the run was produced*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgDT = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-rgpi")) { /*predicated median insert size*/
			argind++;
			if (argind < argc) {
				if (strlen(argv[argind]) > 0) {
					_rgPI = argv[argind];
					argind++;
				}
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

		} else if (!strcmp(argv[argind], "-min_id")) { /*minimal identity*/
			floatVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%lf", &floatVal);
				if (floatVal < 0) {
					floatVal = 0;
				}
				if (floatVal > 1) {
					floatVal = 1;
				}
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
			_minIdentity = floatVal;
		} else if (!strcmp(argv[argind], "-sensitive")) {
			sensitivity = true;
			argind++;
		} else if (!strcmp(argv[argind], "-min_ratio")) {
			floatVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%lf", &floatVal);
				if (floatVal < 0) {
					floatVal = 0;
				}
				if (floatVal > 1) {
					floatVal = 1;
				}
				argind++;
				_minRatio = floatVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-min_score")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 1) {
					intVal = 1;
				}
				_minAlginScore = intVal;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-miss_prob")) {
			floatVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%lf", &floatVal);
				if (floatVal < 0)
					floatVal = 0;
				if (floatVal > 1)
					floatVal = 1;
				argind++;
				_missProb = floatVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

		} else if (!strcmp(argv[argind], "-min_seed")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < GLOBAL_MIN_SEED_SIZE) {
					intVal = GLOBAL_MIN_SEED_SIZE;
				} else if (intVal > GLOBAL_MAX_SEED_SIZE) {
					intVal = GLOBAL_MAX_SEED_SIZE;
				}
				_lowerMinSeedSize = intVal;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-max_seed")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < GLOBAL_MIN_SEED_SIZE) {
					intVal = GLOBAL_MIN_SEED_SIZE;
				} else if (intVal > GLOBAL_MAX_SEED_SIZE) {
					intVal = GLOBAL_MAX_SEED_SIZE;
				}
				_upperMinSeedSize = intVal;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-max_occ")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 1) {
					intVal = 1;
				}
				_maxSeedOcc = intVal;
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		}

		else if (!strcmp(argv[argind], "-match")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 1)
					intVal = 1;

				argind++;
				_match = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-mismatch")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_mismatch = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-gopen")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_gapOpen = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-gext")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_gapExtend = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-avg_ins")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_insertSize = intVal;
				newInsertSize = true;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-ins_std")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_stdInsertsize = intVal;
				newStdInsertSize = true;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-ins_npairs")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 1)
					intVal = 1;

				argind++;
				_topReadsEstIns = intVal * INS_SIZE_EST_MULTIPLE;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-ins_mapq")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				if (intVal > 255) {
					intVal = 255;
				}

				argind++;
				_mapQualReliable = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

		} else if (!strcmp(argv[argind], "-no_rescue")) {
			++argind;
			_rescueMate = false;
		}
#ifdef HAVE_TWICE_RESCUE
		else if(!strcmp(argv[argind], "-no_rescue_twice")) {
			++argind;
			_rescueTwice = false;
		}
#endif
		else if (!strcmp(argv[argind], "-t")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
				if (intVal < 0)
					intVal = 0;

				argind++;
				_numCPUs = intVal;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}
		} else if (!strcmp(argv[argind], "-g")) {
			intVal = 1;
			argind++;
			if (argind < argc) {
				sscanf(argv[argind], "%d", &intVal);
#ifdef HAVE_WRAPPER
				if(intVal < 1){
					intVal = 1;
				}
				if (intVal > gpuInfo->getNumGPUs()) {
					intVal = gpuInfo->getNumGPUs();
				}
				_numGPUs = intVal;
				_gpuIndex = 0;
#else
				if (intVal >= gpuInfo->getNumGPUs()) {
					intVal = gpuInfo->getNumGPUs() - 1;
				}

				if (intVal < 0) {
					_numGPUs = 0; /*disable the use of GPUs*/
					_gpuIndex = -1;
				} else {
					_gpuIndex = intVal;
				}
#endif
				argind++;
			} else {
				Utils::log("not specify value for the parameter %s\n",
						argv[argind - 1]);
				return false;
			}

		} else if (!strcmp(argv[argind], "-h")) {
			return false;
		} else {
			Utils::log("Unknown parameter: %s\n", argv[argind]);
			return false;
		}
	}
	/*calculate the total number of threads*/
	_numThreads = _numCPUs + _numGPUs;
	Utils::log("Number of extra CPU threads: %d\n", _numCPUs);
	Utils::log("GPU index: %d\n", _gpuIndex);

	/*check if FIFO is used*/
	if(_viaFifo){
		if(_readsFileNames.size() != 1){
			Utils::exit("FIFO parameters were not correctly specified\n");
		}
	}

#ifdef HAVE_WRAPPER
	/*check the output file*/
	if(_samFileName.length() == 0){
		Utils::exit("When using the wrapper, the ouput file base name must be specified\n");
	}
#endif

	/*check the genome*/
	if (_bwtFileBase.size() == 0) {
		Utils::log(
				"The reference genome must be specified using option \"-r\"");
		return false;
	}

	/*check the sensitivity*/
	if (sensitivity) {
		Utils::log(
				"Will only use min_score, since we are concerned more about sensitivity\n");
		_minRatio = 0;
		_minIdentity = 0;
	}

	/*re-check the insert size*/
	if (newInsertSize && !newStdInsertSize) {
		/*if the standard deviation is not specified, just use 10% of the insert size*/
		_stdInsertsize = (int) (_insertSize * 0.1);
	}

	/*check if insert size is etimated automatically*/
	if (!newInsertSize && !newStdInsertSize) {
		_estInsertSize = true;
	} else {
		_estInsertSize = false;
	}

	/*check the seed size*/
	if (_upperMinSeedSize < _lowerMinSeedSize) {
		Utils::exit(
				"The upper bound of the minimal seed size (%d) is not allowed to be less than its lower bound (%d)\n",
				_upperMinSeedSize, _lowerMinSeedSize);
	}
	/*check the read group header information*/
	if (_rgID.size() > 0) {
		if (_rgSM.size() == 0) {
			Utils::exit(
					"Must specify read group sample name if read group identifier is specified\n");
		}
	}

	/*update the minimal local alignment score*/
	_minAlginScore *= _match;

	/*initialize the table*/
	int seedSize, numErrors;
	int lastSeedSize = 0;
	for (uint32_t length = 1; length <= MAX_SEQ_LENGTH; ++length) {
		numErrors = estimateNumErrors(length, _missProb);
		_numErrorTable[length] = numErrors;

		/*estimate the minimal seed size according to dove hole principle*/
		seedSize = length / (numErrors + 1);
		if (seedSize < _lowerMinSeedSize) {
			seedSize = _lowerMinSeedSize;
		}
		if (seedSize > _upperMinSeedSize) {
			seedSize = _upperMinSeedSize;
		}
		if (lastSeedSize != seedSize) {
			lastSeedSize = seedSize;
			//if(length % 10 == 0) Utils::log("%d-bp: %d\n", length, seedSize);
		}
		_minSeedSizeTable[length] = seedSize;
	}
	return true;
}

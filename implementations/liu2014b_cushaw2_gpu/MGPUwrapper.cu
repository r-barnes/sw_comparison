/*
 * MGPUwrapper.cu
 *
 *  Created on: Jan 28, 2013
 *      Author: yongchao
 */
#include "Options.h"
#include "SeqFileParser.h"
#include "Sequence.h"
#include "Utils.h"

struct WrapperThreadParams
{
	WrapperThreadParams(Options* options, int fifoFile, int tid) {
		_options = options;
		_tid = tid;
		_fifoFile = fifoFile;
	}
	inline void setSeqFileParser(SeqFileParser* parser1,
			SeqFileParser* parser2 = NULL) {
		_parser1 = parser1;
		_parser2 = parser2;
	}
	int _tid;
	int _fifoFile;
	SeqFileParser* _parser1;
	SeqFileParser* _parser2;
	Options* _options;
};

static string getAbsolutePath(const char* file, char* path, int mode) {
	char* tok;

	/*parse the PATH enviroment*/
	tok = strtok((char*) path, ":");
	while (tok) {
		/*form a new absolute path*/
		string filePath = tok;
		filePath += "/";
		filePath += file;

		/*test the file attribute*/
		if (access(filePath.c_str(), mode) == 0) {
			return filePath;
		}
		/*get the next*/
		tok = strtok(NULL, ":");
	}
	return string("");
}

static string getBinaryPath(char* binary) {

	string null = "";
	/*get the PATH enviroment variable*/
	string path = getenv("PATH");
	string local = "./";
	local += binary;

	/*check the existence of the binary*/
	if (access(local.c_str(), X_OK) != 0) {
		string newPath = getAbsolutePath(binary, (char*) path.c_str(), X_OK);
		if (newPath.length() > 0) {
			return newPath;
		} else {
			Utils::exit("Failed to find the program: %s in your system\n",
					binary);
			return null;
		}
	}
	return local;
}

static string createArgList(Options* options, vector<char*>& arglist,
		char* binary, int argc, char* argv[], int pid) {

	char buffer[1024];
	string fifoFile;
	/*save the binary space*/
	arglist.push_back(strdup(binary));

	/*save the command lines*/
	int index = 1;
	while (index < argc) {
		/*filter out parameters*/
		if (!strcmp(argv[index], "-f") || !strcmp(argv[index], "-q")
				|| !strcmp(argv[index], "-s") || !strcmp(argv[index], "-b")
				|| !strcmp(argv[index], "-fifo")
				|| !strcmp(argv[index], "-fifope")
				|| !strcmp(argv[index], "-o")) {
			++index;

			while (index < argc) {
				if (argv[index][0] == '-') {
					break;
				}
				++index;
			}
			continue;
		}
		/*save the command line*/
		arglist.push_back(argv[index]);
		++index;
	}

	/*add new FIFO parameters*/
	int numGPUs = options->getNumGPUs();
	if (options->isPaired()) {
		arglist.push_back(strdup("-fifope"));
	} else {
		arglist.push_back(strdup("-fifo"));
	}
	sprintf(buffer, "/tmp/cushaw2_gpu_%d", pid);
	unlink(buffer);

	arglist.push_back(strdup(buffer));
	fifoFile = buffer;

	/*GPU index*/
	arglist.push_back(strdup("-g"));
	sprintf(buffer, "%d", pid);
	arglist.push_back(strdup(buffer));

	/*number of CPUs*/
	int numCPUs = options->getNumCPUs();
	if (numCPUs > 0) {
		int avgNumCPUs = (numCPUs + numGPUs - 1) / numGPUs;
		arglist.push_back(strdup("-t"));
		sprintf(buffer, "%d", min(avgNumCPUs, numCPUs - avgNumCPUs * pid));
		arglist.push_back(strdup(buffer));
	}

	/*output file*/
	sprintf(buffer, "%s.%d", options->getSamFileName().c_str(), pid);
	arglist.push_back(strdup("-o"));
	arglist.push_back(strdup(buffer));

	/*a NULL terminator for the list*/
	arglist.push_back(NULL);

	return fifoFile;
}
static void* singleEnd(void* args) {
	WrapperThreadParams* params = (WrapperThreadParams*) args;
	Options* options = params->_options;
	SeqFileParser* parser = params->_parser1;
	int fifoFile = params->_fifoFile;
	int numGPUs = options->getNumGPUs();

	/*allocate a buffer*/
	int32_t nseqs;
	size_t bufferSize = 1024, bufferLength = 0;
	uint8_t* buffer = new uint8_t[bufferSize];
	if (!buffer) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	/*allocate space for read batches*/
	int32_t numReadsPerBatch = 256;
	Sequence* sequences = new Sequence[numReadsPerBatch];
	if (!sequences) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}

	/*read sequences from the current file*/
	do {
		/*read a batch of files*/
		if ((nseqs = parser->getSeq(sequences, numReadsPerBatch)) == 0) {
			break;
		}

		/*write the sequences to the fifo*/
		for (int i = 0; i < nseqs; ++i) {
			bufferLength = sequences[i].compose(buffer, bufferSize);
			if (write(fifoFile, buffer, bufferLength) != bufferLength) {
				Utils::exit("FIFO write failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
		}

	} while (1);

	delete[] sequences;
	delete[] buffer;

	return NULL;
}

static void* pairedEnd(void* args) {
	WrapperThreadParams* params = (WrapperThreadParams*) args;
	Options* options = params->_options;
	SeqFileParser *parser = params->_parser1;
	SeqFileParser *parser2 = params->_parser2;
	int fifoFile = params->_fifoFile;
	int numGPUs = options->getNumGPUs();

	/*allocate a buffer*/
	int32_t nseqs;
	size_t bufferSize = 1024, bufferLength = 0;
	uint8_t* buffer = new uint8_t[bufferSize];
	if (!buffer) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	/*allocate space for read batches*/
	int32_t numReadsPerBatch = 256;
	Sequence* sequences = new Sequence[numReadsPerBatch];
	if (!sequences) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	Sequence* sequences2 = new Sequence[numReadsPerBatch];
	if (!sequences2) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}

	/*perform paired-end alignments*/
	do {
		/*read a sequence pair*/
		options->lock();
		if ((nseqs = parser->getSeqLockFree(sequences, numReadsPerBatch))
				== 0) {
			options->unlock();
			break;
		}
		if (parser2->getSeqLockFree(sequences2, numReadsPerBatch) != nseqs) {
			Utils::log("The two files have different number of sequences\n");
			options->unlock();
			break;
		}
		options->unlock();

		/*write the paired-end sequences to the fifo*/
		for (int i = 0; i < nseqs; ++i) {
			/*left read*/
			bufferLength = sequences[i].compose(buffer, bufferSize);
			if (write(fifoFile, buffer, bufferLength) != bufferLength) {
				Utils::exit("FIFO write failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
			/*right read*/
			bufferLength = sequences2[i].compose(buffer, bufferSize);
			if (write(fifoFile, buffer, bufferLength) != bufferLength) {
				Utils::exit("FIFO write failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
		}
	} while (1);

	delete[] sequences2;
	delete[] sequences;
	delete[] buffer;

	return NULL;
}

static void* estimateInsertSize(void* args) {
	SeqFileParser *parser, *parser2;
	WrapperThreadParams* params = (WrapperThreadParams*) args;
	Options* options = params->_options;
	int fifoFile = params->_fifoFile;
	int numGPUs = options->getNumGPUs();

	/*allocate a buffer*/
	size_t bufferSize = 1024, bufferLength = 0;
	uint8_t* buffer = new uint8_t[bufferSize];
	if (!buffer) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	/*allocate space for read batches*/
	int32_t numReadsPerBatch = 256;
	Sequence* sequences = new Sequence[numReadsPerBatch];
	if (!sequences) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}
	Sequence* sequences2 = new Sequence[numReadsPerBatch];
	if (!sequences2) {
		Utils::exit("Memory allocation failed at line %d in function %s\n",
				__LINE__, __FUNCTION__);
	}

	/*get the input file list*/
	vector<pair<string, int> > &inputs = options->getInputFileList();

	/*get the number of reads*/
	int32_t numTopReads = options->getTopReadsEstIns() / INS_SIZE_EST_MULTIPLE;
	numTopReads *= INS_SIZE_EST_MULTIPLE;

	Sequence seq1, seq2;
	for (size_t file = 0; file < inputs.size(); file += 2) {
		/*open the file for the left sequences*/
		parser = new SeqFileParser(inputs[file].first.c_str(), false,
				inputs[file].second);
		/*open the file for the right sequences*/
		parser2 = new SeqFileParser(inputs[file + 1].first.c_str(), false,
				inputs[file].second);

		while (numTopReads > 0) {

			/*read a sequence pair*/
			options->lock();
			if (parser->getSeq(seq1) == 0) {
				options->unlock();
				break;
			}
			if (parser2->getSeq(seq2) == 0) {
				Utils::log(
						"The two files have different number of sequences\n");
				options->unlock();
				break;
			}
			options->unlock();

			/*write the sequences to the fifo*/
			bufferLength = seq1.compose(buffer, bufferSize);
			if (write(fifoFile, buffer, bufferLength) != bufferLength) {
				Utils::exit("FIFO write failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
			/*right read*/
			bufferLength = seq2.compose(buffer, bufferSize);
			if (write(fifoFile, buffer, bufferLength) != bufferLength) {
				Utils::exit("FIFO write failed at line %d in file %s\n",
						__LINE__, __FILE__);
			}
			--numTopReads;
		}

		/*release the file parser*/
		delete parser;
		delete parser2;

		/*check if it is done*/
		if (numTopReads <= 0) {
			break;
		}
	}
	if (numTopReads > 0) {
		/*write the end-of-data mark*/
		buffer[0] = buffer[1] = buffer[2] = buffer[3] = 0;
		if (write(fifoFile, buffer, 4) != 4) {
			Utils::exit("FIFO write failed at line %d in file %s\n", __LINE__,
					__FILE__);
		}
	}

	delete[] sequences;
	delete[] sequences2;
	delete[] buffer;

	return NULL;
}
int main(int argc, char* argv[]) {

	double stime, etime;

	/*get the startup time*/
	stime = Utils::getSysTime();

	Options* options = new Options();

	/*parse the arguments*/
	if (!options->parse(argc, argv)) {
		options->printUsage();
		return 0;
	}
	int numGPUs = options->getNumGPUs();

	/*get the binary path*/
	string binary = getBinaryPath("cushaw2-gpu");

	/*create argument lists*/
	vector<vector<char*> > arglists;
	vector<int> fifoFiles;
	vector<string> fifoFileNames;
	arglists.resize(numGPUs);
	fifoFiles.resize(numGPUs);
	fifoFileNames.resize(numGPUs);

	for (size_t pid = 0; pid < arglists.size(); ++pid) {
		fifoFileNames[pid] = createArgList(options, arglists[pid],
				"cushaw2-gpu", argc, argv, pid);

		/*create FIFOs*/
		string& file = fifoFileNames[pid];
		if (mkfifo(file.c_str(), 0777) != 0) {
			Utils::exit("Failed to create the FIFO %s at line %d in file %s\n",
					file.c_str(), __LINE__, __FILE__);
		}
	}

	/*for child processes*/
	vector<int> children(arglists.size(), 0);
	for (size_t index = 0; index < arglists.size(); ++index) {

		/*create a child process*/
		int pid = fork();
		if (pid == 0) {
			/*child process*/
			if (execvp(binary.c_str(), arglists[index].data()) == -1) {
				Utils::exit("Failed to launch the binary %s\n", binary.c_str());
			}
			return 0; /*NEVER reachs here on success*/
		} else if (pid > 0) {
			/*parent process*/

			/*open the fifos*/
			string& fifoFileName = fifoFileNames[index];
			fifoFiles[index] = open(fifoFileName.c_str(), O_WRONLY);
			if (fifoFiles[index] == -1) {
				Utils::exit(
						"Failed to open FIFO %s in write at line %d in file %s\n",
						fifoFileName.c_str(), __LINE__, __FILE__);
			}

			/*save the process ID*/
			children[index] = pid;
			Utils::log("Child process (%d) has been created\n", pid);
		} else {
			Utils::exit(
					"Failed to create a child process at line %d in file %s\n",
					__LINE__, __FILE__);
		}
	}

	/*create threads to feed data on all child processes*/
	vector<pthread_t> threadIDs(arglists.size(), 0);
	vector<WrapperThreadParams*> params(arglists.size(), NULL);

	/*create parameters for threads*/
	for (size_t tid = 0; tid < params.size(); ++tid) {
		params[tid] = new WrapperThreadParams(options, fifoFiles[tid], tid);
	}

	/*get the input file list*/
	vector<pair<string, int> > &inputs = options->getInputFileList();
	uint8_t fileEndMarker[4] = {0, 0, 0, 0};
	if (options->isPaired()) {

		/*estimate insert size*/
		if (options->estimateInsertSize()) {
			/*create threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
				if (pthread_create(&threadIDs[tid], NULL, estimateInsertSize,
						params[tid]) != 0) {
					Utils::exit(
							"Thread creating failed at line %d in file %s\n",
							__LINE__, __FILE__);
				}
			}
			/*wait for the completion of all threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
				pthread_join(threadIDs[tid], NULL);
			}
		}

		/*perform paired-end alignments*/
		for (size_t file = 0; file < inputs.size(); file += 2) {
			SeqFileParser* parser = new SeqFileParser(
					inputs[file].first.c_str(), false, inputs[file].second);
			SeqFileParser* parser2 = new SeqFileParser(
					inputs[file + 1].first.c_str(), false, inputs[file].second);

			/*create threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
				/*set file parser*/
				params[tid]->setSeqFileParser(parser, parser2);

				if (pthread_create(&threadIDs[tid], NULL, pairedEnd,
						params[tid]) != 0) {
					Utils::exit(
							"Thread creating failed at line %d in file %s\n",
							__LINE__, __FILE__);
				}
			}

			/*wait for the completion of all threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
				pthread_join(threadIDs[tid], NULL);
			}

			/*delete file parser*/
			delete parser;
			delete parser2;
		}

		/*wrie the end-of-data mark*/
		for(size_t tid = 0; tid < fifoFiles.size(); ++tid){
			if (write(fifoFiles[tid], fileEndMarker, 4) != 4) {
				Utils::exit("FIFO write failed at line %d in file %s\n", __LINE__,
						__FILE__);
			}
		}
	} else {

		/*perform single-end alignments*/
		for (size_t file = 0; file < inputs.size(); ++file) {
			SeqFileParser* parser = new SeqFileParser(
					inputs[file].first.c_str(), numGPUs > 1,
					inputs[file].second);

			/*create threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {

				/*set file parser*/
				params[tid]->setSeqFileParser(parser);

				/*create threads*/
				if (pthread_create(&threadIDs[tid], NULL, singleEnd,
						params[tid]) != 0) {
					Utils::exit(
							"Thread creating failed at line %d in file %s\n",
							__LINE__, __FILE__);
				}
			}
			/*wait for the completion of all threads*/
			for (size_t tid = 0; tid < threadIDs.size(); ++tid) {
				pthread_join(threadIDs[tid], NULL);
			}

			/*release file parser*/
			delete parser;
		}

		/*wrie the end-of-data mark*/
		for(size_t tid = 0; tid < fifoFiles.size(); ++tid){
			if (write(fifoFiles[tid], fileEndMarker, 4) != 4) {
				Utils::exit("FIFO write failed at line %d in file %s\n", __LINE__,
						__FILE__);
			}
		}
	}

	/*wait for the completion of all child processes*/
	Utils::log("Waiting for the completion of child processes\n");
	for (size_t pid = 0; pid < children.size(); ++pid) {
		waitpid(children[pid], 0, 0);
	}

	/*remove the fifo files*/
	for (size_t i = 0; i < fifoFiles.size(); ++i) {
		/*close the fifo*/
		close(fifoFiles[i]);

		/*delete the files*/
		unlink(fifoFileNames[i].c_str());
	}

	etime = Utils::getSysTime();
	Utils::log("Overall time: %.2f seconds\n", etime - stime);

	/*leave the memory to be released by the OS*/

	return 0;
}


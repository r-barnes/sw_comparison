/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#include "CParams.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define CUDASW_VERSION	"2.0.11"

CParams::CParams()
{
	gapOpen = DEFAULT_GAPO;
	gapExtend = DEFAULT_GAPE;
	scoreThreshold = DEFAULT_MIN_SCORE;
	topScoresNum = DEFAULT_TOPSCORE_NUM;
	useSingleGPU = false;	//default false;
	singleGPUID = 0;		//default GPU ID;
	useSIMTModel = true;

	subMatName[0] = '\0';
	queryFile[0] = '\0';
	dbFile[0] = '\0';
}
CParams::~CParams()
{
}
void CParams::getSysTime(double *dtime)
{
  struct timeval tv;

  gettimeofday (&tv, NULL);

  *dtime = (double) tv.tv_sec;
  *dtime = *dtime + (double) (tv.tv_usec) / 1000000.0;
}
void CParams::printUsage()
{
    fprintf(stderr,"Usage:\n");
    fprintf(stderr,"./cudasw [options]\n");
    fprintf(stderr,"Standard options:\n");
    fprintf(stderr,"\t-mod <string>\t: specify the programming model used (default simt)\n");
    fprintf(stderr,"\t\tsupported model names: simt and simd \n");
    fprintf(stderr,"\t-mat <string>\t: specify the substitution matrix name (default blosum62)\n");
    fprintf(stderr,"\t\tsupported matrix names: blosum45, blosum50, blosum62 and blosum80\n");
    fprintf(stderr,"\t-query <string>\t: specify the query sequence file\n");
    fprintf(stderr,"\t-db <string>\t: specify the database sequence file\n");
    fprintf(stderr,"\t-gapo <integer>\t: specify the gap open panelty (0 ~ 255), (default %d)\n", DEFAULT_GAPO);
    fprintf(stderr,"\t-gape <integer>\t: specify the gap extension panelty (0 ~ 255), (default %d)\n", DEFAULT_GAPE);
    fprintf(stderr,"\t-min_score <integer>\t: specify the minimum score reported(default %d)\n", DEFAULT_MIN_SCORE);
    fprintf(stderr,"\t-topscore_num <integer>\t: specify the number of top scores reported(default %d)\n", DEFAULT_TOPSCORE_NUM);
    fprintf(stderr,"\t-use_single <integer>\t: force to use the single GPU with ID #integer\n");
    fprintf(stderr,"\t-version\t: print out the version\n");
	
}
int CParams::parseParams(int argc, char* argv[])
{
	bool queryAvail = false;
	bool dbAvail = false;

	int index;
	//display usages
	if (argc < 2 || !strcmp(argv[1], "-help") || !strcmp(argv[1], "-?")){
		printUsage();
		return false;
	}
	index = 1;
	// parse other arguments
	char* arg;
	for ( ; index < argc; index++) {
		arg = argv[index];	
		if (index >= argc){
			fprintf(stderr, "The number of the specified argments does not match!");
			printUsage();
			return 0;
		}
		if(strcmp(arg, "-mod") == 0){
			char buffer[256];
			sscanf(argv[++index], "%s", buffer);
			if(!strcmp(buffer, "simt") || !strcmp(buffer, "SIMT")){
				useSIMTModel = true;
			}else if(!strcmp(buffer, "simd") || !strcmp(buffer, "SIMD")){
				useSIMTModel = false;
			}else{
				fprintf(stderr, "using the default model SIMT\n");
			}
		}else if (strcmp(arg, "-mat") == 0){
			sscanf(argv[++index], "%s",	subMatName);
		}else if (strcmp(arg, "-query") == 0){
			sscanf(argv[++index], "%s", queryFile);
			queryAvail = true;
		}else if (strcmp(arg, "-db") == 0){
			sscanf(argv[++index], "%s", dbFile);
			dbAvail = true;
		}else if (strcmp(arg, "-gapo") == 0){
			sscanf(argv[++index], "%d", &gapOpen);
			if(gapOpen < 0 || gapOpen > 255){
				gapOpen = DEFAULT_GAPO;
				fprintf(stderr, "using the default gap open penalty: %d\n", gapOpen);
			}
		}else if (strcmp(arg, "-gape") == 0){
			sscanf(argv[++index], "%d", &gapExtend);
			if(gapExtend < 0 || gapExtend > 255){
				gapExtend = DEFAULT_GAPE;
				fprintf(stderr, "using the default gap extension penalty: %d\n", gapExtend);
			}
		}else if (strcmp(arg, "-min_score") == 0){
			sscanf(argv[++index], "%d", &scoreThreshold);
			if(scoreThreshold < 0){
				scoreThreshold = 0;
			}
		}else if (strcmp(arg, "-topscore_num") == 0){
			sscanf(argv[++index], "%d", &topScoresNum);
			if(topScoresNum < 1){
				topScoresNum = 1;
			}
		}else if (strcmp(arg, "-use_single") == 0){
			useSingleGPU = true;
			sscanf(argv[++index], "%d", &singleGPUID);
			if(singleGPUID < 0){
				singleGPUID = 0;
			}	
		}else if (strcmp(arg, "-help") == 0 || strcmp(arg, "-?") == 0) {
            printUsage();
            return 0;
		}else if(strcmp(arg, "-version") == 0){
			fprintf(stderr, "CUDASW++ version: %s\n", CUDASW_VERSION);
			return 0;
        }else {
            fprintf(stderr,"\n************************************\n");
            fprintf(stderr,"Unknown option: %s;\n", arg);
            fprintf(stderr,"\n************************************\n");
            printUsage();
            return 0;
        }
	}
	if(queryAvail == false){
		fprintf(stderr, "Please specifiy the query sequnece file\n");
		printUsage();
		return 0;
	}
	if(dbAvail == false){
		fprintf(stderr, "Please specifiy the database sequnece file\n");
		printUsage();
		return 0;
	}
	return 1;
}

//blosum45
const char CParams::blosum45[32][32] = {
{5,-1,-1,-2,-1,-2,0,-2,-1,-1,-1,-1,-1,-1,-1,-2,1,0,0,-2,0,-2,-1,0,0,0,0,0,0,0,0,0,},
{-1,4,-2,5,1,-3,-1,0,-3,0,-3,-2,4,-2,0,-1,0,0,-3,-4,-1,-2,2,0,0,0,0,0,0,0,0,0,},
{-1,-2,12,-3,-3,-2,-3,-3,-3,-3,-2,-2,-2,-4,-3,-3,-1,-1,-1,-5,-2,-3,-3,0,0,0,0,0,0,0,0,0,},
{-2,5,-3,7,2,-4,-1,0,-4,0,-3,-3,2,-1,0,-1,0,-1,-3,-4,-1,-2,1,0,0,0,0,0,0,0,0,0,},
{-1,1,-3,2,6,-3,-2,0,-3,1,-2,-2,0,0,2,0,0,-1,-3,-3,-1,-2,4,0,0,0,0,0,0,0,0,0,},
{-2,-3,-2,-4,-3,8,-3,-2,0,-3,1,0,-2,-3,-4,-2,-2,-1,0,1,-1,3,-3,0,0,0,0,0,0,0,0,0,},
{0,-1,-3,-1,-2,-3,7,-2,-4,-2,-3,-2,0,-2,-2,-2,0,-2,-3,-2,-1,-3,-2,0,0,0,0,0,0,0,0,0,},
{-2,0,-3,0,0,-2,-2,10,-3,-1,-2,0,1,-2,1,0,-1,-2,-3,-3,-1,2,0,0,0,0,0,0,0,0,0,0,},
{-1,-3,-3,-4,-3,0,-4,-3,5,-3,2,2,-2,-2,-2,-3,-2,-1,3,-2,-1,0,-3,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,0,1,-3,-2,-1,-3,5,-3,-1,0,-1,1,3,-1,-1,-2,-2,-1,-1,1,0,0,0,0,0,0,0,0,0,},
{-1,-3,-2,-3,-2,1,-3,-2,2,-3,5,2,-3,-3,-2,-2,-3,-1,1,-2,-1,0,-2,0,0,0,0,0,0,0,0,0,},
{-1,-2,-2,-3,-2,0,-2,0,2,-1,2,6,-2,-2,0,-1,-2,-1,1,-2,-1,0,-1,0,0,0,0,0,0,0,0,0,},
{-1,4,-2,2,0,-2,0,1,-2,0,-3,-2,6,-2,0,0,1,0,-3,-4,-1,-2,0,0,0,0,0,0,0,0,0,0,},
{-1,-2,-4,-1,0,-3,-2,-2,-2,-1,-3,-2,-2,9,-1,-2,-1,-1,-3,-3,-1,-3,-1,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,0,2,-4,-2,1,-2,1,-2,0,0,-1,6,1,0,-1,-3,-2,-1,-1,4,0,0,0,0,0,0,0,0,0,},
{-2,-1,-3,-1,0,-2,-2,0,-3,3,-2,-1,0,-2,1,7,-1,-1,-2,-2,-1,-1,0,0,0,0,0,0,0,0,0,0,},
{1,0,-1,0,0,-2,0,-1,-2,-1,-3,-2,1,-1,0,-1,4,2,-1,-4,0,-2,0,0,0,0,0,0,0,0,0,0,},
{0,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,2,5,0,-3,0,-1,-1,0,0,0,0,0,0,0,0,0,},
{0,-3,-1,-3,-3,0,-3,-3,3,-2,1,1,-3,-3,-3,-2,-1,0,5,-3,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-2,-4,-5,-4,-3,1,-2,-3,-2,-2,-2,-2,-4,-3,-2,-2,-4,-3,-3,15,-2,3,-2,0,0,0,0,0,0,0,0,0,},
{0,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,-1,-2,-1,-1,-1,0,0,0,0,0,0,0,0,0,},
{-2,-2,-3,-2,-2,3,-3,2,0,-1,0,0,-2,-3,-1,-1,-2,-1,-1,3,-1,8,-2,0,0,0,0,0,0,0,0,0,},
{-1,2,-3,1,4,-3,-2,0,-3,1,-2,-1,0,-1,4,0,0,-1,-3,-2,-1,-2,4,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}
};
//blosum50
const char CParams::blosum50[32][32] = {
{5,-2,-1,-2,-1,-3,0,-2,-1,-1,-2,-1,-1,-1,-1,-2,1,0,0,-3,-1,-2,-1,0,0,0,0,0,0,0,0,0,},
{-2,5,-3,5,1,-4,-1,0,-4,0,-4,-3,4,-2,0,-1,0,0,-4,-5,-1,-3,2,0,0,0,0,0,0,0,0,0,},
{-1,-3,13,-4,-3,-2,-3,-3,-2,-3,-2,-2,-2,-4,-3,-4,-1,-1,-1,-5,-2,-3,-3,0,0,0,0,0,0,0,0,0,},
{-2,5,-4,8,2,-5,-1,-1,-4,-1,-4,-4,2,-1,0,-2,0,-1,-4,-5,-1,-3,1,0,0,0,0,0,0,0,0,0,},
{-1,1,-3,2,6,-3,-3,0,-4,1,-3,-2,0,-1,2,0,-1,-1,-3,-3,-1,-2,5,0,0,0,0,0,0,0,0,0,},
{-3,-4,-2,-5,-3,8,-4,-1,0,-4,1,0,-4,-4,-4,-3,-3,-2,-1,1,-2,4,-4,0,0,0,0,0,0,0,0,0,},
{0,-1,-3,-1,-3,-4,8,-2,-4,-2,-4,-3,0,-2,-2,-3,0,-2,-4,-3,-2,-3,-2,0,0,0,0,0,0,0,0,0,},
{-2,0,-3,-1,0,-1,-2,10,-4,0,-3,-1,1,-2,1,0,-1,-2,-4,-3,-1,2,0,0,0,0,0,0,0,0,0,0,},
{-1,-4,-2,-4,-4,0,-4,-4,5,-3,2,2,-3,-3,-3,-4,-3,-1,4,-3,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,-1,1,-4,-2,0,-3,6,-3,-2,0,-1,2,3,0,-1,-3,-3,-1,-2,1,0,0,0,0,0,0,0,0,0,},
{-2,-4,-2,-4,-3,1,-4,-3,2,-3,5,3,-4,-4,-2,-3,-3,-1,1,-2,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-1,-3,-2,-4,-2,0,-3,-1,2,-2,3,7,-2,-3,0,-2,-2,-1,1,-1,-1,0,-1,0,0,0,0,0,0,0,0,0,},
{-1,4,-2,2,0,-4,0,1,-3,0,-4,-2,7,-2,0,-1,1,0,-3,-4,-1,-2,0,0,0,0,0,0,0,0,0,0,},
{-1,-2,-4,-1,-1,-4,-2,-2,-3,-1,-4,-3,-2,10,-1,-3,-1,-1,-3,-4,-2,-3,-1,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,0,2,-4,-2,1,-3,2,-2,0,0,-1,7,1,0,-1,-3,-1,-1,-1,4,0,0,0,0,0,0,0,0,0,},
{-2,-1,-4,-2,0,-3,-3,0,-4,3,-3,-2,-1,-3,1,7,-1,-1,-3,-3,-1,-1,0,0,0,0,0,0,0,0,0,0,},
{1,0,-1,0,-1,-3,0,-1,-3,0,-3,-2,1,-1,0,-1,5,2,-2,-4,-1,-2,0,0,0,0,0,0,0,0,0,0,},
{0,0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,2,5,0,-3,0,-2,-1,0,0,0,0,0,0,0,0,0,},
{0,-4,-1,-4,-3,-1,-4,-4,4,-3,1,1,-3,-3,-3,-3,-2,0,5,-3,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-3,-5,-5,-5,-3,1,-3,-3,-3,-3,-2,-1,-4,-4,-1,-3,-4,-3,-3,15,-3,2,-2,0,0,0,0,0,0,0,0,0,},
{-1,-1,-2,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-1,-1,-1,0,-1,-3,-1,-1,-1,0,0,0,0,0,0,0,0,0,},
{-2,-3,-3,-3,-2,4,-3,2,-1,-2,-1,0,-2,-3,-1,-1,-2,-2,-1,2,-1,8,-2,0,0,0,0,0,0,0,0,0,},
{-1,2,-3,1,5,-4,-2,0,-3,1,-3,-1,0,-1,4,0,0,-1,-3,-2,-1,-2,5,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}
};
//blosum62
const char CParams::blosum62[32][32]={
{4,-2,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-2,-1,-1,-1,1,0,0,-3,0,-2,-1,0,0,0,0,0,0,0,0,0,},
{-2,4,-3,4,1,-3,-1,0,-3,0,-4,-3,3,-2,0,-1,0,-1,-3,-4,-1,-3,1,0,0,0,0,0,0,0,0,0,},
{0,-3,9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2,-2,-3,0,0,0,0,0,0,0,0,0,},
{-2,4,-3,6,2,-3,-1,-1,-3,-1,-4,-3,1,-1,0,-2,0,-1,-3,-4,-1,-3,1,0,0,0,0,0,0,0,0,0,},
{-1,1,-4,2,5,-3,-2,0,-3,1,-3,-2,0,-1,2,0,0,-1,-2,-3,-1,-2,4,0,0,0,0,0,0,0,0,0,},
{-2,-3,-2,-3,-3,6,-3,-1,0,-3,0,0,-3,-4,-3,-3,-2,-2,-1,1,-1,3,-3,0,0,0,0,0,0,0,0,0,},
{0,-1,-3,-1,-2,-3,6,-2,-4,-2,-4,-3,0,-2,-2,-2,0,-2,-3,-2,-1,-3,-2,0,0,0,0,0,0,0,0,0,},
{-2,0,-3,-1,0,-1,-2,8,-3,-1,-3,-2,1,-2,0,0,-1,-2,-3,-2,-1,2,0,0,0,0,0,0,0,0,0,0,},
{-1,-3,-1,-3,-3,0,-4,-3,4,-3,2,1,-3,-3,-3,-3,-2,-1,3,-3,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,-1,1,-3,-2,-1,-3,5,-2,-1,0,-1,1,2,0,-1,-2,-3,-1,-2,1,0,0,0,0,0,0,0,0,0,},
{-1,-4,-1,-4,-3,0,-4,-3,2,-2,4,2,-3,-3,-2,-2,-2,-1,1,-2,-1,-1,-3,0,0,0,0,0,0,0,0,0,},
{-1,-3,-1,-3,-2,0,-3,-2,1,-1,2,5,-2,-2,0,-1,-1,-1,1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,},
{-2,3,-3,1,0,-3,0,1,-3,0,-3,-2,6,-2,0,0,1,0,-3,-4,-1,-2,0,0,0,0,0,0,0,0,0,0,},
{-1,-2,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2,7,-1,-2,-1,-1,-2,-4,-2,-3,-1,0,0,0,0,0,0,0,0,0,},
{-1,0,-3,0,2,-3,-2,0,-3,1,-2,0,0,-1,5,1,0,-1,-2,-2,-1,-1,3,0,0,0,0,0,0,0,0,0,},
{-1,-1,-3,-2,0,-3,-2,0,-3,2,-2,-1,0,-2,1,5,-1,-1,-3,-3,-1,-2,0,0,0,0,0,0,0,0,0,0,},
{1,0,-1,0,0,-2,0,-1,-2,0,-2,-1,1,-1,0,-1,4,1,-2,-3,0,-2,0,0,0,0,0,0,0,0,0,0,},
{0,-1,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,0,-1,-1,-1,1,5,0,-2,0,-2,-1,0,0,0,0,0,0,0,0,0,},
{0,-3,-1,-3,-2,-1,-3,-3,3,-2,1,1,-3,-2,-2,-3,-2,0,4,-3,-1,-1,-2,0,0,0,0,0,0,0,0,0,},
{-3,-4,-2,-4,-3,1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11,-2,2,-3,0,0,0,0,0,0,0,0,0,},
{0,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-1,-1,0,0,-1,-2,-1,-1,-1,0,0,0,0,0,0,0,0,0,},
{-2,-3,-2,-3,-2,3,-3,2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1,2,-1,7,-2,0,0,0,0,0,0,0,0,0,},
{-1,1,-3,1,4,-3,-2,0,-3,1,-3,-1,0,-1,3,0,0,-1,-2,-3,-1,-2,4,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,},
{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,}
};
//blosum80
const char CParams::blosum80[32][32] = {
{7, -3, -1, -3, -2, -4, 0, -3, -3, -1, -3, -2, -3, -1, -2, -3, 2, 0, -1, -5, -1, -4, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, 6, -6, 6, 1, -6, -2, -1, -6, -1, -7, -5, 5, -4, -1, -2, 0, -1, -6, -8, -3, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-1, -6, 13, -7, -7, -4, -6, -7, -2, -6, -3, -3, -5, -6, -5, -6, -2, -2, -2, -5, -4, -5, -7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, 6, -7, 10, 2, -6, -3, -2, -7, -2, -7, -6, 2, -3, -1, -3, -1, -2, -6, -8, -3, -6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-2, 1, -7, 2, 8, -6, -4, 0, -6, 1, -6, -4, -1, -2, 3, -1, -1, -2, -4, -6, -2, -5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-4, -6, -4, -6, -6, 10, -6, -2, -1, -5, 0, 0, -6, -6, -5, -5, -4, -4, -2, 0, -3, 4, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, -2, -6, -3, -4, -6, 9, -4, -7, -3, -7, -5, -1, -5, -4, -4, -1, -3, -6, -6, -3, -6, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, -1, -7, -2, 0, -2, -4, 12, -6, -1, -5, -4, 1, -4, 1, 0, -2, -3, -5, -4, -2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, -6, -2, -7, -6, -1, -7, -6, 7, -5, 2, 2, -6, -5, -5, -5, -4, -2, 4, -5, -2, -3, -6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-1, -1, -6, -2, 1, -5, -3, -1, -5, 8, -4, -3, 0, -2, 2, 3, -1, -1, -4, -6, -2, -4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, -7, -3, -7, -6, 0, -7, -5, 2, -4, 6, 3, -6, -5, -4, -4, -4, -3, 1, -4, -2, -2, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-2, -5, -3, -6, -4, 0, -5, -4, 2, -3, 3, 9, -4, -4, -1, -3, -3, -1, 1, -3, -2, -3, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, 5, -5, 2, -1, -6, -1, 1, -6, 0, -6, -4, 9, -4, 0, -1, 1, 0, -5, -7, -2, -4, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-1, -4, -6, -3, -2, -6, -5, -4, -5, -2, -5, -4, -4, 12, -3, -3, -2, -3, -4, -7, -3, -6, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-2, -1, -5, -1, 3, -5, -4, 1, -5, 2, -4, -1, 0, -3, 9, 1, -1, -1, -4, -4, -2, -3, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-3, -2, -6, -3, -1, -5, -4, 0, -5, 3, -4, -3, -1, -3, 1, 9, -2, -2, -4, -5, -2, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{2, 0, -2, -1, -1, -4, -1, -2, -4, -1, -4, -3, 1, -2, -1, -2, 7, 2, -3, -6, -1, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, -1, -2, -2, -2, -4, -3, -3, -2, -1, -3, -1, 0, -3, -1, -2, 2, 8, 0, -5, -1, -3, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-1, -6, -2, -6, -4, -2, -6, -5, 4, -4, 1, 1, -5, -4, -4, -4, -3, 0, 7, -5, -2, -3, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-5, -8, -5, -8, -6, 0, -6, -4, -5, -6, -4, -3, -7, -7, -4, -5, -6, -5, -5, 16, -5, 3, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-1, -3, -4, -3, -2, -3, -3, -2, -2, -2, -2, -2, -2, -3, -2, -2, -1, -1, -2, -5, -2, -3, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-4, -5, -5, -6, -5, 4, -6, 3, -3, -4, -2, -3, -4, -6, -3, -4, -3, -3, -3, 3, -3, 11, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{-2, 0, -7, 1, 6, -6, -4, 0, -6, 1, -5, -3, -1, -2, 5, 0, -1, -2, -4, -5, -1, -4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
};
void CParams::getMatrix(char* name, int matrix[32][32])
{
	int i, j;
	if(!strcasecmp(name, "blosum45")){
		for( i = 0; i < 32; i++){
			for( j = 0; j < 32; j++){
				matrix[i][j] = blosum45[i][j];
			}
		}
	}else if(!strcasecmp(name, "blosum50")){
	 	for( i = 0; i < 32; i++){
            for( j = 0; j < 32; j++){
                matrix[i][j] = blosum50[i][j];
            }
        }
	}else if(!strcasecmp(name, "blosum62")){
        for( i = 0; i < 32; i++){
            for( j = 0; j < 32; j++){
                matrix[i][j] = blosum62[i][j];
            }
        }
	}else if(!strcasecmp(name, "blosum80")){
		for( i = 0; i < 32; i++){
            for( j = 0; j < 32; j++){
                matrix[i][j] = blosum80[i][j];
            }
        }
	}else{
		fprintf(stderr, "*************************************************\n");
		fprintf(stderr, "the scoring matrix (%s) can not be found\n", name);
		fprintf(stderr, "the default scoring matrix (BLOSUM62) is used\n");
		fprintf(stderr, "*************************************************\n");

        for( i = 0; i < 32; i++){
            for( j = 0; j < 32; j++){
                matrix[i][j] = blosum62[i][j];
            }
        }
	}
	//check the validaty ofthe matrix;
	for(i = 0; i < 32; i++){
		for(j = 0; j <= i; j ++){
			if(matrix[i][j] != matrix[j][i]){
				printf("values are not equal (%d %d)\n", matrix[i][j], matrix[j][i]);
				getchar();
				break;
			}
		}
	}
}
void CParams::getMatrix(int matrix[32][32]){
	getMatrix(getSubMatrixName(), matrix);
}

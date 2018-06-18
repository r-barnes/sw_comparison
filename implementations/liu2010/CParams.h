/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#ifndef _CPARAMS_H
#define _CPARAMS_H

#define DEFAULT_GAPO			10
#define DEFAULT_GAPE			2
#define DEFAULT_MIN_SCORE		100
#define DEFAULT_TOPSCORE_NUM	10
#define MAX_PATH_LENGTH			1024

#define MAX_AMINO_ACIDS     	23
#define DUMMY_AMINO_ACID   		MAX_AMINO_ACIDS
#define MATRIX_SIZE				(MAX_AMINO_ACIDS + 1)

class CParams
{
public:
	CParams();
	~CParams();

	static void	getSysTime(double * dtime);
	int		parseParams(int argc, char* argv[]);
	int		getGapOpen() {return gapOpen;}
	int		getGapExtend() {return gapExtend;}
	char*	getSubMatrixName() {return subMatName;}
	char*	getQueryFile() {return queryFile;}
	char*	getDbFile() {return dbFile;}
	void 	getMatrix(char* name, int matrix[32][32]);
	void	getMatrix(int matirx[32][32]);
	int		getScoreThreshold() { return scoreThreshold;};
	void	setScoreThreshold(int score){ scoreThreshold = score;}
	int		getTopScoresNum() {return topScoresNum;}
	void	setTopScoresNum(int num){ topScoresNum = num;}
	bool	isUseSingleGPU() {return useSingleGPU;}
	int		getSingleGPUID() {return singleGPUID;}
	bool	isUseSIMTModel() { return useSIMTModel;}
private:
	static void printUsage();
	static const char blosum45[32][32];
	static const char blosum50[32][32];
	static const char blosum62[32][32];
	static const char blosum80[32][32];

	//gap open penalty
	int gapOpen;
	//gap extension penalty
	int gapExtend;
	//score threshold
	int	scoreThreshold;
	//number of top alignment scores
	int topScoresNum;
	//use single-gpu
	bool useSingleGPU;
	int singleGPUID;
	//using SIMT or SIMD models
	bool useSIMTModel;

	char subMatName [MAX_PATH_LENGTH];
	char queryFile[MAX_PATH_LENGTH];
	char dbFile [MAX_PATH_LENGTH];
};
#endif


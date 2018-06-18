/***********************************************
* # Copyright 2009. Liu Yongchao
* # Contact: Liu Yongchao
* #          liuy0039@ntu.edu.sg; nkcslyc@hotmail.com
* #
* # GPL 2.0 applies.
* #
* ************************************************/

#include "CSearchScalar.h"
#include "CSearchVec.h"
#include "CSearchMGPUScalar.h"
#include "CSearchMGPUVec.h"
#include <stdio.h>

int main(int argc, char* argv[])
{
	CParams params;
	CSearch* search = 0;

	//parse parameters
	if(!params.parseParams(argc, argv)){
		return 0;
	}

	//init graphics card device
	pInitDevice(argc, argv);
	
	GPUInfo* info = pGetGPUInfo();

	bool useSIMTModel = params.isUseSIMTModel();
	int singleGPUID = params.getSingleGPUID();
	if((params.isUseSingleGPU() && info->n_device > 0) || info->n_device == 1){
		printf("Using single-GPU (ID %d ) to perform Smith-Waterman\n", singleGPUID);
		
		if(singleGPUID >= info->n_device){
			singleGPUID = info->n_device - 1;
			printf("Use the single GPU with ID %d\n", singleGPUID);
		}
		pSetDevice(info, info->devices[singleGPUID]);	//select the specified compatible GPU
		
		if(useSIMTModel){
			search = new CSearchScalar(&params);
		}else{
			search = new CSearchVec(&params);
		}
	}else if (info->n_device > 1){
		printf("Using multi-GPU to perform Smith-Waterman\n");
		
		if(useSIMTModel){
			search = new CSearchMGPUScalar(&params);
		}else{
			search = new CSearchMGPUVec(&params);
		}
	}else{
		fprintf(stderr, " No compatible device available\n");
		return 0;
	}
	
	if(search){
		search->run();
		delete search;
	}

	return 1;
}

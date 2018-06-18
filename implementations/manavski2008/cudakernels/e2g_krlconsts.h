
#ifndef _TEMPLATE_KERNELCONSTS_H_
#define _TEMPLATE_KERNELCONSTS_H_


//for kernel_2 64 threads x 100 blocks gives 27ms per query on 8800GTX (DB size is 10MB)

// forkernel_6 64 threads x 400 blocks OR 128 threads x 200 blocks gives 5ms per query on 8800GT (DB size is 10MB)

#define NUM_THREADS 128
#define WORD_CODE_TYPE unsigned short
#define QUERY_LENGTH4 8
#define QUERY_LENGTH6 5
#define MAX_RESULTS_PER_QUERY 100
#define MAX_TEXTURE_WIDTH 134000000

// for the hashed kernels
#define MAX_RESULTS_PER_QUERY_HASHED 5120
#define NUM_THREADS_HASHED 128
#define GUARANTEED_RUMOR_LEVEL_HASHED 2

#endif

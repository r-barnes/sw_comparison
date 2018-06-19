/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */
#ifndef _SS_CU_
#define _SS_CU_

// includes, kernels
#include <ss_kernel.cu>
#include <assert.h>


inline bool 
isPowerOfTwo_ss(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
floorPow2_ss(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}


#define BLOCK_SIZE_ss 256

int** g_scanBlockSums_ss;
unsigned int g_numEltsAllocated_ss = 0;
unsigned int g_numLevelsAllocated_ss = 0;

void preallocBlockSums_ss(unsigned int maxNumElements)
{
    assert(g_numEltsAllocated_ss == 0); // shouldn't be called 

    g_numEltsAllocated_ss = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE_ss; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums_ss = (int**) malloc(level * sizeof(int*));
    g_numLevelsAllocated_ss = level;
    
    numElts = maxNumElements;
    level = 0;
    
    do
    {       
        unsigned int numBlocks = 
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1) 
        {
            CUDA_SAFE_CALL(cudaMalloc((void**) &g_scanBlockSums_ss[level++],  
                                      numBlocks * sizeof(int)));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    CUT_CHECK_ERROR("preallocBlockSums_ss");
}

void deallocBlockSums_ss()
{
    for (unsigned int i = 0; i < g_numLevelsAllocated_ss; i++)
    {
        cudaFree(g_scanBlockSums_ss[i]);
    }

    CUT_CHECK_ERROR("deallocBlockSums_ss");
    
    free((void**)g_scanBlockSums_ss);

    g_scanBlockSums_ss = 0;
    g_numEltsAllocated_ss = 0;
    g_numLevelsAllocated_ss = 0;
}


void prescanArrayRecursive_ss(int *outArray, 
                           const int *inArray, 
                           int numElements, 
                           int level)
{
    unsigned int blockSize = BLOCK_SIZE_ss; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo_ss(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2_ss(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo_ss(numEltsLastBlock))
            numThreadsLastBlock = floorPow2_ss(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(int) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(int) * (numEltsPerBlock + extraSpace);

#ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(g_numEltsAllocated_ss >= numElements);
    }
#endif

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);

    // make sure there are no CUDA errors before we start
    CUT_CHECK_ERROR("prescanArrayRecursive before kernels_ss");

    // execute the scan
    if (numBlocks > 1)
    {
        prescan_ss<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums_ss[level],
                                                                 numThreads * 2, 0, 0);
        CUT_CHECK_ERROR("prescanWithBlockSums_ss");
        if (np2LastBlock)
        {
            prescan_ss<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums_ss[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
            CUT_CHECK_ERROR("prescanNP2WithBlockSums_ss");
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive_ss(g_scanBlockSums_ss[level], 
                              g_scanBlockSums_ss[level], 
                              numBlocks, 
                              level+1);

        uniformAdd_ss<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums_ss[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        CUT_CHECK_ERROR("uniformAdd_ss");
        if (np2LastBlock)
        {
            uniformAdd_ss<<< 1, numThreadsLastBlock >>>(outArray, 
                                                     g_scanBlockSums_ss[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
            CUT_CHECK_ERROR("uniformAdd_ss");
        }
    }
    else if (isPowerOfTwo_ss(numElements))
    {
        prescan_ss<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
        CUT_CHECK_ERROR("prescan_ss");
    }
    else
    {
         prescan_ss<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
         CUT_CHECK_ERROR("prescanNP2_ss");
    }
}

void sum_scan(int *outArray, int *inArray, int numElements)
{
    prescanArrayRecursive_ss(outArray, inArray, numElements, 0);
}


#endif // _SS_CU_

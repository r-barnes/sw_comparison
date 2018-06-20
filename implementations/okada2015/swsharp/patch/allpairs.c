/*
allpairs - All-pairs comparison of base sequences based on
Korpar's SW#: CUDA parallelized Smith Waterman.
Copyright (C) 2015 Daiki Okada, contributor Fumihiko Ino

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "swsharp/swsharp.h"

int main(int argc, char* argv[]) {
    Chain* query = NULL;
    Chain* target = NULL;
    int bound;
    int i,j,k;
    char outputFile[6] = {'\0'};
    char qName[15] = {'\0'};
    char tName[15] = {'\0'};
    struct timeval s, e;

    // Usage
    if (argc != 5){
        fprintf(stderr,"Usage : %s <DIR> Strings GPUs 0or1\n",argv[0]);
        exit(1);
    } 

    // set the number of strings
    int strings = 0;
    strings = atoi(argv[2]);

    // set the number of GPUs
    int cards[] = { 1 };
    int cardsLen = 0;
    cardsLen = atoi(argv[3]);

    // select method (0=kizon 1=teian)
    int method = 0;
    method = atoi(argv[4]);

    // create a scorer object
    // match = 1
    // mismatch = -3
    // gap open = 5
    // gap extend = 2
    Scorer* scorer;
    scorerCreateScalar(&scorer, 1, -3, 5, 2);
    Alignment* alignment;

    for (i=0;i<strings-1;i++) {
        bound=0;
        query = NULL;
        sprintf(qName,"%s%d.fasta",argv[1],i);
        printf("query %s\n",qName);
        for (j=i+1;j<strings;j++) {
            alignment = NULL;
            target = NULL;
            sprintf(tName,"%s%d.fasta",argv[1],j);
            printf("target %s\n",tName);
            // read the query and the target
            readFastaChain(&query, qName);
            readFastaChain(&target, tName);
//            printf("read fastaChain\n");
            // set output file name
            sprintf(outputFile,"%d%d",i,j);
            printf("output : %s\n",outputFile);

            // TIMER START
            gettimeofday( &s, NULL );

            //calculate bound
            if( (i>0) && (method==1) ) {
                bound = calculateBound(i,j);
                printf("bound %d\n",bound);
            }
            // do the pairwise alignment, use Smith-Waterman algorithm
            alignPair(&alignment, SW_ALIGN, query, target, scorer, cards, cardsLen, NULL, bound);

            // TIMER STOP
            gettimeofday( &e, NULL );
            printf("time = %lf\n", (e.tv_sec - s.tv_sec) + (e.tv_usec - s.tv_usec)*1.0E-6);

            // output the results in emboss stat-pair format
            outputAlignment(alignment, outputFile, SW_OUT_STAT);

            // clean the memory
        }
    }
    alignmentDelete(alignment);
    chainDelete(query);
    chainDelete(target);
    scorerDelete(scorer);
    return 0;
}

int calculateBound(int query, int target) {
    char input1[5] = {'\0'};
    char input2[5] = {'\0'};
    FILE *fp1, *fp2;
    int i,c;
    int bound=0;
    char readline[32] = {'\0'};
    int mismatch1, gap1, start1, end1;
    int mismatch2, gap2, start2, end2;

    for (i=0;i<query;i++) {
        sprintf(input1,"%d%d",i,query);
        sprintf(input2,"%d%d",i,target);
        if ( ( fp1 = fopen( input1, "r" ) ) == NULL) {
            fprintf( stderr,"%s is not exist\n", input1 );
            exit(1);
        }
        if ( ( fp2 = fopen( input2, "r" ) ) == NULL) {
            fprintf( stderr,"%s is not exist\n", input2 );
            exit(1);
        }

        fgets( readline, 32, fp1 );
        mismatch1 = atoi(readline);
        fgets( readline, 32, fp1 );
        gap1 = atoi(readline);
        fgets( readline, 32, fp1 );
        start1 = atoi(readline);
        fgets( readline, 32, fp1 );
        end1 = atoi(readline);

        fgets( readline, 32, fp2 );
        mismatch2 = atoi(readline);
        fgets( readline, 32, fp2 );
        gap2 = atoi(readline);
        fgets( readline, 32, fp2 );
        start2 = atoi(readline);
        fgets( readline, 32, fp2 );
        end2 = atoi(readline);

        if ( (end1 < start2)||( end2 < start1) ) {
            bound = 0;
        }else {
            if (start1 < start2) start1 = start2;
            if (end1 >= end2) end1 = end2;
            bound = (end1-start1+1-(mismatch1+mismatch2))*1 - (mismatch1+mismatch2)*3 - (gap1+gap2)*5;
            if (bound < 0) bound=0;
        }
        fclose(fp1);
        fclose(fp2);
    }
    return bound;
}

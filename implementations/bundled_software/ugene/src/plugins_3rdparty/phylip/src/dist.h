#ifndef NDIST_H
#define NDIST_H
#include "phylip.h"
#include <U2Core/AppResources.h>
/* version 3.696.
Written by Joseph Felsenstein, Akiko Fuseki, Sean Lamont, and Andrew Keeffe.

Copyright (c) 1993-2014, Joseph Felsenstein
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*
    dist.h: included in fitch, kitsch, & neighbor
*/

#define over            60

typedef long *intvector;

typedef node **pointptr;

#ifndef OLDC
/*function prototypes*/
void dist_alloctree(pointptr *, long, U2::MemoryLocker& memLocker);
void dist_freetree(pointptr *, long);
void allocd(long, pointptr);
void freed(long, pointptr);
void allocw(long, pointptr);
void freew(long, pointptr);
void dist_setuptree(tree *, long);
void dist_inputdata(boolean, boolean, boolean, boolean, vector *, intvector *);
void dist_inputdata_modified(boolean replicates, boolean printdata, boolean lower,
                             boolean upper, vector *x, intvector *reps);
void dist_coordinates(node *, double, long *, double *, node *, boolean);
void dist_drawline(long, double, node *, boolean);
void dist_printree(node *, boolean, boolean, boolean);
void treeoutr(node *, long *, tree *);
void dist_treeout(node *, long *, double, boolean, node *);
/*function prototypes*/
#endif



#endif


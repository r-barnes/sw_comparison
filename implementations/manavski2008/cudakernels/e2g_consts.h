
#ifndef _TEMPLATE_E2GCONSTS_H_
#define _TEMPLATE_E2GCONSTS_H_


#define MAX_NUM_THREADS 32

// query size = 9 x 4B = 36
// the real query size must be at most SOLEXA_QUERY_SIZE - 1
#define SOLEXA_QUERY_SIZE 36

typedef enum { NOT_A_SITE=1, DONOR=2, ACCEPTOR=4 } donor_acceptor;

#endif

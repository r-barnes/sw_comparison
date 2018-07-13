
/*******************************************************************************/
/* Copyright 2008 by Gregory Striemer.  All rights reserved. This program      */
/* may not be sold or incorporated into a commercial product in whole or in    */
/* part, without written consent of Gregory Striemer.  For further information */
/* regarding permission for use or reproduction, please                        */
/* contact: Gregory Striemer at gmstrie@email.ece.arizona.edu                  */
/*******************************************************************************/

#ifndef _SmithWaterman_KERNEL_H_
#define _SmithWaterman_KERNEL_H_

__global__ void
SmithWaterman_Kernel(
  char* device_DatabaseArray,
  int* device_SW_Results,
  int* device_offset,
  int* device_protein_length,
  short int* device_temp_1
);

#endif // #ifndef _SmithWaterman_KERNEL_H_


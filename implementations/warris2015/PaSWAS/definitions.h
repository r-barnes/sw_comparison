#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_


/**
 * Warris's adaptation of the Smith-Waterman algorithm (WASWA).
 *
 * Requires a NVidia Geforce CUDA 2.1 with at least 1.3 compute capability.
 *
 * @author Sven Warris
 * @version 1.1
 */

/** maximum X per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_X 8
/** maximum Y per block (used in dimensions for blocks and amount of shared memory */
#define SHARED_Y 8

/** kernel contains a for-loop in which the score is calculated. */
#define DIAGONAL SHARED_X + SHARED_Y

/** amount of score elements in a single block */
#define blockSize (SHARED_X * SHARED_Y)

/** define which type of matrix you need, comment out the rest */
#define DNA_RNA
//#define BASIC
//#define BLOSUM62

#define NUMBER_SEQUENCES 265
#define X (304)
#define NUMBER_TARGETS 65
#define Y (24)


#define MINIMUM_SCORE 70.0
#define LOWER_LIMIT_MAX_SCORE 0.0

/** amount of blocks across the X axis */
#define XdivSHARED_X (X/SHARED_X)
/** amount of blocks across the Y axis */
#define YdivSHARED_Y (Y/SHARED_Y)


/** start of the alphabet, so scoringsmatrix index can be calculated */
#define characterOffset 'A'
/** character used to fill the sequence if length < X */
#define FILL_CHARACTER 'x'
#define FILL_SCORE -1E10

/** Direction definitions for the direction matrix. These are needed for the traceback */
#define NO_DIRECTION 0
#define UPPER_LEFT_DIRECTION 1
#define UPPER_DIRECTION 2
#define LEFT_DIRECTION 3
#define STOP_DIRECTION 4

/** Specifies the value for which a traceback can be started. If the
 * value in the alignment matrix is larger than or equal to
 * LOWER_LIMIT_SCORE * maxValue the traceback is started at this point.
 * A lower value for LOWER_LIMIT_SCORE will give more aligments.
 */
#define LOWER_LIMIT_SCORE 1.0

/** Only report alignments with a given minimum score. A good setting is:
 * (length shortest seq)*(lowest positive score) - (number of allowed gaps/mismatches)*(lowest negative score)
 * For testing: keep it low to get many alignments back.
 * @todo: make this a config at runtime.
 */

#define ALLOWED_ERRORS 1.0

/** this value is used to allocate enough memory to store the starting points */
#define MAXIMUM_NUMBER_STARTING_POINTS (NUMBER_SEQUENCES*NUMBER_TARGETS*100)

/** these characters are used for the alignment output plot */
#define GAP_CHAR_SEQ '-'
#define GAP_CHAR_ALIGN ' '
#define MISMATCH_CHAR '.'
#define MATCH_CHAR '|'

/**** Scorings Section ****/

/** score used for a gap */
#define gapScore  -5.0
#define HIGHEST_SCORE 5.0

#ifdef DNA_RNA
__constant__ float scoringsMatrix[26][26] = {
{5	,-1	,-3	,-1	,-1	,-1	,-3	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-3	,-3	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-3	,-1	,5	,-1	,-1	,-1	,-3	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-3	,-3	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-3	,-1	,-3	,-1	,-1	,-1	,5	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-3	,-3	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-3	,-1	,-3	,-1	,-1	,-1	,-3	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,5	,-3	,-1	,-1	,-1	,-1	,-1},
{-3	,-1	,-3	,-1	,-1	,-1	,-3	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-3	,5	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1},
{-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1	,-1}};
#endif

#ifdef BLOSUM62
__constant__ float scoringsMatrix[26][26] = 						
			{{4,-2,0,-2,-1,-2,0,-2,-1,-1,-1,-1,-1,-2,-4,-1,-1,-1,1,0,-4,0,-3,-1,-2,-1},
                        {-2,4,-3,4,1,-3,-1,0,-3,-3,0,-4,-3,4,-4,-2,0,-1,0,-1,-4,-3,-4,-1,-3,0},
                        {0,-3,9,-3,-4,-2,-3,-3,-1,-1,-3,-1,-1,-3,-4,-3,-3,-3,-1,-1,-4,-1,-2,-1,-2,-3},
                        {-2,4,-3,6,2,-3,-1,-1,-3,-3,-1,-4,-3,1,-4,-1,0,-2,0,-1,-4,-3,-4,-1,-3,1},
                        {-1,1,-4,2,5,-3,-2,0,-3,-3,1,-3,-2,0,-4,-1,2,0,0,-1,-4,-2,-3,-1,-2,4},
                        {-2,-3,-2,-3,-3,6,-3,-1,0,0,-3,0,0,-3,-4,-4,-3,-3,-2,-2,-4,-1,1,-1,3,-3},
                        {0,-1,-3,-1,-2,-3,6,-2,-4,-4,-2,-4,-3,0,-4,-2,-2,-2,0,-2,-4,-3,-2,-1,-3,-2},
                        {-2,0,-3,-1,0,-1,-2,8,-3,-3,-1,-3,-2,1,-4,-2,0,0,-1,-2,-4,-3,-2,-1,2,0},
                        {-1,-3,-1,-3,-3,0,-4,-3,4,3,-3,2,1,-3,-4,-3,-3,-3,-2,-1,-4,3,-3,-1,-1,-3},
                        {-1,-3,-1,-3,-3,0,-4,-3,3,3,-3,3,2,-3,-4,-3,-2,-2,-2,-1,-4,2,-2,-1,-1,-3},
                        {-1,0,-3,-1,1,-3,-2,-1,-3,-3,5,-2,-1,0,-4,-1,1,2,0,-1,-4,-2,-3,-1,-2,1},
                        {-1,-4,-1,-4,-3,0,-4,-3,2,3,-2,4,2,-3,-4,-3,-2,-2,-2,-1,-4,1,-2,-1,-1,-3},
                        {-1,-3,-1,-3,-2,0,-3,-2,1,2,-1,2,5,-2,-4,-2,0,-1,-1,-1,-4,1,-1,-1,-1,-1},
                        {-2,4,-3,1,0,-3,0,1,-3,-3,0,-3,-2,6,-4,-2,0,0,1,0,-4,-3,-4,-1,-2,0},
                        {-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4},
                        {-1,-2,-3,-1,-1,-4,-2,-2,-3,-3,-1,-3,-2,-2,-4,7,-1,-2,-1,-1,-4,-2,-4,-1,-3,-1},
                        {-1,0,-3,0,2,-3,-2,0,-3,-2,1,-2,0,0,-4,-1,5,1,0,-1,-4,-2,-2,-1,-1,4},
                        {-1,-1,-3,-2,0,-3,-2,0,-3,-2,2,-2,-1,0,-4,-2,1,5,-1,-1,-4,-3,-3,-1,-2,0},
                        {1,0,-1,0,0,-2,0,-1,-2,-2,0,-2,-1,1,-4,-1,0,-1,4,1,-4,-2,-3,-1,-2,0},
                        {0,-1,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1,-1,0,-4,-1,-1,-1,1,5,-4,0,-2,-1,-2,-1},
                        {-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4},
                        {0,-3,-1,-3,-2,-1,-3,-3,3,2,-2,1,1,-3,-4,-2,-2,-3,-2,0,-4,4,-3,-1,-1,-2},
                        {-3,-4,-2,-4,-3,1,-2,-2,-3,-2,-3,-2,-1,-4,-4,-4,-2,-3,-3,-2,-4,-3,11,-1,2,-2},
                        {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-4,-1,-1,-1,-1,-1,-4,-1,-1,-1,-1,-1},
                        {-2,-3,-2,-3,-2,3,-3,2,-1,-1,-2,-1,-1,-2,-4,-3,-1,-2,-2,-2,-4,-1,2,-1,7,-2},
                        {-1,0,-3,1,4,-3,-2,0,-3,-3,1,-3,-1,0,-4,-1,4,0,0,-1,-4,-2,-2,-1,-2,4}};
#endif

#ifdef BASIC
__constant__ char scoringsMatrix[26][26] = {
	{1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1},
	{-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1}};
#endif

/**** Other definitions ****/

/** bit mask to get the negative value of a float, or to keep it negative */
#define SIGN_BIT_MASK 0x80000000
#define MAX_LINE_LENGTH 500
#endif /*DEFINITIONS_H_*/

"""
This module implements the Needleman-Wunsch Algorithm.

The backtracing step is modified to start with maximum score and extend
to the top-left and down to bottom-right. This follows one path that
contains the maximal matching sequence.

This algorithm also implements an X-Drop termination condition.
"""

import os
import argparse
from pyaligner import Sequence, Scorer, DPMatrix

if __name__ == "__main__":
    description = "Smith-Waterman Algorithm with X-Drop"
    parser = argparse.ArgumentParser( description = description )

    parser.add_argument( "input1", type = argparse.FileType(),
                         help = "Sequence File 1" )

    parser.add_argument( "input2", type = argparse.FileType(),
                         help = "Sequence File 2" )

    parser.add_argument( "-x", "--xdrop", type = int, default = 7,
                         help = "X-Drop Value" )

    parser.add_argument( "-m", "--match-score", type = int, default = 1,
                         help = "Match Score" )

    parser.add_argument( "-i", "--mismatch-score", type = int, default = -1,
                         help = "Mismatch Score" )

    parser.add_argument( "-g", "--gap-score", type = int, default = -1,
                         help = "Gap Score" )

    parser.add_argument( "-s", "--semiglobal", action = "store_true",
                         help = "Only run the semi-global alignment." )

    parser.add_argument( "-v", "--verbosity", action = "count", default = 0,
                         help = "Level 1: print match score," + \
                                "2: print match sequences, 3: print dp matrix." )

    args = parser.parse_args()

    seq1 = Sequence( args.input1.read() )
    seq2 = Sequence( args.input2.read() )

    scorer = Scorer( args.match_score, args.mismatch_score,
                     args.gap_score, args.xdrop )

    dp_matrix = DPMatrix( seq1, seq2, scorer, args.semiglobal )

    if args.verbosity == 0:
        print( dp_matrix.calc_alignment_score() )

    if args.verbosity >= 1:
        print( "Exit Alignment Score:", dp_matrix.calc_alignment_score() )
        print( "Best Alignment Score:", dp_matrix.max_score )
        match_seqs = dp_matrix.calc_match_seq()
        print( "Number of matches:", match_seqs[2] )

        if args.verbosity >= 2:
            print( "First Matched Sequence:", match_seqs[0] )
            print( "Second Matched Sequence:", match_seqs[1] )

        if args.verbosity >= 3:
            print()
            print( dp_matrix )

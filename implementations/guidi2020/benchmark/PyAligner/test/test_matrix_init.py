import unittest

from pyaligner import *


class TestMatrixInit ( unittest.TestCase ):

    def test_matrix_init_global ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer )

        self.assertEqual( matrix.seqh.seq_string, seqh.seq_string )
        self.assertEqual( matrix.seqv.seq_string, seqv.seq_string )
        self.assertEqual( matrix.scorer.match, scorer.match )
        self.assertEqual( matrix.scorer.mismatch, scorer.mismatch )
        self.assertEqual( matrix.scorer.gap, scorer.gap )
        self.assertEqual( matrix.scorer.xdrop, scorer.xdrop )
        self.assertEqual( matrix.semiglobal, False )
        self.assertEqual( matrix.dimh, 5 )
        self.assertEqual( matrix.dimv, 6 )
        self.assertEqual( matrix.max_score, 10 )
        self.assertEqual( matrix.max_row, 2 )
        self.assertEqual( matrix.max_col, 2 )
        self.assertEqual( matrix.dp_matrix, [[  0,  -3,  -6,  -9, -12 ],
                                             [ -3,   5,   2, "X", "X" ],
                                             [ -6,   2,  10,   7,   4 ],
                                             [ -9,  -1,   7,   9,   6 ],
                                             [-12, "X",   4,   6,   8 ],
                                             [-15, "X", "X", "X",   5 ]])

    def test_matrix_init_semiglobal ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer, True )

        self.assertEqual( matrix.seqh.seq_string, seqh.seq_string )
        self.assertEqual( matrix.seqv.seq_string, seqv.seq_string )
        self.assertEqual( matrix.scorer.match, scorer.match )
        self.assertEqual( matrix.scorer.mismatch, scorer.mismatch )
        self.assertEqual( matrix.scorer.gap, scorer.gap )
        self.assertEqual( matrix.scorer.xdrop, scorer.xdrop )
        self.assertEqual( matrix.semiglobal, True )
        self.assertEqual( matrix.dimh, 5 )
        self.assertEqual( matrix.dimv, 6 )
        self.assertEqual( matrix.max_score, 10 )
        self.assertEqual( matrix.max_row, 2 )
        self.assertEqual( matrix.max_col, 2 )
        self.assertEqual( matrix.dp_matrix, [[  0,  -3,  -6,  -9, -12 ],
                                             [ -3,   5,   2, "X", "X" ],
                                             [ -6,   2,  10,   7,   4 ],
                                             [ -9,  -1,   7,   9,   6 ],
                                             [-12, "X",   4,   6,   8 ],
                                             [-15, "X", "X", "X",   5 ]])


if __name__ == '__main__':
    unittest.main()

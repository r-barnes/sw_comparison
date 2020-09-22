import unittest

from pyaligner import *


class TestMatrixMatchSeq ( unittest.TestCase ):

    def test_matrix_match_seq_global ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer )
        match_seq = matrix.calc_match_seq()

        self.assertEqual( match_seq[0], "ACTG-" )
        self.assertEqual( match_seq[1], "ACAAA" )
        self.assertEqual( match_seq[2], 2 )

    def test_matrix_match_seq_semiglobal ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer, True )
        match_seq = matrix.calc_match_seq()

        self.assertEqual( match_seq[0], "ACTG" )
        self.assertEqual( match_seq[1], "ACAA" )
        self.assertEqual( match_seq[2], 2 )


if __name__ == '__main__':
    unittest.main()

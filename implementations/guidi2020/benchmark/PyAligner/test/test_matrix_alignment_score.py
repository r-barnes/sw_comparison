import unittest

from pyaligner import *


class TestMatrixAlignmentScore ( unittest.TestCase ):

    def test_matrix_alignment_score_global ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer )
        score = matrix.calc_alignment_score()

        self.assertEqual( score, 5 )

    def test_matrix_alignment_score_semiglobal ( self ):
        scorer = Scorer( 5, -1, -3, 7 )
        seqh = Sequence( "ACTG" )
        seqv = Sequence( "ACAAA" )
        matrix = DPMatrix( seqh, seqv, scorer, True )
        score = matrix.calc_alignment_score()

        self.assertEqual( score, 8 )


if __name__ == '__main__':
    unittest.main()

import unittest

from pyaligner import *


class TestScoreCalcSqDirForw ( unittest.TestCase ):

    def test_score_calc_sq_dir_forw_invalid ( self ):
        scorer = Scorer( 1, -1, -1, 5 )

        self.assertRaises( TypeError, scorer.calc_sq_dir_forw, 0.9, 1, 1, "A", "A" )
        self.assertRaises( TypeError, scorer.calc_sq_dir_forw, 1, 0.9, 1, "A", "A" )
        self.assertRaises( TypeError, scorer.calc_sq_dir_forw, 1, 1, 0.9, "A", "A" )
        self.assertRaises( TypeError, scorer.calc_sq_dir_forw, 1, 1, 1, 0.9, "A" )
        self.assertRaises( TypeError, scorer.calc_sq_dir_forw, 1, 1, 1, "A", 0.9 )

        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, "A", 1, 1, "A", "A" )
        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, 1, "A", 1, "A", "A" )
        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, "A", "A", 1, "A", "A" )

        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, 1, 1, 1, "AB", "A" )
        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, 1, 1, 1, "A", "AB" )
        self.assertRaises( ValueError, scorer.calc_sq_dir_forw, 1, 1, 1, "AB", "AB" )

        self.assertRaises( RuntimeError, scorer.calc_sq_dir_forw, "X", "X", "X", "A", "A" )

    def test_score_calc_sq_dir_forw_left ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_forw( 5, 1, 1, "A", "G" ), (0, 1) )
        self.assertEqual( scorer.calc_sq_dir_forw( 5, 1, 1, "A", "A" ), (0, 1) )
        self.assertEqual( scorer.calc_sq_dir_forw( 5, "X", "X", "A", "A" ), (0, 1) )

    def test_score_calc_sq_dir_forw_above ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_forw( 1, 5, 1, "A", "G" ), (1, 0) )
        self.assertEqual( scorer.calc_sq_dir_forw( 1, 5, 1, "A", "A" ), (1, 0) )
        self.assertEqual( scorer.calc_sq_dir_forw( "X", 5, "X", "A", "A" ), (1, 0) )

    def test_score_calc_sq_dir_forw_diag ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_forw( 1, 1, 5, "A", "G" ), (1, 1) )
        self.assertEqual( scorer.calc_sq_dir_forw( 1, 1, 5, "A", "A" ), (1, 1) )
        self.assertEqual( scorer.calc_sq_dir_forw( "X", "X", 5, "A", "A" ), (1, 1) )
        self.assertEqual( scorer.calc_sq_dir_forw( "X", "X", 6, "A", "A" ), (1, 1) )


if __name__ == '__main__':
    unittest.main()

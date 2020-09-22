import unittest

from pyaligner import *


class TestScoreCalcSqDirBack ( unittest.TestCase ):

    def test_score_calc_sq_dir_back_invalid ( self ):
        scorer = Scorer( 1, -1, -1, 5 )

        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 0.9, 1, 1, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 1, 0.9, 1, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 1, 1, 0.9, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 1, 1, 1, 0.9, "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 1, 1, 1, "A", 0.9, 2 )
        self.assertRaises( TypeError, scorer.calc_sq_dir_back, 1, 1, 1, "A", "A", 0.9 )

        self.assertRaises( ValueError, scorer.calc_sq_dir_back, "A", 1, 1, "A", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_dir_back, 1, "A", 1, "A", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_dir_back, "A", "A", 1, "A", "A", 2 )

        self.assertRaises( ValueError, scorer.calc_sq_dir_back, 1, 1, 1, "AB", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_dir_back, 1, 1, 1, "A", "AB", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_dir_back, 1, 1, 1, "AB", "AB", 2 )

        self.assertRaises( RuntimeError, scorer.calc_sq_dir_back, "X", "X", "X", "A", "A", 2 )
        self.assertRaises( RuntimeError, scorer.calc_sq_dir_back, 5, 1, 1, "A", "G", 7 )
        self.assertRaises( RuntimeError, scorer.calc_sq_dir_back, 1, 5, 1, "A", "G", 7 )
        self.assertRaises( RuntimeError, scorer.calc_sq_dir_back, 1, 1, 5, "A", "G", 7 )

    def test_score_calc_sq_dir_back_left ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_back( 5, 1, 1, "A", "G", 4 ), (0, -1) )
        self.assertEqual( scorer.calc_sq_dir_back( 5, 1, 1, "A", "A", 4 ), (0, -1) )
        self.assertEqual( scorer.calc_sq_dir_back( 5, "X", "X", "A", "A", 4 ), (0, -1) )

    def test_score_calc_sq_dir_back_above ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_back( 1, 5, 1, "A", "G", 4 ), (-1, 0) )
        self.assertEqual( scorer.calc_sq_dir_back( 1, 5, 1, "A", "A", 4 ), (-1, 0) )
        self.assertEqual( scorer.calc_sq_dir_back( "X", 5, "X", "A", "A", 4 ), (-1, 0) )

    def test_score_calc_sq_dir_back_diag ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_dir_back( 1, 1, 5, "A", "G", 4 ), (-1, -1) )
        self.assertEqual( scorer.calc_sq_dir_back( 1, 1, 5, "A", "A", 6 ), (-1, -1) )
        self.assertEqual( scorer.calc_sq_dir_back( "X", "X", 5, "A", "A", 6 ), (-1, -1) )
        self.assertEqual( scorer.calc_sq_dir_back( "X", "X", 6, "A", "A", 7 ), (-1, -1) )


if __name__ == '__main__':
    unittest.main()

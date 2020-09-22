import unittest

from pyaligner import *


class TestScoreCalcSq ( unittest.TestCase ):

    def test_score_calc_sq_invalid ( self ):
        scorer = Scorer( 1, -1, -1, 5 )

        self.assertRaises( TypeError, scorer.calc_sq_value, 0.9, 1, 1, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_value, 1, 0.9, 1, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_value, 1, 1, 0.9, "A", "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_value, 1, 1, 1, 0.9, "A", 2 )
        self.assertRaises( TypeError, scorer.calc_sq_value, 1, 1, 1, "A", 0.9, 2 )
        self.assertRaises( TypeError, scorer.calc_sq_value, 1, 1, 1, "A", "A", 0.9 )

        self.assertRaises( ValueError, scorer.calc_sq_value, "A", 1, 1, "A", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_value, 1, "A", 1, "A", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_value, "A", "A", 1, "A", "A", 2 )

        self.assertRaises( ValueError, scorer.calc_sq_value, 1, 1, 1, "AB", "A", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_value, 1, 1, 1, "A", "AB", 2 )
        self.assertRaises( ValueError, scorer.calc_sq_value, 1, 1, 1, "AB", "AB", 2 )

    def test_score_calc_sq_left ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_value( 5, 1, 1, "A", "G", 7 ), 4 )
        self.assertEqual( scorer.calc_sq_value( 5, 1, 1, "A", "A", 7 ), 4 )
        self.assertEqual( scorer.calc_sq_value( 5, "X", "X", "A", "A", 11 ), "X" )

    def test_score_calc_sq_above ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_value( 1, 5, 1, "A", "G", 7 ), 4 )
        self.assertEqual( scorer.calc_sq_value( 1, 5, 1, "A", "A", 7 ), 4 )
        self.assertEqual( scorer.calc_sq_value( "X", 5, "X", "A", "A", 11 ), "X" )

    def test_score_calc_sq_diag ( self ):
        scorer = Scorer( 1, -1, -1, 5 )
        self.assertEqual( scorer.calc_sq_value( 1, 1, 5, "A", "G", 7 ), 4 )
        self.assertEqual( scorer.calc_sq_value( 1, 1, 5, "A", "A", 7 ), 6 )
        self.assertEqual( scorer.calc_sq_value( "X", "X", 5, "A", "A", 11 ), "X" )
        self.assertEqual( scorer.calc_sq_value( "X", "X", 6, "A", "A", 11 ), 7 )


if __name__ == '__main__':
    unittest.main()

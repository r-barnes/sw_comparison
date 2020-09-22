import unittest

from pyaligner import *


class TestScoreInit ( unittest.TestCase ):

    def test_score_init_invalid ( self ):
        self.assertRaises( TypeError, Scorer, 1, 1, 1, "X" )
        self.assertRaises( TypeError, Scorer, 1, 1, "X", 1 )
        self.assertRaises( TypeError, Scorer, 1, "X", 1, 1 )
        self.assertRaises( TypeError, Scorer, "X", 1, 1, 1 )

    def test_score_init_valid ( self ):
        scorer = Scorer( 1, 1, 1, 1 )
        self.assertEqual( scorer.match, 1 )
        self.assertEqual( scorer.mismatch, 1 )
        self.assertEqual( scorer.gap, 1 )
        self.assertEqual( scorer.xdrop, 1 )

        scorer = Scorer( 1, 2, -3, 4 )
        self.assertEqual( scorer.match, 1 )
        self.assertEqual( scorer.mismatch, 2 )
        self.assertEqual( scorer.gap, -3 )
        self.assertEqual( scorer.xdrop, 4 )


if __name__ == '__main__':
    unittest.main()

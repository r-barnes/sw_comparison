import unittest

from pyaligner import *


class TestSeqInit ( unittest.TestCase ):

    def test_seq_init_not_string ( self ):
        self.assertRaises( TypeError, Sequence, 52 )

    def test_seq_init_not_valid ( self ):
        self.assertRaises( ValueError, Sequence, "xcx" )

    def test_seq_init_sanitize ( self ):
        seq = Sequence( "AcagA" )
        self.assertEqual( seq.seq_string, "ACAGA" )
        seq = Sequence( "AcagA\n" )
        self.assertEqual( seq.seq_string, "ACAGA" )
        seq = Sequence( "AcagA\r\n" )
        self.assertEqual( seq.seq_string, "ACAGA" )

    def test_seq_init_normal ( self ):
        seq = Sequence( "ACAGA" )
        self.assertEqual( seq.seq_string, "ACAGA" )


if __name__ == '__main__':
    unittest.main()

import unittest

from pyaligner import *


class TestSeqSanitize ( unittest.TestCase ):

    def test_seq_sanitize_not_string ( self ):
        seq = Sequence( "A" )
        self.assertRaises( TypeError, seq.sanitize_seq_string, 52 )

    def test_seq_sanitize_sanitize ( self ):
        seq = Sequence( "A" )
        seq_string = seq.sanitize_seq_string( "AcagA" )
        self.assertEqual( seq_string, "ACAGA" )
        seq_string = seq.sanitize_seq_string( "AcagA\n" )
        self.assertEqual( seq_string, "ACAGA" )
        seq_string = seq.sanitize_seq_string( "AcagA\r\n" )
        self.assertEqual( seq_string, "ACAGA" )

    def test_seq_sanitize_normal ( self ):
        seq = Sequence( "A" )
        seq_string = seq.sanitize_seq_string( "ACAGA" )
        self.assertEqual( seq_string, "ACAGA" )


if __name__ == '__main__':
    unittest.main()

import unittest

from pyaligner import *

class TestSeqIsValidSeqString ( unittest.TestCase ):

    def test_seq_valid_string_not_string ( self ):
        seq = Sequence( "A" )
        self.assertRaises( TypeError, seq.is_valid_seq_string, 52 )

    def test_seq_valid_string_valid ( self ):
        seq = Sequence( "A" )
        self.assertEqual( seq.is_valid_seq_string( "A" ), True )
        self.assertEqual( seq.is_valid_seq_string( "C" ), True )
        self.assertEqual( seq.is_valid_seq_string( "G" ), True )
        self.assertEqual( seq.is_valid_seq_string( "T" ), True )
        self.assertEqual( seq.is_valid_seq_string( "ACAGA" ), True )
        self.assertEqual( seq.is_valid_seq_string( "ACGGA" ), True )
        self.assertEqual( seq.is_valid_seq_string( "AAAAA" ), True )
        self.assertEqual( seq.is_valid_seq_string( "CCCCC" ), True )
        self.assertEqual( seq.is_valid_seq_string( "GGGGG" ), True )
        self.assertEqual( seq.is_valid_seq_string( "TTTTT" ), True )

    def test_seq_valid_string_not_valid ( self ):
        seq = Sequence( "A" )
        self.assertEqual( seq.is_valid_seq_string( "AX" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XA" ), False )
        self.assertEqual( seq.is_valid_seq_string( "CX" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XC" ), False )
        self.assertEqual( seq.is_valid_seq_string( "GX" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XG" ), False )
        self.assertEqual( seq.is_valid_seq_string( "TX" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XT" ), False )
        self.assertEqual( seq.is_valid_seq_string( "GXA" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XGA" ), False )
        self.assertEqual( seq.is_valid_seq_string( "TXA" ), False )
        self.assertEqual( seq.is_valid_seq_string( "XTA" ), False )


if __name__ == '__main__':
    unittest.main()

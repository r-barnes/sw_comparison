import unittest

from pyaligner import *

class TestSeqIsValidNucleotide ( unittest.TestCase ):

    def test_seq_valid_nucleotide_not_string ( self ):
        seq = Sequence( "A" )
        self.assertRaises( TypeError, seq.is_valid_nucleotide, 52 )

    def test_seq_valid_nucleotide_valid ( self ):
        seq = Sequence( "A" )
        self.assertEqual( seq.is_valid_nucleotide( "A" ), True )
        self.assertEqual( seq.is_valid_nucleotide( "C" ), True )
        self.assertEqual( seq.is_valid_nucleotide( "G" ), True )
        self.assertEqual( seq.is_valid_nucleotide( "T" ), True )

    def test_seq_valid_nucleotide_not_valid ( self ):
        seq = Sequence( "A" )
        self.assertEqual( seq.is_valid_nucleotide( "B" ), False )
        self.assertEqual( seq.is_valid_nucleotide( "X" ), False )
        self.assertEqual( seq.is_valid_nucleotide( "BX" ), False )
        self.assertEqual( seq.is_valid_nucleotide( "DXC" ), False )


if __name__ == '__main__':
    unittest.main()

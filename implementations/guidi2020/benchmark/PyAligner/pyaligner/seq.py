"""
This module implements the Sequence Class.

A Nucleotide Sequence is represented by a string of nucleotides.
"""

class Sequence():
    """
    Nucleotide Sequence Class
    """

    def __init__ ( self, seq_string ):
        """
        Sequence Constructor.

        Args:
            seq_string (str): Nucleotide string
        """

        if not isinstance( seq_string, str ):
            raise TypeError( "Invalid Sequence String Type" )

        seq_string = self.sanitize_seq_string( seq_string )

        if not self.is_valid_seq_string( seq_string ):
            raise ValueError( "Invalid Sequence String" )

        self.seq_string = seq_string

    def sanitize_seq_string ( self, seq_string ):
        """
        Sanitize the nucleotide string by removing trailing line breaks
        and forcing upper case for consistency.

        Args:
            seq_string (str): Nucleotide string

        Returns:
            (str): Sanitized nucleotide string
        """

        if not isinstance( seq_string, str ):
            raise TypeError( "Invalid Sequence String Type" )

        while seq_string[-1] == '\n':
            seq_string = seq_string[:-1]

        while seq_string[-1] == '\r':
            seq_string = seq_string[:-1]

        return seq_string.upper()

    def is_valid_seq_string ( self, seq_string ):
        """
        Check that a nucleotide string is made up only of nucleotides.

        Args:
            seq_string (str): Nucleotide string

        Returns:
            (bool): True if the seq_string argument is valid
        """

        if not isinstance( seq_string, str ):
            raise TypeError( "Invalid Sequence String Type" )

        return all( [ self.is_valid_nucleotide( c ) for c in seq_string ] )

    def is_valid_nucleotide ( self, char ):
        """
        Check that a character is a nucleotide.

        Args:
            char (char): Nucleotide character

        Returns:
            (bool): True if the char is a valid nucleotide
        """

        if not isinstance( char, str ):
            raise TypeError( "Invalid Nucleotide Type" )

        return char == "A" or char == "C" or char == "T" or char == "G"

    def __iter__ ( self ):
        """Class iterator."""

        return self.seq_string.__iter__()

    def __len__ ( self ):
        """Class length."""

        return len( self.seq_string )

    def __setitem__ ( self, key, item ):
        """Class [] update."""

        self.seq_string[ key ] = item

    def __getitem__ ( self, key ):
        """Class [] access."""

        return self.seq_string[ key ]



"""
This module implements the Scorer Class.

A Scorer tracks the match, mismatch, gap, and xdrop values.
The Scorer is also responsible for calculating the dynamic programming
function value and direction.
"""

class Scorer():
    """Scorer Class"""

    def __init__ ( self, match, mismatch, gap, xdrop ):
        """
        Scorer Constructor.

        Args:
            match (int): Match value

            mismatch (int): Mismatch value

            gap (int): Match value

            xdrop (int): X-Drop termination condition
        """

        if not isinstance( match, int ):
            raise TypeError( "Invalid Match Score Type" )

        if not isinstance( mismatch, int ):
            raise TypeError( "Invalid Mismatch Score Type" )

        if not isinstance( gap, int ):
            raise TypeError( "Invalid Gap Score Type" )

        if not isinstance( xdrop, int ):
            raise TypeError( "Invalid X-Drop Value Type" )

        self.match    = match
        self.mismatch = mismatch
        self.gap      = gap
        self.xdrop    = xdrop

    def calc_sq_value ( self, left_score, above_score, diag_score,
                        nuch, nucv, max_score ):
        """
        Computes the value of a square in the dynamic programming grid
        as a function of previous three squares and the two nucleotides.

        Args:
            left_score (Union[str, int]): Value of square to the left of
                                          output square

            above_score (Union[str, int]): Value of square above the
                                           output square

            diag_score (Union[str, int]): Value of square up and to the
                                          left of output square

            nuch (str): Horizontal sequence nucleotide associated
                        with output square

            nucv (str): Vertical sequence nucleotide associated with
                        output square

            max_score (int): The "best" score used to calculate X-Drop
                             termination condition

        Returns:
            (Union[str, int]): The value of the square or "X" denoting
                               the X-Drop termination condition has been
                               triggered
        """

        if not isinstance( left_score, (int, str) ):
            raise TypeError( "Invalid Left Square Score Type" )

        if not isinstance( above_score, (int, str) ):
            raise TypeError( "Invalid Above Square Score Type" )

        if not isinstance( diag_score, (int, str) ):
            raise TypeError( "Invalid Diagonal Square Score Type" )

        if isinstance( left_score, str ) and left_score != "X":
            raise ValueError( "Invalid Left Square Score Value" )

        if isinstance( above_score, str ) and above_score != "X":
            raise ValueError( "Invalid Above Square Score Value" )

        if isinstance( diag_score, str ) and diag_score != "X":
            raise ValueError( "Invalid Diagonal Square Score Value" )

        if not isinstance( nuch, str ):
            raise TypeError( "Invalid Horizontal Nucleotide Type" )

        if not isinstance( nucv, str ):
            raise TypeError( "Invalid Vertical Nucleotide Type" )

        if not isinstance( max_score, int ):
            raise TypeError( "Invalid Max Score Type" )

        if len( nuch ) != 1 or len( nucv ) != 1:
            raise ValueError( "Invalid Nucleotide Value" )

        potential_values = []

        if diag_score != "X":
            if nuch == nucv:
                potential_values.append( diag_score + self.match )
            else:
                potential_values.append( diag_score + self.mismatch )

        if left_score != "X":
            potential_values.append( left_score + self.gap )

        if above_score != "X":
            potential_values.append( above_score + self.gap )

        if len( potential_values ) == 0:
            return "X"

        res = max( potential_values )

        if res <= max_score - self.xdrop:
            res = "X"

        return res

    def calc_sq_dir_back ( self, left_score, above_score, diag_score,
                           nuch, nucv, this_score ):
        """
        Computes the direction the this_score value came from in the
        dynamic programming grid. This corresponds to the inverse of the
        dynamic programming function.

        Args:
            left_score (Union[str, int]): Value of square to the left of
                                          this_value square

            above_score (Union[str, int]): Value of square above the
                                           this_value square

            diag_score (Union[str, int]): Value of square up and to the
                                          left of this_value square

            nuch (str): Horizontal sequence nucleotide associated
                        with this_value square

            nucv (str): Vertical sequence nucleotide associated with
                        this_value square

            this_score (int): The score of the square in the dynamic
                              programming grid

        Returns:
            (Tuple[int, int]): The direction the this_score value came
                               from given as a vector
        """

        if not isinstance( left_score, (int, str) ):
            raise TypeError( "Invalid Left Square Score Type" )

        if not isinstance( above_score, (int, str) ):
            raise TypeError( "Invalid Above Square Score Type" )

        if not isinstance( diag_score, (int, str) ):
            raise TypeError( "Invalid Diagonal Square Score Type" )

        if isinstance( left_score, str ) and left_score != "X":
            raise ValueError( "Invalid Left Square Score Value" )

        if isinstance( above_score, str ) and above_score != "X":
            raise ValueError( "Invalid Above Square Score Value" )

        if isinstance( diag_score, str ) and diag_score != "X":
            raise ValueError( "Invalid Diagonal Square Score Value" )

        if not isinstance( nuch, str ):
            raise TypeError( "Invalid Horizontal Nucleotide Type" )

        if not isinstance( nucv, str ):
            raise TypeError( "Invalid Vertical Nucleotide Type" )

        if not isinstance( this_score, int ):
            raise TypeError( "Invalid Square Score Type" )

        if len( nuch ) != 1 or len( nucv ) != 1:
            raise ValueError( "Invalid Nucleotide Value" )

        if diag_score != "X":
            if nuch == nucv and diag_score + self.match == this_score:
                return (-1, -1)
            if nuch != nucv and diag_score + self.mismatch == this_score:
                return (-1, -1)

        if left_score != "X":
            if left_score + self.gap == this_score:
                return (0, -1)

        if above_score != "X":
            if above_score + self.gap == this_score:
                return (-1, 0)

        raise RuntimeError( "Failure to calculate direction." )

    def calc_sq_dir_forw ( self, right_score, below_score, diag_score,
                           nuch, nucv ):
        """
        Computes the direction the this_score value went to in the
        dynamic programming grid.

        Args:
            right_score (Union[str, int]): Value of square to the right
                                           of this square

            below_score (Union[str, int]): Value of square below the
                                           this square

            diag_score (Union[str, int]): Value of square down and to
                                          the right of this square

            nuch (str): Horizontal sequence nucleotide associated
                        with this square

            nucv (str): Vertical sequence nucleotide associated with
                        this square

        Returns:
            (Tuple[int, int]): The direction the this value went
                               to given as a vector
        """

        if not isinstance( right_score, (int, str) ):
            raise TypeError( "Invalid Right Square Score Type" )

        if not isinstance( below_score, (int, str) ):
            raise TypeError( "Invalid Below Square Score Type" )

        if not isinstance( diag_score, (int, str) ):
            raise TypeError( "Invalid Diagonal Square Score Type" )

        if isinstance( right_score, str ) and right_score != "X":
            raise ValueError( "Invalid Right Square Score Value" )

        if isinstance( below_score, str ) and below_score != "X":
            raise ValueError( "Invalid Below Square Score Value" )

        if isinstance( diag_score, str ) and diag_score != "X":
            raise ValueError( "Invalid Diagonal Square Score Value" )

        if not isinstance( nuch, str ):
            raise TypeError( "Invalid Horizontal Nucleotide Type" )

        if not isinstance( nucv, str ):
            raise TypeError( "Invalid Vertical Nucleotide Type" )

        if len( nuch ) != 1 or len( nucv ) != 1:
            raise ValueError( "Invalid Nucleotide Value" )

        potential_values = []

        if diag_score != "X":
            potential_values.append( diag_score )

        if right_score != "X":
            potential_values.append( right_score )

        if below_score != "X":
            potential_values.append( below_score )

        if len( potential_values ) == 0:
            raise RuntimeError( "Failure to calculate direction." )

        res = max( potential_values )

        if res == diag_score:
            return (1, 1)

        elif res == right_score:
            return (0, 1)

        elif res == below_score:
            return (1, 0)

        raise RuntimeError( "Failure to calculate direction." )

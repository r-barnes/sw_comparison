"""
This module implements the DPMatrix Class.

A DPMatrix tracks the entire dynamic programming grid and sequences.
"""

import math

from .seq import Sequence
from .score import Scorer

class DPMatrix():
    """
    Dynamic Programming Matrix Class
    """

    def __init__ ( self, seqh, seqv, scorer, semiglobal = False ):
        """
        DPMatrix Constructor.

        Args:
            seqh (Sequence): Horizontal nucleotide sequence

            seqv (Sequence): Vertical nucleotide sequence

            scorer (Scorer): Matrix scoring class

            semiglobal (bool): Flag to specify semi-global alignment
        """

        if not isinstance( seqh, Sequence ):
            raise TypeError( "Invalid Horizontal Sequence Type" )

        if not isinstance( seqv, Sequence ):
            raise TypeError( "Invalid Vertical Sequence Type" )

        if not isinstance( scorer, Scorer ):
            raise TypeError( "Invalid Scorer Type" )

        self.seqh   = seqh
        self.seqv   = seqv
        self.scorer = scorer
        self.semiglobal = semiglobal

        self.dimh = len( seqh ) + 1
        self.dimv = len( seqv ) + 1

        self.dp_matrix = [ [ "I" for h in range( self.dimh ) ]
                           for v in range( self.dimv ) ]

        self.max_score = 0
        self.max_row   = -1
        self.max_col   = -1

        self.fill_grid()

    def fill_grid ( self ):
        """
        Fills in the dynamic programming grid according the semiglobal
        alignment algorithm.
        """

        # Fill Off-Grid Squares
        for h in range( self.dimh ):
            self.dp_matrix[0][h] = h * self.scorer.gap

        for v in range( self.dimv ):
            self.dp_matrix[v][0] = v * self.scorer.gap

        # Fill On-Grid Squares
        for c, h in enumerate( self.seqh ):
            for r, v in enumerate( self.seqv ):
                r_p = r + 1
                c_p = c + 1
                res = self.scorer.calc_sq_value( self.dp_matrix[r_p][c_p-1],
                                                 self.dp_matrix[r_p-1][c_p],
                                                 self.dp_matrix[r_p-1][c_p-1],
                                                 h, v, self.max_score  )

                if res != "X" and res >= self.max_score:
                    self.max_score = res
                    self.max_row   = r_p
                    self.max_col   = c_p

                self.dp_matrix[r_p][c_p] = res

    def calc_alignment_score( self ):
        """
        Calculates the alignment score on termination of the algorithm.

        Returns:
            (int): The exit alignment score
        """
        r = self.max_row
        c = self.max_col

        while r < len( self.seqv ) and c < len( self.seqh ):

            if self.dp_matrix[r][c+1] == "X" and \
               self.dp_matrix[r+1][c] == "X" and \
               self.dp_matrix[r+1][c+1] == "X":
               break

            v = self.seqv[r - 1]
            h = self.seqh[c - 1]

            dirt = self.scorer.calc_sq_dir_forw( self.dp_matrix[r][c+1],
                                                  self.dp_matrix[r+1][c],
                                                  self.dp_matrix[r+1][c+1],
                                                  h, v )

            r += dirt[0]
            c += dirt[1]

        if not self.semiglobal:
            while r < len( self.seqv ) and self.dp_matrix[r+1][c] != "X":
                r += 1

            while c < len( self.seqh ) and self.dp_matrix[r][c+1] != "X":
                c += 1

        return self.dp_matrix[r][c]

    def calc_match_seq ( self ):
        """
        Calculates the matching or edited sequences and counts the total
        number of matches.

        Returns:
            (Tuple[str, str, int]): The horizontal matching sequence,
                                    the vertical matching sequence,
                                    and the total number of matches
        """
        r = self.max_row
        c = self.max_col

        matchv = ""
        matchh = ""
        matches  = 0

        # Extend up and to the left
        while r > 0 and c > 0:
            v = self.seqv[r - 1]
            h = self.seqh[c - 1]

            dirt = self.scorer.calc_sq_dir_back( self.dp_matrix[r][c-1],
                                                 self.dp_matrix[r-1][c],
                                                 self.dp_matrix[r-1][c-1],
                                                 h, v, self.dp_matrix[r][c] )

            r += dirt[0]
            c += dirt[1]

            if dirt[0] == -1:
                matchv += v
            else:
                matchv += "-"

            if dirt[1] == -1:
                matchh += h
            else:
                matchh += "-"

            matches += 1 if h == v else 0

        while r > 1:
            v = self.seqv[r]
            matchv += v
            matchh += "-"
            r -= 1

        while c > 1:
            h = self.seqh[c]
            matchh += h
            matchv += "-"
            c -= 1

        matchv = matchv[::-1]
        matchh = matchh[::-1]

        # Extend down and to to the right
        r = self.max_row
        c = self.max_col

        while r < len( self.seqv ) and c < len( self.seqh ):

            if self.dp_matrix[r][c+1] == "X" and \
               self.dp_matrix[r+1][c] == "X" and \
               self.dp_matrix[r+1][c+1] == "X":
               break

            dirt = self.scorer.calc_sq_dir_forw( self.dp_matrix[r][c+1],
                                                  self.dp_matrix[r+1][c],
                                                  self.dp_matrix[r+1][c+1],
                                                  h, v )

            r += dirt[0]
            c += dirt[1]
            v = self.seqv[r - 1]
            h = self.seqh[c - 1]


            if dirt[0] == 1:
                matchv += v
            else:
                matchv += "-"

            if dirt[1] == 1:
                matchh += h
            else:
                matchh += "-"

        if not self.semiglobal:
            while r < len( self.seqv ) and self.dp_matrix[r+1][c] != "X":
                v = self.seqv[r]
                matchv += v
                matchh += "-"
                r += 1

            while c < len( self.seqh ) and self.dp_matrix[r][c+1] != "X":
                h = self.seqh[c]
                matchh += h
                matchv += "-"
                c += 1

        return ( matchh, matchv, matches )

    def __str__ ( self ):
        """Class string."""

        # Calculate length of longest score
        max_length = 1
        for row in self.dp_matrix:
            for value in row:
                if value == "X" or value == 0:
                    continue
                if value < 0:
                    length = int( math.ceil( math.log( -value + 1, 10 ) ) ) + 1
                else:
                    length = int( math.ceil( math.log( value + 1, 10 ) ) )
                if length > max_length:
                    max_length = length
        format_str = "{:>" + str( max_length ) + "}"

        str_builder = ""

        if len( self.dp_matrix ) <= 1:
            return str_builder

        # Print horizontal sequence on the top of the grid
        str_builder += format_str.format(" ") + " "
        for n in self.seqh:
            str_builder += format_str.format(n) + " "
        str_builder += "\n"

        # Print the rows starting with the appropriate
        # character of the vertical sequence
        for i, r in enumerate( self.dp_matrix[1:] ):
            str_builder += format_str.format( self.seqv[i] ) + " "
            for v in r[1:]:
                str_builder += format_str.format( str(v) ) + " "
            str_builder += '\n'

        # Remove last '\n' and return
        return str_builder[:-1]


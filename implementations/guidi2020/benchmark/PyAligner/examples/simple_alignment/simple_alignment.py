from pyaligner import *

scorer = Scorer( 5, -1, -3, 7 )
seqh = Sequence( "ACTG" )
seqv = Sequence( "ACAAA" )
matrix = DPMatrix( seqh, seqv, scorer, semiglobal = True )

print( "Exit Alignment Score:", matrix.calc_alignment_score() )
print( "Best Alignment Score:", matrix.max_score )
match_seqs = matrix.calc_match_seq()
print( "Number of matches:", match_seqs[2] )
print( "First Matched Sequence:", match_seqs[0] )
print( "Second Matched Sequence:", match_seqs[1] )
print( matrix )

# PyAligner

PyAligner is a python implementation of the Needleman-Wunsch Pairwise
alignment algorithm with an X-Drop termination condition. The focus of
this project has been on correctness rather than performance. This is
so PyAligner can be used as a "ground-truth" alignment algorithm when
examining the correctness of high-performance projects, such as [Xavier](https://github.com/giuliaguidi/xavier).

## Usage

PyAligner can be used directly as a command-line executable:

```
python pyaligner seq1.seq seq2.seq
```

Where the seqX.seq files follow the sequence file format specified below.
For semi-global alignment:

```
python pyaligner seq1.seq seq2.seq -s
```

You can pass the help flag "-h" for more information.

PyAligner can also be used as a library, see the examples directory for
an example.

## Sequence File Format

The grammar is given in extended BNF starting with sequence:

```
sequence = { nucleotide } ;
nucleotide = "A" | "C" | "G" | "T" ;
```

## References

S. B. Needleman and C. D. Wunsch, “A general method applicable
to the search for similarities in the amino acid sequence of two
proteins,” Journal of molecular biology, vol. 48, no. 3, pp. 443–453,
1970.

Z. Zhang, S. Schwartz, L. Wagner, and W. Miller, “A greedy
algorithm for aligning dna sequences,” Journal of Computational
biology, vol. 7, no. 1-2, pp. 203–214, 2000.
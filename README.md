# SWIMM2.0
Enhanced Smith-Waterman implementation for Intel Multicore and Manycore architectures

## Description
SWIMM2.0 is a software to accelerate Smith-Waterman protein database search on Intel Xeon and Xeon Phi processors. SWIMM2.0 extends its previous version adding support to the new Xeon Phi Knights Landing accelerators and the Xeon Skylake general-purpose processors. In particular, this enhanced version takes advantage of multi-threading and SIMD computing through the use of SSE4.1 (128-bit), AVX2 (256-bit) and AVX-512 (512-bit) extensions.

## Usage
Databases must be preprocessed before searching it.

### Parameters
SWIMM2.0 execution

      -S <string> 'preprocess' for database preprocessing, 'search' for database search. [REQUIRED]

* preprocess
```
  -i,   --input=<string> Input sequence filename (must be in FASTA format). [REQUIRED]
  
  -o,   --output=<string> Output filename. [REQUIRED]
```

* search
```
  -q,   --query=<string> Input query sequence filename (must be in FASTA format). [REQUIRED]
  
  -d,   --db=<string> Preprocessed database output filename. [REQUIRED]
  
  -t,   --cpu_threads=<integer> Number of threads (default: SWIMM2.0 select the best choice considering your processor).
  
  -e,   --gap_extend=<integer> Gap extend penalty (default: 2).
  
  -g,   --gap_open=<integer> Gap open penalty (default: 10).

  -p,   --profile=<char> Profile technique: ’Q’ for Query Profile, ’S’ for Score Profile (default: S).
  
  -r,   --top=<integer> Number of scores to show (default: 10). 
  
  -s,   --sm=<string> Substitution matrix. Supported values: blosum45, blosum50, blosum62, blosum80, blosum90, pam30, pam70, pam250 (default: blosum62).
 
  -?,   --help Give this help list
        --usage Give a short usage message
```

### Examples

* Database preprocessing

  `./swimm2 -S preprocess -i db.fasta -o out `
  
  Preprocess *db.fasta* database (*out* will be the preprocessed database name).
  
* Database search


  `./swimm2 -S search -q query.fasta -d out`
  
  Search query sequence *query.fasta* against *out* preprocessed database (*replace swimm2 with the corresponding binary*).
  
  `./swimm2 --help`
  
  `./swimm2 -?`
  
  Print help list.

### Importante notes
* Database and query files must be in FASTA format.
* On Xeon Phi Knights Landing accelerators, we strongly recommend to use MCDRAM through the *numactl* utility, as we found it beneficial in all cases.
* Supported substitution matrixes: BLOSUM45, BLOSUM50, BLOSUM62, BLOSUM80, BLOSUM90, PAM30, PAM70 and PAM250.
* SWIMM2.0 tunes its execution according to your processor features. The first database search may take some minutes but the tuning stage is executed only once.

## Reference
*SWIMM 2.0: enhanced Smith-Waterman on Intel's Multicore and Manycore architectures based on AVX-512 vector extensions*
Enzo Rucci, Carlos García, Guillermo Botella, Armando De Giusti, Marcelo Naiouf and Manuel Prieto-Matías.
*Under evaluation*

*First Experiences Accelerating Smith-Waterman on Intel’s Knights Landing Processor.*
Enzo Rucci, Carlos Garcia, Guillermo Botella, Armando De Giusti, Marcelo Naiouf and Manuel Prieto-Matias
In: Ibrahim S., Choo KK., Yan Z., Pedrycz W. (eds) Algorithms and Architectures for Parallel Processing. ICA3PP 2017. Lecture Notes in Computer Science, vol 10393. Springer, Cham.
DOI: https://doi.org/10.1007/978-3-319-65482-9_42

## Changelog
* November 30, 2017 (v2.0.0-BETA)
Binary code released. *Source code will be releases soon*

## Contact
If you have any question or suggestion, please contact Enzo Rucci (erucci [at] lidi.info.unlp.edu.ar)

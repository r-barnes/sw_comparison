Smith-Waterman Implementation Comparison
==========================================

This repository contains material for comparing the performance of
implementations of the [Smith-Waterman
algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm),
widely used as a step in genome assembly.



Selection Criteria
==========================================

A "good" implementation of the Smith-Waterman algorithm for our purposes must
possess the following properties.

  1. Able to run on a GPU.
  2. Suitable for reads of our lengths: ~10K-100K (long reads) and ~100 vs. ~1K-10K (short reads vs. contigs).
  3. Utilizes the CPU as well (discuss this)
  4. Separable. Not so deeply integrated with another codebase as to require
     excessive dependencies or so as to operate like a blackbox.
  X. TODO

Possible performance metrics: GCUPS, PPW (Performance per Watt), an analog of arithmetic intensity using GCUPS in place of FLOPS...



Candidate Implementations
==========================================

Smith-Waterman Comparison Matrix
--------------------------------

    ID                | R | Software Name    | doi                           | Architecture            | Compiles | TargetLength  |    CUPS | Claims Faster Than | License     | Source dir     | Homepage
    Steinfadt2009     |   | SWAMP            | 10.1109/OCCBIO.2009.12        | ASC                     |          |               |         | TODO               |             |                |                                                       | Uncontacted
    Steinfadt2013     |   | SWAMP            | 10.1016/j.parco.2013.08.008   | ASC                     |          |               |         | TODO               |             |                |                                                       | Uncontacted
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Farrar2007        |   |                  | 10.1093/bioinformatics/btl582 | CPU-SSE2                |          |               |         | TODO               | TODO        | --             |
    Szalkowski2008    |   | SWPS3            | 10.1186/1756-0500-1-107       | CPU-SSE2                |          |               |         | TODO               | MIT         | szalkowski2008 | https://lab.dessimoz.org/swps3/
    Rumble2009        |   | SHRIMP           | 10.1371/journal.pcbi.1000386  | CPU-SIMD                |          |               |         | TODO               | MIT?        | shrimp         | http://compbio.cs.toronto.edu/shrimp/
    David2011         |   | SHRIMP2          | 10.1093/bioinformatics/btr046 | CPU-SIMD                |          |               |         | TODO               | MIT?        | shrimp         | http://compbio.cs.toronto.edu/shrimp/
    Rognes2011        |   | SWIPE            | 10.1186/1471-2105-12-221      | CPU-SSSE3               |          |               |         | Farrar2007         | AGPL-3.0    | rogness2011    |
    Rucci2014         |   | SWIMM            | 10.1109/CLUSTER.2014.6968784  | CPU-Xeon Phi            |          |               |         | TODO               | Unspecified | rucci2015      |
    Zhao2013          |   | SSW              | 10.1371/journal.pone.0082138  | CPU-SIMD                |          |               |         | TODO               | MIT         | zhao2013       |
    Rucci2015         |   | SWIMM            | 10.1002/cpe.3598              | CPU-Xeon Phi            |          |               |         | TODO               | Unspecified | rucci2015      |
    Sjolund2016       |   | DiagonalSW       | software-no-paper             | CPU-SSE4/AltiVec        |          |               |         | TODO               | MIT         | sjolund2016    | http://diagonalsw.sourceforge.net/
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    Liu2006           |   |                  | 10.1007/11758549_29           | GPU-OpenGL              |          |               |         | TODO               | TODO        | --             |
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
    Manavski2008      |   | SWCUDA           | 10.1186/1471-2105-9-S2-S10    | GPU-CUDA                |          |               |         | TODO               | TODO        | manavski2008   | http://bioinformatics.cribi.unipd.it/cuda/swcuda.html
    Munekawa2008      |   |                  | 10.1109/BIBE.2008.4696721     | GPU-CUDA                |          |               |         |                    |             |                |                                                       | Emailed for source code on 2018-06-19. y-munekw address is dead.
    Liu2009           |   |                  | 10.1186/1756-0500-2-73        | GPU-CUDA                |          |               |         | TODO               |             |                | http://cudasw.sourceforge.net/homepage.htm#latest     | CUDASW++2 and CUDASW++3 likely obviate the need to track down this code.
    Akoglu2009        |   |                  | 10.1007/s10586-009-0089-8     | GPU-CUDA                |          |               |         | TODO               |             | striemer2009   |                                                       | Code likely the same as striemer2009
    Ligowski2009      |   |                  | 10.1109/IPDPS.2009.5160931    | GPU-CUDA                |          |               |         | TODO               |             |                |                                                       | Emailed for source code on 2018-06-19. Witold replied 2018-06-19. Sent further request back on 2018-06-19.
    Striemer2009      |   | GSW              | 10.1109/IPDPS.2009.5161066    | GPU-CUDA                |          |               |         | TODO               | Custom      | striemer2009   | http://www2.engr.arizona.edu/~rcl/SmithWaterman.html
    Ling2009          |   |                  | 10.1109/SASP.2009.5226343     | GPU-CUDA                |          |               |         | TODO               |             |                |                                                       |
    Liu2010           |   | CUDASW++ 2.0     | 10.1186/1756-0500-3-93        | GPU-CUDA                |          |               |         | TODO               | GPLv2       | liu2010        | http://cudasw.sourceforge.net/homepage.htm#latest
    Khajeh-Saeed2010  |   |                  | 10.1016/j.jcp.2010.02.009     | GPU-CUDA                |          |               |         | TODO               | Unknown     |                |
    Sandes2010        |   | MASA             | 10.1145/1693453.1693473       | GPU-CUDA                |          |               |         | TODO               |             |                | https://github.com/edanssandes/MASA-Core/wiki         | There are *many* papers from this group.
    Sandes2011        |   | MASA             | 10.1109/IPDPS.2011.114        | GPU-CUDA                |          |               |         | TODO               |             |                | https://github.com/edanssandes/MASA-Core/wiki         | There are *many* papers from this group.
    Hains2011         |   |                  |                               | GPU-CUDA                |          |               |         |                    |             |                |
    Klus2012          |   | BarraCUDA        | 10.1186/1756-0500-5-27        | GPU-CUDA                |          |               |         | TODO               | MIT/GPLv3   | klus2012       | http://seqbarracuda.sourceforge.net/
    Pankaj2012        |   | SWIFT            |                               | GPU-CUDA                |          |               |         | TODO               | GPL-2.0     | pankaj2012     |
    Venkatachalam2012 |   |                  |                               | GPU-CUDA                |          |               |         |                    |             |                |
    Dicker2014        | 6 |                  |                               | GPU-CUDA                |          |               |         | TODO               |             |                |                                                       |
    Sandes2014_hetero |   | MASA             | 10.1145/2555243.2555280       | GPU-CUDA                |          |               |         |                    | GPLv3       |                |                                                       |
    Sandes2014        |   | MASA-CUDAlign3.0 | 10.1109/CCGrid.2014.18        | GPU-CUDA                | Yes      |        228MBP |         |                    | GPLv3       |                |                                                       |
    Okada2015         |   | SW#              | 10.1186/s12859-015-0744-4     | GPU-CUDA                |          |               |         | TODO               |             | okada2015      | http://www-hagi.ist.osaka-u.ac.jp/research/code/
    Warris2015        |   | PaSWAS           | 10.1371/journal.pone.0122524  | GPU-CUDA                | Error    |               |         | TODO               | MIT         | warris2015     |
    Huang2015         |   |                  | 10.1155/2015/185179           | GPU-CUDA                |          |               |         | TODO               |             |                |                                                       | TODO: Should contact
    Sandes2016_masa   |   | MASA             | 10.1145/2858656               | GPU-CUDA                |          |               |         |                    | GPLv3       |                |                                                       |
    Sandes2016        | 9 | MASA-CUDAlign4.0 | 10.1109/TPDS.2016.2515597     | GPU-CUDA                | NoSource |        249MBP |  10.37T |                    | GPLv3       |                |                                                       |
    nvbio_sw          |   | nvbio            | github.com/NVlabs/nvbio       | GPU-CUDA                | Error    |               |         | TODO               | BSD-3       | nvbio_sw       | https://nvlabs.github.io/nvbio/
    ugene             |   | ugene            |                               | GPU-CUDA                |          |               |         | TODO               | GPLv2       | ugene          | http://ugene.net/download.html
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    Liu2013           |   | CUDASW++ 3.0     | 10.1186/1471-2105-14-117      | GPU-CUDA + CPU-SSE      | Yes      |               |         | TODO               | GPLv2       | liu2013        | http://cudasw.sourceforge.net/homepage.htm#latest
    Luo2013           |   | SOAP3            | 10.1371/journal.pone.0065632  | GPU-CUDA + CPU          |          |               |         |                    | GPLv2+      | luo2013        | http://www.cs.hku.hk/2bwt-tools/soap3-dp/             |
    Marcos2014        |   |                  |                               | GPU-CUDA + CPU          |          |               |         | TODO               |             |                |                                                       |
    Warris2018        |   | pyPaSWAS         | 10.1371/journal.pone.0190279  | GPU-CUDA + CPU + Python |          |               |         | TODO               | MIT         | warris2018     |
   
Reviews:

    Muhammadzadeh2014 |   | 
    Pandey2015        | 1 | 10.9790/0661-17264852
    Liu2013_review    |   | 10.5220/0004191202680271

Other methods:

    Myers1986
    Aluru2002
    Rajko2004


Summary of Algorithmic Tricks/Improvements
------------------------------------------

### Liu2006 **GPU Accelerated Smith-Waterman**

### Farrar2007 **Striped Smith–Waterman speeds database searches six times over other SIMD implementations**

### Manavski2008 **CUDA compatible GPU cards as efficient hardware accelerators for Smith-Waterman sequence alignment**

### Rognes2011 **Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation**

### Pankaj2012 **Swift: A GPU-based Smith-Waterman Sequence Alignment Program**

  Video: http://on-demand.gputechconf.com/gtc/2012/video/S0083-Swift-GPU-Based-Smith-Waterman-Sequence-Alignment-Program.flv

### Sandes2014

Code compiles on Titan using the following per the `build.titan` script in `implementations/masa/masa-cudalign/`.

### Sandes2016

Code for 4.0 doesn't seem to be available. TODO: email authors.

### Warris2015

Doesn't compile on Titan. Error `../smithwaterman.h:12:25: error: helper_cuda.h: No such file or directory`.

Probably a result of wanting CUDA 6.0 and using non-standard header includes.

### Liu2013

Compilation succeeded. Straight-forward.

### nvbio

Fork says to use flag `-DGPU_ARCHITECTURE=sm_XX` with cmake. ([Link](https://github.com/vmiheer/nvbio/))

nvbio repo says that support is for GCC 4.8 with CUDA 6.5 ([Link](https://github.com/NVlabs/nvbio/issues/13#issuecomment-156530070)).

Neither original nvlabs repo nor vmiheer repo compile on Titan. Error `nvcc fatal   : Host compiler targets unsupported OS.`

An alternative repo at https://github.com/ngstools/nvbio doesn't exist any more.



Sites examined
--------------

All material on these sites has been examined and linked references downloaded.

 * http://www.nvidia.com/object/cuda_showcase_html.html
 * http://www.nvidia.com/object/bio_info_life_sciences.html

Recordings of talks:

 * https://www.youtube.com/watch?v=dTjvJmOpbM4
 * http://on-demand.gputechconf.com/gtc/2012/video/S0083-Swift-GPU-Based-Smith-Waterman-Sequence-Alignment-Program.flv


Test Data
==========================================

Bulk Download
------------------------------------------

All the test files can be acquired quickly using the following commands:

    wget https://svwh.dl.sourceforge.net/project/cudasw/data/simdb.fasta.gz -P data/
    wget https://iweb.dl.sourceforge.net/project/cudasw/data/Queries.zip    -P data/
    cat data/pacbio_human54x_files | xargs -n 1 -P 4 wget --continue -P data/

Downloads
------------------------------------------

The test data comes from the following sources:

 * http://cudasw.sourceforge.net/homepage.htm#installation : CUDASW++ search for
   "Example query sequences". Download the files `simdb.fasta.gz` and
   `Queries.zip`.
 * http://sourceforge.net/projects/cudasw/files/data : Same as above, but a more
   direct link.
 * The PacBIO Human54x files are drawn from
   [here](http://datasets.pacb.com/2014/Human54x/fast.html) and linked to from
   [here](https://github.com/PacificBiosciences/DevNet/wiki/H_sapiens_54x_release).

Using the Test Data
------------------------------------------

### CUDASW++ (2.0)

An example of running CUDASW++ (2.0) with an arbitrary query from Queries/
against the simdb.fasta database with all the default parameter values:

    ./cudasw -query Queries/P01008.fasta -db simdb.fasta 

The example assumes CUDASW++ (2.0) is compiled as the executable "cudasw",
cudasw is in $PATH, and Queries/ and simdb.fasta are in the current working
directory (simply provide the absolute path if not).

See [here](http://cudasw.sourceforge.net/homepage.htm#installation) for
additional instructions and options for CUDASW++.



Misc
==========================================

Omitted repos:

 * https://github.com/vgteam/gssw conflicts with Zhao2013 and is a generalization, so probably not needed

Reading sequence data:

 * https://bitbucket.org/aydozz/longreads/src/master/kmercode/fq_reader.c


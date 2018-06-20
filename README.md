Smith-Waterman Implementation Comparison
==========================================

This repository contains material for comparing the performance of
implementations of the [Smith-Waterman
algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm),
widely used as a step in genome sequencing.



Selection Criteria
==========================================

A "good" implementation of the Smith-Waterman algorithm for our purposes must
possess the following properties.

  1. Able to run on a GPU
  2. TODO



Candidate Implementations
==========================================

Comparison Matrix
-----------------

    ID                | Software Name | doi                           | Architecture            | Claims Faster Than | License     | Source dir     | Homepage
    Steinfadt2009     | SWAMP         | 10.1109/OCCBIO.2009.12        | ASC                     | TODO               |             |                |                                                       | Uncontacted
    Steinfadt2013     | SWAMP         | 10.1016/j.parco.2013.08.008   | ASC                     | TODO               |             |                |                                                       | Uncontacted
    -------------------
    Farrar2007        |               | 10.1093/bioinformatics/btl582 | CPU-SSE2                | TODO               | TODO        | --             |
    Szalkowski2008    | SWPS3         | 10.1186/1756-0500-1-107       | CPU-SSE2                | TODO               | MIT         | szalkowski2008 | https://lab.dessimoz.org/swps3/
    Rognes2011        | SWIPE         | 10.1186/1471-2105-12-221      | CPU-SSSE3               | Farrar2007         | AGPL-3.0    | rogness2011    |
    Rucci2014         | SWIMM         | 10.1109/CLUSTER.2014.6968784  | CPU-Xeon Phi            | TODO               | Unspecified | rucci2015      |
    Zhao2013          | SSW           | 10.1371/journal.pone.0082138  | CPU-SIMD                | TODO               | MIT         | zhao2013       |
    Rucci2015         | SWIMM         | 10.1002/cpe.3598              | CPU-Xeon Phi            | TODO               | Unspecified | rucci2015      |
    Sjolund2016       | DiagonalSW    |                               | CPU-SSE4/AltiVec        | TODO               | MIT         | sjolund2016    | http://diagonalsw.sourceforge.net/
    -------------------    
    Liu2006           |               | 10.1007/11758549_29           | GPU-OpenGL              | TODO               | TODO        | --             |
    -------------------     
    Manavski2008      | SWCUDA        | 10.1186/1471-2105-9-S2-S10    | GPU-CUDA                | TODO               | TODO        | manavski2008   | http://bioinformatics.cribi.unipd.it/cuda/swcuda.html
    Munekawa2008      |               | 10.1109/BIBE.2008.4696721     | GPU-CUDA                |                    |             |                |                                                       | Emailed for source code on 2018-06-19. y-munekw address is dead.
    Liu2009           |               | 10.1186/1756-0500-2-73        | GPU-CUDA                | TODO               |             |                | http://cudasw.sourceforge.net/homepage.htm#latest     | CUDASW++2 and CUDASW++3 likely obviate the need to track down this code.
    Akoglu2009        |               | 10.1007/s10586-009-0089-8     | GPU-CUDA                | TODO               |             | striemer2009   |                                                       | Code likely the same as striemer2009
    Ligowski2009      |               | 10.1109/IPDPS.2009.5160931    | GPU-CUDA                | TODO               |             |                |                                                       | Emailed for source code on 2018-06-19. Witold replied 2018-06-19. Sent further request back on 2018-06-19.
    Striemer2009      | GSW           | 10.1109/IPDPS.2009.5161066    | GPU-CUDA                | TODO               | Custom      | striemer2009   | http://www2.engr.arizona.edu/~rcl/SmithWaterman.html
    Liu2010           | CUDASW++ 2.0  | 10.1186/1756-0500-3-93        | GPU-CUDA                | TODO               | GPLv2       | liu2010        | http://cudasw.sourceforge.net/homepage.htm#latest
    Khajeh-Saeed2010  |               | 10.1016/j.jcp.2010.02.009     | GPU-CUDA                | TODO               | Unknown     |                |
    Sandes2010        |               | 10.1145/1693453.1693473       | GPU-CUDA                | TODO               |             |                |                                                       | Emailed for source code on 2018-06-19. edans address is dead.
    Hains2011         |               |                               | GPU-CUDA                |                    |             |                |
    Klus2012          | BarraCUDA     | 10.1186/1756-0500-5-27        | GPU-CUDA                | TODO               | MIT/GPLv3   | klus2012       | http://seqbarracuda.sourceforge.net/
    Pankaj2012        | SWIFT         |                               | GPU-CUDA                | TODO               | GPL-2.0     | pankaj2012     |
    Venkatachalam2012 |               |                               | GPU-CUDA                |                    |             |                |
    Okada2015         | SW#           | 10.1186/s12859-015-0744-4     | GPU-CUDA                | TODO               |             | okada2015      | http://www-hagi.ist.osaka-u.ac.jp/research/code/
    Warris2015        | PaSWAS        | 10.1371/journal.pone.0122524  | GPU-CUDA                | TODO               | MIT         | warris2015     |
    nvbio_sw          | nvbio         | github.com/NVlabs/nvbio       | GPU-CUDA                | TODO               | BSD-3       | nvbio_sw       | https://nvlabs.github.io/nvbio/
    ugene             | ugene         |                               | GPU-CUDA                | TODO               | GPLv2       | ugene          | http://ugene.net/download.html
    -------------------    
    Liu2013           | CUDASW++ 3.0  | 10.1186/1471-2105-14-117      | GPU-CUDA + CPU-SSE      | TODO               | GPLv2       | liu2013        | http://cudasw.sourceforge.net/homepage.htm#latest
    Warris2018        | pyPaSWAS      | 10.1371/journal.pone.0190279  | GPU-CUDA + CPU + Python | TODO               | MIT         | warris2018     |

Reviews:

    Muhammadzadeh2014 | 
    Pandey2015        | 10.9790/0661-17264852
    Liu2013_review    | 10.5220/0004191202680271

Summary of Algorithmic Tricks/Improvements
------------------------------------------

### Liu2006 **GPU Accelerated Smith-Waterman**

### Farrar2007 **Striped Smith–Waterman speeds database searches six times over other SIMD implementations**

### Manavski2008 **CUDA compatible GPU cards as efficient hardware accelerators for Smith-Waterman sequence alignment**

### Rognes2011 **Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation**

### Pankaj2012 **Swift: A GPU-based Smith-Waterman Sequence Alignment Program**

  Video: http://on-demand.gputechconf.com/gtc/2012/video/S0083-Swift-GPU-Based-Smith-Waterman-Sequence-Alignment-Program.flv



Sites examined
--------------

All material on these sites has been examined and linked references downloaded.

 * http://www.nvidia.com/object/cuda_showcase_html.html
 * http://www.nvidia.com/object/bio_info_life_sciences.html


Misc
==========================================

Omitted repos:

 * https://github.com/vgteam/gssw conflicts with Zhao2013 and is a generalization, so probably not needed

Reading sequence data:

 * https://bitbucket.org/aydozz/longreads/src/master/kmercode/fq_reader.c
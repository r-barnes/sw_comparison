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

    ID           | Software Name | doi                           | Architecture | Claims Faster Than | Open Source | Source dir
    Liu2006      |               | 10.1007/11758549_29           | GPU          | TODO               | TODO        | --
    Manavski2008 |               | 10.1186/1471-2105-9-S2-S10    | GPU          | TODO               | TODO        | --
    Rognes2011   | SWIPE         | 10.1186/1471-2105-12-221      | CPU-SSSE3    | Farrar2007         | AGPL-3.0    | rogness2011
    Farrar2007   |               | 10.1093/bioinformatics/btl582 | CPU-SSE2     | TODO               | TODO        | --
    Pankaj2012   | SWIFT         |                               | GPU          | TODO               | GPL-2.0     | pankaj2012
    Zhao2013     | SSW           | 10.1371/journal.pone.0082138  | SIMD         | TODO               | MIT         | zhao2013
    Rucci2014    | SWIMM         | 10.1109/CLUSTER.2014.6968784  | Xeon Phi     | TODO               | Unspecified | rucci2015
    Rucci2015    | SWIMM         | 10.1002/cpe.3598              | Xeon Phi     | TODO               | Unspecified | rucci2015
    Warris2015   | PaSWAS        | 10.1371/journal.pone.0122524  | GPU          | TODO               | MIT         | warris2015


Summary of Algorithmic Tricks/Improvements
------------------------------------------

### Liu2006 **GPU Accelerated Smith-Waterman**

### Farrar2007 **Striped Smith–Waterman speeds database searches six times over other SIMD implementations**

### Manavski2008 **CUDA compatible GPU cards as efficient hardware accelerators for Smith-Waterman sequence alignment**

### Rognes2011 **Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation**

### Pankaj2012 **Swift: A GPU-based Smith-Waterman Sequence Alignment Program**

  Video: http://on-demand.gputechconf.com/gtc/2012/video/S0083-Swift-GPU-Based-Smith-Waterman-Sequence-Alignment-Program.flv



Misc
==========================================

Sites with useful information to be extracted:

 * http://www.nvidia.com/object/bio_info_life_sciences.html
 * http://www.nvidia.com/object/cuda_showcase_html.html

Communication:

 * https://github.com/torognes/swipe

Omitted repos:

 * https://github.com/vgteam/gssw conflicts with Zhao2013 and is a generalization, so probably not needed
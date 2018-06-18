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

    ID           | doi                           | Architecture | Claims Faster Than | Open Source | Source dir
    Liu2006      | 10.1007/11758549_29           | GPU          | TODO               | TODO        | --
    Manavski2008 | 10.1186/1471-2105-9-S2-S10    | GPU          | TODO               | TODO        | --
    Rognes2011   | 10.1186/1471-2105-12-221      | CPU-SSSE3    | Farrar2007         | AGPL-3.0    | swipe
    Farrar2007   | 10.1093/bioinformatics/btl582 | CPU-SSE2     | TODO               | TODO        | --
    Zhao2013     | 10.1371/journal.pone.0082138  | SIMD         | TODO               | MIT         | Complete-Striped-Smith-Waterman-Library


Summary of Algorithmic Tricks/Improvements
------------------------------------------

### Liu2006 **GPU Accelerated Smith-Waterman**

### Farrar2007 **Striped Smith–Waterman speeds database searches six times over other SIMD implementations**

### Manavski2008 **CUDA compatible GPU cards as efficient hardware accelerators for Smith-Waterman sequence alignment**

### Rognes2011 **Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation**




Misc
==========================================

Sites with useful information to be extracted:

 * http://www.nvidia.com/object/bio_info_life_sciences.html
 * http://www.nvidia.com/object/cuda_showcase_html.html

Communication:

 * https://github.com/torognes/swipe

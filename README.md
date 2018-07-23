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

Key:

 * R   = Paper rating. 0=very, very bad. 9=Quite good, actually.
 * ?:? = Problem being solved. 1:1, 1:Many, Many:1, Many:Many 

    ID                | R | Software Name    | doi                           | Architecture            | Compiles | ?:? |argetLength BP|             CUPS | Architectural Notes | Files | Blanks | Comments | Code   | Claims Faster Than | License     | Source dir     | Homepage
    Steinfadt2009     |   | SWAMP            | 10.1109/OCCBIO.2009.12        | ASC                     |          |     |              |                  |                     |       |        |          |        | TODO               |             |                |                                                       | Uncontacted
    Steinfadt2013     |   | SWAMP            | 10.1016/j.parco.2013.08.008   | ASC                     |          |     |              |                  |                     |       |        |          |        | TODO               |             |                |                                                       | Uncontacted
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------       |        |          |        | --------------------------------------------------------------------------------------------------------------------------------------
    Farrar2007        |   |                  | 10.1093/bioinformatics/btl582 | CPU-SSE2                |          |     |              | 3.0G             |                     |       |        |          |        | TODO               | TODO        | --             |
    Szalkowski2008    |   | SWPS3            | 10.1186/1756-0500-1-107       | CPU-SSE2                | Yes      |     |              |                  |                     | 23    | 581    | 1360     | 2696   | TODO               | MIT         | szalkowski2008 | https://lab.dessimoz.org/swps3/
    Rumble2009        |   | SHRIMP           | 10.1371/journal.pcbi.1000386  | CPU-SIMD                | Yes      |     |              |                  |                     |       |        |          |        | TODO               | MIT?        | shrimp         | http://compbio.cs.toronto.edu/shrimp/
    David2011         |   | SHRIMP2          | 10.1093/bioinformatics/btr046 | CPU-SIMD                | Yes      |     |              |                  |                     | 108   | 4347   | 3854     | 24752  | TODO               | MIT?        | shrimp         | http://compbio.cs.toronto.edu/shrimp/
    Rognes2011        |   | SWIPE            | 10.1186/1471-2105-12-221      | CPU-SSSE3               | Fixable  |     |              |                  |                     | 15    | 1889   | 808      | 9899   | Farrar2007         | AGPL-3.0    | rogness2011    |                                                       | Emailed ORNL help staff about getting MPIC++ on Titan.
    Rucci2014         |   | SWIMM            | 10.1109/CLUSTER.2014.6968784  | CPU-Xeon Phi            | Error    |     |              |                  |                     | 16    | 789    | 774      | 3542   | TODO               | Unspecified | rucci2015      |
    Zhao2013          |   | SSW              | 10.1371/journal.pone.0082138  | CPU-SIMD                | Yes      |     |              |                  |                     | 11    | 380    | 694      | 2356   | TODO               | MIT         | zhao2013       |
    Rucci2015         |   | SWIMM            | 10.1002/cpe.3598              | CPU-Xeon Phi            | Fixable  |     |              |                  |                     |       |        |          |        | TODO               | Unspecified | rucci2015      |
    Sjolund2016       |   | DiagonalSW       | software-no-paper             | CPU-SSE4/AltiVec        | Yes      |     |              |                  |                     | 19    | 321    | 72       | 1322   | TODO               | MIT         | sjolund2016    | http://diagonalsw.sourceforge.net/
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------       |        |          |        | --------------------------------------------------------------------------------------------------------------------------------------    
    Liu2006           |   |                  | 10.1007/11758549_29           | GPU-OpenGL              |          |     |              |                  |                     |       |        |          |        | TODO               | TODO        | --             |
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------       |        |          |        | --------------------------------------------------------------------------------------------------------------------------------------     
    Munekawa2008      | 9 |                  | 10.1109/BIBE.2008.4696721     | GPU-CUDA                |          | 1:M |63-511v362 90M| 5.65G            |                     |       |        |          |        |                    |             |                |                                                       | Emailed for source code on 2018-06-19. y-munekw address is dead.
    Liu2009           |   |                  | 10.1186/1756-0500-2-73        | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        | TODO               |             |                | http://cudasw.sourceforge.net/homepage.htm#latest     | CUDASW++2 and CUDASW++3 likely obviate the need to track down this code.
    Akoglu2009        | 9 |                  | 10.1007/s10586-009-0089-8     | GPU-CUDA                |RuntimeErr|     |              |                  |                     | 3     | 488    | 171      | 445    | TODO               |             | striemer2009   |                                                       | Code likely the same as striemer2009
    Ligowski2009      |   |                  | 10.1109/IPDPS.2009.5160931    | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        | Manavski2008       |             |                |                                                       | Emailed for source code on 2018-06-19. Witold replied 2018-06-19. Sent further request back on 2018-06-19.
    Striemer2009      |   | GSW              | 10.1109/IPDPS.2009.5161066    | GPU-CUDA                |RuntimeErr|     |              |                  |                     | 3     | 488    | 171      | 445    | TODO               | Custom      | striemer2009   | http://www2.engr.arizona.edu/~rcl/SmithWaterman.html
    Ling2009          |   |                  | 10.1109/SASP.2009.5226343     | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        | TODO               |             |                |                                                       |
    Liu2010           |   | CUDASW++ 2.0     | 10.1186/1756-0500-3-93        | GPU-CUDA                | Yes      |     |              |                  |                     | 23    | 1821   | 2356     | 9174   | TODO               | GPLv2       | liu2010        | http://cudasw.sourceforge.net/homepage.htm#latest
    Khajeh-Saeed2010  |   |                  | 10.1016/j.jcp.2010.02.009     | GPU-CUDA                |          |     |              |                  |                     | 28    | 776    | 553      | 3459   | TODO               | Unknown     |                |
    Sandes2010        |   | MASA             | 10.1145/1693453.1693473       | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        | TODO               |             |                | https://github.com/edanssandes/MASA-Core/wiki         | There are *many* papers from this group.
    Sandes2011        |   | MASA             | 10.1109/IPDPS.2011.114        | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        | TODO               |             |                | https://github.com/edanssandes/MASA-Core/wiki         | There are *many* papers from this group.
    Hains2011         | 6 |                  |                               | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        |                    |             |                |
    Klus2012          |   | BarraCUDA        | 10.1186/1756-0500-5-27        | GPU-CUDA                | Yes      | M:1 |70 v 102M     | Unlisted         | Tesla M2050,M2090   | 54    | 1953   | 2772     | 12653  | TODO               | MIT/GPLv3   | klus2012       | http://seqbarracuda.sourceforge.net/
    Pankaj2012        |   | SWIFT            |                               | GPU-CUDA                | Yes      |     |              |                  |                     | 121   | 5087   | 9662     | 32724  | TODO               | GPL-2.0     | pankaj2012     |
    Venkatachalam2012 | 9 |                  |                               | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        |                    |             |                |
    Sandes2013        |   | CUDAlign2.1      | 10.1109/TPDS.2012.194         | GPU-CUDA                | Yes (3.9)|     | 162kBP-59MBP |                  |                     |       |        |          |        |                    | GPLv3       |                |                                                       |    
    Dicker2014        | 6 |                  |                               | GPU-CUDA                |          |     |              |                  | GTX 460             |       |        |          |        | TODO               |             |                |                                                       |
    Sandes2014_hetero |   | MASA             | 10.1145/2555243.2555280       | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        |                    | GPLv3       |                |                                                       |
    Sandes2014        |   | MASA-CUDAlign3.0 | 10.1109/CCGrid.2014.18        | GPU-CUDA                | Yes (3.9)|     |       228MBP |                  |                     |       |        |          |        |                    | GPLv3       |                |                                                       |
    Okada2015         | 9 | SW#              | 10.1186/s12859-015-0744-4     | GPU-CUDA                | Yes      | M:M |5M v 5M       |  66G (1) 202G (2)|      ????           | 65    | 6537   | 3914     | 17665  | TODO               |             | okada2015      | http://www-hagi.ist.osaka-u.ac.jp/research/code/
    Warris2015        |   | PaSWAS           | 10.1371/journal.pone.0122524  | GPU-CUDA                | Yes      | M:M |              |                  |                     | 19    | 1239   | 652      | 5128   | TODO               | MIT         | warris2015     |
    Huang2015         | 9 |                  | 10.1155/2015/185179           | GPU-CUDA                |          |     |              |                  | Tesla C1060, K20    |       |        |          |        | TODO               |             |                |                                                       | TODO: Should contact
    Sandes2016_masa   |   | MASA             | 10.1145/2858656               | GPU-CUDA                |          |     |              |                  |                     |       |        |          |        |                    | GPLv3       |                |                                                       |
    Sandes2016        | 9 | MASA-CUDAlign4.0 | 10.1109/TPDS.2016.2515597     | GPU-CUDA                | NoSource |     |       249MBP |  10.37T (384)    |                     |       |        |          |        |                    | GPLv3       |                |                                                       |
    nvbio_sw          |   | nvbio            | github.com/NVlabs/nvbio       | GPU-CUDA                | Yes      |     |              |                  |                     |       |        |          |        | TODO               | BSD-3       | nvbio_sw       | https://nvlabs.github.io/nvbio/
    ugene             |   | ugene            |                               | GPU-CUDA                | Error    |     |              |                  |                     |       |        |          |        | TODO               | GPLv2       | ugene          | http://ugene.net/download.html
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------       |        |          |        | -----------------------------------------------------------------------------------------------------------------------------    
    Manavski2008      | 7 | SWCUDA           | 10.1186/1471-2105-9-S2-S10    | GPU-CUDA + CPU-SSE      |RequiresQt|     |              |                  |                     | 68    | 3974   | 2861     | 8715   | TODO               | TODO        | manavski2008   | http://bioinformatics.cribi.unipd.it/cuda/swcuda.html | 
    Liu2013           | 9 | CUDASW++ 3.0     | 10.1186/1471-2105-14-117      | GPU-CUDA + CPU-SSE      | Yes      |     |5k v 35k: 190M|  119G (1) 186G(2)| GeForce GTX 680, 690| 21    | 642    | 568      | 4476   | TODO               | GPLv2       | liu2013        | http://cudasw.sourceforge.net/homepage.htm#latest
    Luo2013           |   | SOAP3            | 10.1371/journal.pone.0065632  | GPU-CUDA + CPU          | Yes      |     |              |                  |TesC2070,M2050;GTX680| 215   | 14057  | 16852    | 74183  | TODO               | GPLv2+      | luo2013        | http://www.cs.hku.hk/2bwt-tools/soap3-dp/             |
    Marcos2014        |   |                  |                               | GPU-CUDA + CPU          |          |     |              |                  |                     |       |        |          |        | TODO               |             |                |                                                       |
    Warris2018        |   | pyPaSWAS         | 10.1371/journal.pone.0190279  | GPU-CUDA + CPU + Python |          |     |              |                  |                     | 39    | 1120   | 1437     | 4766   | TODO               | MIT         | warris2018     |
   
Reviews:

    Muhammadzadeh2014 |   | 
    Pandey2015        | 1 | 10.9790/0661-17264852
    Liu2013_review    |   | 10.5220/0004191202680271

Other methods:

    Myers1986
    Aluru2002:     parallel prefix computation
    Rajko2004:     Improves on the techniques from Aluru2002
    Boukerche2007: MPI-based method
    Zhang2000:     Greedy algorithm

Background:

    Gotoh1982
    Hirschberg1975



Summary of Algorithmic Tricks/Improvements
------------------------------------------

 * Search space reduction
   * Zhang2000:     Greedy algorithm for sequences with low error rates
   * Boukerche2007: Block pruning
   * Sandes2013:    block pruning
   * Okada2015:     Banded
   * Okada2015:     "interpair pruning"
 * Query profile (uses texture cache):
   * Farrar2007:   Variant-striped
   * Manavski2008: Uses it in a standard way. Has a decent diagram.
   * Akoglu2009:   Criticizes Manavski2008 usage. Query profile too large for texture cache, leads to cache misses.
   * Liu2010:      (discusses sequential vs striped)
   * Hains2011
   * Rognes2011:        Variant-sequential
   * Venkatachalam2012: Query profile reduces random access to substitution matrix with sequential profile access
 * Data layouts:
   * Munekawa2008: Notes that local memory cannot be used in a coalesced manner, but that it is the fallback if there are too few registers available, so it is better to explicitly use GM than to implicitly allow LM to be used.
   * Munekawa2008: Sort sequences by length
   * Munekawa2008: Stores (k-1) antidiagonal in shared memory (multiple threads access it) and (k-2) and current antidiagonal in registers (only accessed by a single thread)
   * Munekawa2008: Stores query sequence in constant memory, since all threads refer to it
   * Munekawa2008: Stores database seqeuences in texture memory, possibly only because they take a lot of memory. Not a clear rationale.
   * Manavski2008: Pack char data into integers (4 per int) to make efficient use of local memory accesses.
   * Akoglu2009:   Puts both query sequence and substitution matrix in constant memory because: "reading from the constant cache is as fast as reading from a register if all threads read the same address, which is the case when reading values from the query sequence"
   * Akoglu2009:   Rearranges the substitution matrix for efficient access
   * Liu2010:      Packed data format to better leverage query profile
   * Liu2013:      Sorting the database and queries by length
   * Huang2015:    Interleaving sequences in memory for coalesced access
   * Ligowski2009: Storing scores and backtracking data both in 4-byte integers
 * Input-size dependent choice of algorithms:
   * Hains2011:  Switching between interthread and intrathread parallelism as sequence size changes
   * Dicker2014: Parallel prefix versus diagonal wavefront
   * Luo2013:    If all sequences are within 1% of each other's lengths, sequences are allocated statically. Otherwise an atomic increment is used to reallocate sequences to processors as processing completes.
 * Speculation:
   * TODO:         Speculative calculation of H scores before F dependencies available (CUDASW++2.0)
   * Farrar2007:   For most cells in alignment matrix, F remains at zero and does not contribute to H. Only when H is greater than Ginit+Gext will F start to influence the value of H. So F is not considered initially. If required, a second step tries to correct the introduced errors. Manavski2008 claim their solution, which doesn't use this optimization, runs faster than Farrar2007.
   * Ligowski2009: Only store score information and only as a single byte. Reprocess those sequences which were sufficiently high-scoring using a full algorithm.
 * Storage reduction:
   * Munekawa2008: Stores only three anti-diagonals
   * Munekawa2008: Packs sequences into vector data formatted in type char4. Four succeeding columns are assigned to each thread.
   * Manavski2008: Pack bytes into integers; integer types had just become available
   * Sandes2013:   Using Myers-Miller for linear space
   * Huang2015:    Saving only the most recent rows/columns/diagonals rather than the whole dynamic programming matrix
 * Processing order:
   * Guan1994:          Divide-and-conquer for Myers-Miller
   * Hains2011:         Filling matrix in columns to increase utilization and decrease global memory accesses\
   * Venkatachalam2012: Briefly mentions that assigning multiple rows per thread reduces synchronization costs
 * Fine-tuning block/thread counts:
   * Sandes2013
 * Available as a library:
   * Okada2015: Example code included
 * SIMD instructions
   * Liu2013:           Four adjacent subject sequences from pre-sorted list are assigned to a single thread, each vector lane corresponds to a sequence. Two-dimensional sequence profile is created.
   * Venkatachalam2012: Short vectors can be used to read and manipulate four values at once, rather than using one thread per cell
 * Use of CPU and GPU:
   * Liu2013
   * Luo2013
   * Warris2018
 * Use of local memory:
   * Luo2013: 512kB per-thread local memory is used to store one row for matrices H and E.
 * Multi-GPU:
   * Sandes2014: Splits data into short-phase and long-phase to minimize time spent waiting by downstream GPUs for communication from upstream
 * Calculation time prediction equation:
   * Sandes2014:
 * Pipelining:
   * Venkatachalam2012: Data can be loaded to GPU while other alignments are happening
 * Tricks:
   * Using the modulus operator is extremely inefficient on CUDA

 * Parallel (prefix?) scan
 * Tiling
 * Blazewicz boolean matrices
 * Block pruning
 * Burrow-Wheeler Transformer? (Klus2012)





Summaries of papers and implementation notes
--------------------------------------------

### Szalkowski2008 **SWPS3 – fast multi-threaded vectorized Smith-Waterman for IBM Cell/B.E. and ×86/SSE2**

    mkdir build
    cmake ..
    make

### Striemer2009

    module load cudatoolkit
    qsub -I -A CSC261 -l nodes=1,walltime=30:00
    nvcc -gencode arch=compute_35,code=sm_35 -I. -Iinc *cu *cpp inc/*cpp


### Manavski2008

    #Acquire CUDA 6.5

        wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

    #Install it to (you may need to `mkdir -p` this directory)

        $HOME/os/cuda-6.5/

    #Try compiling:

        module unload pgi
        module remove cudatoolkit
        module load cmake
        module load gcc/4.8.2
        export PATH="$HOME/os/cuda-6.5/bin:$PATH"
        export LIBRARY_PATH="$HOME/os/cuda-6.5/lib64"
        ./comp_cu.sh

### Liu2006 **GPU Accelerated Smith-Waterman**

### Farrar2007 **Striped Smith–Waterman speeds database searches six times over other SIMD implementations**

### Manavski2008 **CUDA compatible GPU cards as efficient hardware accelerators for Smith-Waterman sequence alignment**

### Rumble2009

    make -f Makefile

### David2011

    make -f Makefile

### Liu2010

Compilation succeeded with

    module load cudatoolkit/9.1.85_3.10-1.0502.df1cc54.3.1
    make

### Rognes2011 **Faster Smith-Waterman database searches with inter-sequence SIMD parallelisation**

### Klus2012

Compilation succeeded with

    module load cudatoolkit/9.1.85_3.10-1.0502.df1cc54.3.1
    #Minor makefile adjustment to NVCC path
    make

### Pankaj2012 **Swift: A GPU-based Smith-Waterman Sequence Alignment Program**

  Video: http://on-demand.gputechconf.com/gtc/2012/video/S0083-Swift-GPU-Based-Smith-Waterman-Sequence-Alignment-Program.flv

Compilation successful.

    module load cudatoolkit/7.0.28-1.0502.10280.4.1
    make

### Rucci2014

    make

### Sandes2014

Code compiles on Titan using the following per the `build.titan` script in `implementations/masa/masa-cudalign/`.

### Sandes2016

Code for 4.0 doesn't seem to be available. TODO: email authors.

### Warris2015

Compiled with modifications to Makefile and inclusion of CUDA-deprecated header files.

    cd PaSWAS/onGPU
    module load cudatoolkit

### Luo2013

Compilation succeeded with

    module load cudatoolkit/9.1.85_3.10-1.0502.df1cc54.3.1
    #Several fixes to the code and makefile
    make

### Liu2013

Compilation succeeded. Straight-forward.

### Zhao2013

Compilation succeeded. Straight-forward.

    make

### Okada2015

Compilation successful. Minor alterations of makefile required.

    module load cudatoolkit/7.0.28-1.0502.10280.4.1
    make

Available as a library.

### Sjolund2016

    wget ftp://ftp.gnu.org/gnu/gengetopt/gengetopt-2.22.tar.gz
    tar xvzf gengetopt-2.22.tar.gz
    cd gengetopt-2.22/
    ./configure --prefix=$HOME/os
    #Add `#include <string.h>` to the top of `src/fileutils.cpp`
    make -j 10
    make install
    export PATH="$HOME/os/bin:$PATH"

    module load tbb
    echo $TBB_COMPILE_FLAGS #Get path to TBB
    export LIBRARY_PATH="/lustre/atlas/sw/tbb/43/sles11.3_gnu4.8.2/source/build/linux_intel64_gcc_cc4.8.2_libc2.11.3_kernel3.0.101_release/:$LIBRARY_PATH"

    mkdir build
    cmake ..
    make -j 10

    #Executable is in: build/src/c

### nvbio

Fork says to use flag `-DGPU_ARCHITECTURE=sm_XX` with cmake. ([Link](https://github.com/vmiheer/nvbio/))

nvbio repo says that support is for GCC 4.8 with CUDA 6.5 ([Link](https://github.com/NVlabs/nvbio/issues/13#issuecomment-156530070)).

An alternative repo at https://github.com/ngstools/nvbio doesn't exist any more.

    #Acquire CUDA 6.5

        wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run

    #Install it to (you may need to `mkdir -p` this directory)

        $HOME/os/cuda-6.5/

    #Try compiling:

        module unload pgi
        module remove cudatoolkit
        module load cmake
        module load gcc/4.8.2
        export PATH="$HOME/os/cuda-6.5/bin:$PATH"
        export LIBRARY_PATH="$HOME/os/cuda-6.5/lib64"
        CXX=g++ CC=gcc cmake .. -DGPU_ARCHITECTURE=sm_50 -DCMAKE_INSTALL_PREFIX:PATH=$HOME/os
        make -j 10

### ugene

Seems to require Ubuntu or Fedora. Complicated build process, but a cool idea (generating a deb package on the fly).



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

Generating synthetic data
------------------------------------------

* Might be possible to use: https://github.com/seqan/seqan/tree/master/apps/mason2


Misc
==========================================

Omitted repos:

 * https://github.com/vgteam/gssw conflicts with Zhao2013 and is a generalization, so probably not needed

Reading sequence data:

 * https://bitbucket.org/aydozz/longreads/src/master/kmercode/fq_reader.c



Running on Titan
==========================================

To run on Titan, you'll need to first compile your code. The following, for
example, shows how to compile Striemer2009.

    module load cudatoolkit
    nvcc   -I. -Iinc *cu *cpp inc/*cpp -L${CRAY_LD_LIBRARY_PATH}  -lcudart

You'll then need to either make a batch script or start an interactive batch
job:

    qsub -I -X -A CSC261 -q debug -l nodes=1,walltime=30:00

The only way to access compute nodes if via the `aprun` command. But this
command can only be run from somewhere on the lustre file system. Get there
using (for example):

    cd $MEMBERWORK/csc261
    cd /lustre/atlas/scratch/spinyfan/csc261/

Finally, use `aprun` to run the program:

    aprun ~/crd-swgpu/implementations/striemer2009/SmithWaterman/a.out 

This tar ball includes a sample program and a patch file to SW#,
a CUDA-accelerated Smith-Waterman implementation distributed at
http://sourceforge.net/projects/swsharp/

The aim of the sample program is to accelerate all-pairs comparison
of base sequences by interpair pruning and band optimization.

This code can be compiled with:

patch -p1 < swsharp-allpairs-comparison.patch
make
nvcc allpairs.c -I include/ -L lib/ -l swsharp -l pthread -o allpairs

This code assumes that a series of base sequences are stored in
a single directory that includes a series of files in the fasta format:
01.fasta, 02.fasta, and so on.

The sample program then can be executed with:

./allpairs genome/homosapiens/ 3 1 1

where the first argument is the directory name, the second argument is
the number of files included in the directory, the third argument
indicates the number of GPUs to be used (up to 2), and the last argument
activates interpair pruning (0: inactive, 1: active).

Band optimization can be activated with compiling the swsharp library
with CFLAGS += -DBANDED

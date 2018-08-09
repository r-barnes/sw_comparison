#!/usr/bin/env python3

import sys
import numpy as np

if len(sys.argv)!=5:
  print("{0} <Number of Sequences> <Length of Sequences> <Fasta Out> <Fastq Out>".format(sys.argv[0]))
  sys.exit(-1)

num_seq        = int(sys.argv[1])
seq_len        = int(sys.argv[2])
fasta_filename = sys.argv[3]
fastq_filename = sys.argv[4]

fasta = open(fasta_filename, 'w')
fastq = open(fastq_filename, 'w')

for i in range(num_seq):
  seq = np.random.choice(['G','T','C','A'], size=seq_len)
  seq = ''.join(seq)
  fasta.write(">Seq #{0}\n".format(i))
  fasta.write(seq+"\n")
  fastq.write("@Seq #{0}\n".format(i))
  fastq.write(seq+"\n")
  fastq.write("+\n")
  fastq.write("{0}\n".format('~'*len(seq)))
#!/bin/bash
set -ex

PRGM=test_prog

PREFIX1=""
#i prefix2 can be nvprof. use preferably the following : nvprof --profile-api-trace none -s -f -o /tmp/.nvprof/$(ANALYSIS_FILENAME).nvprof
PREFIX2=""
#PREFIX2="valgrind"
#suffix1 and 2 can be an output file.
SUFFIX1="golden.log"
SUFFIX2="out.log"

OPTARGS1="-p -y local"
OPTARGS2="-p -y local"

FILES_HUMAN600="reads_600_human_10M.fasta ref_600_human_10M.fasta"
FILES_HUMAN300="reads_300_human_10M.fasta ref_300_human_10M.fasta"
FILES_HUMAN150="reads_150_human_10M.fasta ref_150_human_10M.fasta"
FILES_20K="query_batch.fasta target_batch.fasta"
FILES_262K="reads_150.fasta ref_150.fasta"
FILES_SHORT="short_query_batch.fasta short_target_batch.fasta"

echo "human150"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_HUMAN150} ${SUFFIX1}
echo "human150"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_HUMAN150} ${SUFFIX2}
echo "human300"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_HUMAN300} ${SUFFIX1}
echo "human300"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_HUMAN300} ${SUFFIX2}
echo "human600"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_HUMAN600} ${SUFFIX1}
echo "human600"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_HUMAN600} ${SUFFIX2}
echo "run";      ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_SHORT}    ${SUFFIX1}
echo "run2";     ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_SHORT}    ${SUFFIX2}
echo "fullrun";  ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_20K}      ${SUFFIX1}
echo "fullrun2"; ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_20K}      ${SUFFIX2}
echo "262k";     ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS1} ${FILES_262K}     ${SUFFIX1}
echo "262k2";    ${PREFIX1} ${PREFIX2} ./${PRGM} ${OPTARGS2} ${FILES_262K}     ${SUFFIX2}

#echo "cuda-memcheck"; cuda-memcheck ./$(PRGM) $(OPTARGS1) $(FILES_20K) $(SUFFIX1)
#echo "cuda-gdb"; cuda-gdb --args ./test_prog.out -p -y local query_batch.fasta target_batch.fasta
#echo "valgrind"; valgrind ./test_prog.out -p -y local short_query_batch.fasta short_target_batch.fasta
#echo "gdb"; gdb --args ./test_prog.out -p -y local short_query_batch.fasta short_target_batch.fasta
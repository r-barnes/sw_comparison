#!/bin/bash
set -ex

if [ "$#" -ne 2 ]; then
  echo "$0 <QUERY> <REF>"
  exit 1
fi

query=$1
ref=$2

#Can find
threads=1
echo "Ahmd2019 Score Only"
./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n $threads    $query $ref
echo "Ahmed2019 Start"
./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n $threads -s $query $ref
echo "Ahmed2019 Traceback"
./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n $threads -t $query $ref

echo "Khajeh2010"
./build/implementations/khajeh-saeed2010/khajeh2010

echo "Liu2010 simt"
./build/implementations/liu2010/cudasw2 -gapo 6 -gape 1 -mat blosum50 -mod simt -query $query -db $ref

echo "Liu2010 simd"
./build/implementations/liu2010/cudasw2 -gapo 6 -gape 1 -mat blosum50 -mod simd -query $query -db $ref

echo "Liu2013 qprf 1"
./build/implementations/liu2013/cudasw3 -qprf 1 -gapo 6 -gape 1 -mat blosum50 -mod simd -query $query -db $ref

echo "Liu2013 qprf 0"
./build/implementations/liu2013/cudasw3 -num_threads 1 -qprf 0 -gapo 6 -gape 1 -mat blosum50 -mod simd -query $query -db $ref

echo "Okada2015"
./build/implementations/okada2015/swsharpdb -g 6 -e 1 --matrix BLOSUM_50 --algorithm SW --out /dev/null --nocache -T 1 -i $query -j $ref

#TODO: Complicated input?
echo "Pankaj2012"
./build/implementations/pankaj2012/swift -m 1 -M -4 -O -6 -E -1 -o /dev/null -q $query -r $ref

echo "Rognes2011"
threads=1
./build/implementations/rognes2011/swipe -r 1 -q -4 -G 6 -E 1 -a $threads -i $query -d $ref

echo "Sjoluand2016"
threads=1
./build/implementations/sjolund2016/src/c/diagonalsw -i -6 -e -1 -t $threads -q $query -d $ref

echo "Striemer2009"
./build/implementations/striemer2009/striemer $query $ref /dev/null

echo "Szalkowski2008"
threads=1
./build/implementations/szalkowski2008/swps3 -j $threads -i 6 -e 1 implementations/szalkowski2008/matrices/blosum50.mat $query $ref

echo "Awan2020 DNA"
./build/implementations/awan2020/program_gpu dna $query $ref /dev/null

echo "Awan2020 RNA"
./build/implementations/awan2020/program_gpu rna $query $ref /dev/null



#TODO: Warris2015 complicated
#TODO: bowtie2
#TODO: david2011
#TODO: klus2012 - need to understand inputs
#cushaw2 - need to understand inputs

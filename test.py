#!/usr/bin/env python3

from inspect import getcallargs
from pathlib import Path
import itertools
import json
import os
import shelve
import subprocess
import sys

db = shelve.open("results.db")
seq_generator="./build/submodules/alignment_boilerplate/random_sequence_generator"



def generate_data(outname, num, len):
  Path("datasets/").mkdir(parents=True, exist_ok=True)
  if not os.path.isfile(outname):
    subprocess.check_output(f"{seq_generator} {outname} {num} {len}", stderr=subprocess.STDOUT, shell=True)
  return {"name":outname, "num":num, "len":len}



#For M1:1
long_datasets = [
  generate_data("datasets/seq_1M_100_a",   1000000,   100),
  generate_data("datasets/seq_1M_100_b",   1000000,   100),
  # generate_data("datasets/seq_1M_200_a",   1000000,   200),
  # generate_data("datasets/seq_1M_200_b",   1000000,   200),
  # generate_data("datasets/seq_1M_300_a",   1000000,   300),
  # generate_data("datasets/seq_1M_300_b",   1000000,   300),
  generate_data("datasets/seq_1M_500_a",   1000000,   500),
  generate_data("datasets/seq_1M_500_b",   1000000,   500),
  generate_data("datasets/seq_1M_1000_a",  1000000,  1000),
  generate_data("datasets/seq_1M_1000_b",  1000000,  1000),
  # generate_data("datasets/seq_1M_2000_a",  1000000,  2000),
  # generate_data("datasets/seq_1M_2000_b",  1000000,  2000),
  # generate_data("datasets/seq_1M_3000_a",  1000000,  3000),
  # generate_data("datasets/seq_1M_3000_b",  1000000,  3000),
  generate_data("datasets/seq_1M_5000_a",  1000000,  5000),
  generate_data("datasets/seq_1M_5000_b",  1000000,  5000),
  generate_data("datasets/seq_1M_10000_a", 1000000, 10000),
  generate_data("datasets/seq_1M_10000_b", 1000000, 10000),
]

#For A:1
many_to_one_queries = [
  generate_data("datasets/seq_1K_100",   1000,   100),
  generate_data("datasets/seq_1K_500",   1000,   500),
  generate_data("datasets/seq_1K_1000",  1000,  1000),
  generate_data("datasets/seq_1K_5000",  1000,  5000),
  generate_data("datasets/seq_1K_10000", 1000, 10000),
]

many_to_one_targets = [
  generate_data("datasets/seq_1H_10K",  100,   100),
  generate_data("datasets/seq_1H_50K",  100,   500),
  generate_data("datasets/seq_1H_100K", 100,  1000),
  generate_data("datasets/seq_1H_1M",   100,  5000),
  generate_data("datasets/seq_1H_10M",  100, 10000),
]





def all_pairs_data_combos(datasets):
  #First generate an all-pairs-but-self on the a-sequences
  a_datasets = [x for x in datasets if "_a" in x["name"]]
  a_combos = list(itertools.combinations(a_datasets, 2))
  same_len_combos = itertools.combinations(datasets, 2)
  same_len_combos = [x for x in same_len_combos if x[0]["len"]==x[1]["len"]]
  return a_combos + same_len_combos



def get_time_from_output(output, time_start_pattern="Elapsed (wall clock)"):
  for x in output:
    x = x.strip()
    if x.startswith("CUDA error"):
      raise Exception("CUDA Error was raised!")
    if x.startswith(time_start_pattern):
      _, minutes, seconds = x.rsplit(":",2)
      return 60*float(minutes)+float(seconds)
  raise Exception("Could not find time!")


def run_timed_command(command):
  command = "/usr/bin/time -v "+command
  output = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
  output = output.decode("utf-8")
  print(output)
  output = output.split("\n")
  return get_time_from_output(output)



def get_average_time(command, query_data, ref_data, threads=1, warmup=1, run_count=1):
  if "{threads}" in command:
    command = command.format(query=query_data["name"], ref=ref_data["name"], threads=threads)
  else:
    command = command.format(query=query_data["name"], ref=ref_data["name"])
  print(f"\033[94mRunning {command}\033[39m")
  for i in range(warmup):
    subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True)
  total_time = 0
  for i in range(run_count):
    print(f"{i} of {run_count}")
    total_time += run_timed_command(command)
  return total_time/run_count



def cells_for_multiple_11(query_data, ref_data):
  assert query_data["num"]==ref_data["num"]
  return query_data["num"]*query_data["len"]*ref_data["len"]

def cells_for_many_to_one(query_data, ref_data):
  return query_data["num"]*query_data["len"]*ref_data["num"]*ref_data["len"]


def get_giga_cups(name, command, query_data, ref_data, cells_function, threads=1, warmup=1, run_count=1):
  key = json.dumps((name, command, query_data, ref_data))
  if key in db:
    return db[key]

  print(f"\033[94mTiming {command}\033[39m")
  try:
    avg_time = get_average_time(command, query_data, ref_data, threads=threads, warmup=warmup, run_count=run_count)
  except Exception as err:
    db[key] = str(err)
    db.sync()
    print(err)
    return
  cell_count = cells_function(query_data, ref_data)
  gcups      = cell_count/(1e9)/avg_time

  val = {"name":name, "query":query_data, "ref":ref_data, "gcups":gcups, "threads":threads, "avg_time": avg_time, "cells": cell_count}

  db[key] = val
  db.sync()
  return val

long_combos = all_pairs_data_combos(long_datasets)

print("Length",len(long_combos))

threads=1
# for query, ref in long_combos:
#   get_giga_cups("Awan 2020", "./build/implementations/awan2020/program_gpu dna {query} {ref} /dev/null", query, ref, cells_for_multiple_11)
#   get_giga_cups("Awan 2020", "./build/implementations/awan2020/program_gpu rna {query} {ref} /dev/null", query, ref, cells_for_multiple_11)

#   get_giga_cups("Ahmd2019 Score Only", "./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n {threads}    {query} {ref}", query, ref, cells_for_multiple_11, threads=threads)
#   get_giga_cups("Ahmed2019 Start",     "./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n {threads} -s {query} {ref}", query, ref, cells_for_multiple_11, threads=threads)
#   get_giga_cups("Ahmed2019 Traceback", "./build/implementations/ahmed2019/test_prog -y local -a 1 -b 4 -q 6 -r 1 -n {threads} -t {query} {ref}", query, ref, cells_for_multiple_11, threads=threads)


for query, ref in itertools.product(many_to_one_queries, many_to_one_targets):
  get_giga_cups("Liu2010 simt", "./build/implementations/liu2010/cudasw2 -gapo 6 -gape 1 -mat blosum50 -mod simt -query {query} -db {ref}", query, ref, cells_for_many_to_one)
  get_giga_cups("Liu2010 simd", "./build/implementations/liu2010/cudasw2 -gapo 6 -gape 1 -mat blosum50 -mod simd -query {query} -db {ref}", query, ref, cells_for_many_to_one)

  get_giga_cups("Liu2013 qprf 1", "./build/implementations/liu2013/cudasw3 -num_threads {threads} -qprf 1 -gapo 6 -gape 1 -mat blosum50 -query {query} -db {ref}", query, ref, cells_for_many_to_one, threads=threads)
  get_giga_cups("Liu2013 qprf 0", "./build/implementations/liu2013/cudasw3 -num_threads {threads} -qprf 0 -gapo 6 -gape 1 -mat blosum50 -query {query} -db {ref}", query, ref, cells_for_many_to_one, threads=threads)

  get_giga_cups("Okada2015", "./build/implementations/okada2015/swsharpdb -g 6 -e 1 --matrix BLOSUM_50 --algorithm SW --out /dev/null --nocache -T 1 -i {query} -j {ref}", query, ref, cells_for_many_to_one)

  get_giga_cups("Rognes2011",   "./build/implementations/rognes2011/swipe -r 1 -q -4 -G 6 -E 1 -a {threads} -i {query} -d {ref}", query, ref, cells_for_many_to_one, threads=threads)
  get_giga_cups("Sjoluand2016", "./build/implementations/sjolund2016/src/c/diagonalsw -i -6 -e -1 -t {threads} -q {query} -d {ref}", query, ref, cells_for_many_to_one, threads=threads)
  get_giga_cups("Striemer2009", "./build/implementations/striemer2009/striemer {query} {ref} /dev/null", query, ref, cells_for_many_to_one)
  get_giga_cups("Szalkowski2008", "./build/implementations/szalkowski2008/swps3 -j {threads} -i -6 -e -1 implementations/szalkowski2008/matrices/blosum50.mat {query} {ref}", query, ref, cells_for_many_to_one, threads=threads)
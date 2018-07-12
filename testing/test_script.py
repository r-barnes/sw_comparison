#!/usr/bin/env python3

import os
import pandas as pd
from plumbum import local
from plumbum import FG, BG, TEE
import re

#Where the implementations are stored with respect to the working directory of
#this script
aligner_base = "implementations/"

def ParseOutput(output):
  data_out = {}
  output   = output.split("\n")
  for line in output:
    datum = re.match("^STATOUT [a-zA-Z] ([A-Za-z -]) = ([0-9.]+)", line)
    if not datum:
      continue
    data_out[datum.groups(1)] = float(datum.groups(2))
  return output

def AlignerCudaSWpp3(queryfile, databasefile, device):
  #NOTE: Left out `mat` argument for substitution matrix
  #NOTE: Left out `gapo` gap open penalty
  #NOTE: Left out `gape` gap extension penalty
  #NOTE: Left out other arguments...
  aligner = local[os.join(aligner_base,"liu2013/src/cudasw")]
  retcode, stdout, stderr = aligner['-query ',queryfile,'-db',databasefile,'-use_single',device] & TEE
  if retcode!=0:
    raise Exception("Crashed...")
  return ParseOutput(output)




aligners = [AlignerCudaSWpp3]


df = pd.DataFrame()

for aligner in aligners:
  df.append(aligner(queryfile,databasefile,device=0))

df.save_csv("output.csv")
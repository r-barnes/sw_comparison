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
  output = output.split("\n")
  for line in output:
    datum = re.match("^STATOUT ([a-zA-Z]*): ([\d\.]*)", line)
    if datum:
      data_out[datum.groups(1)[0]] = float(datum.groups(1)[1])
  return data_out

def AlignerCUDASWpp3(queryfile, databasefile, device):
  #NOTE: Left out `mat` argument for substitution matrix
  #NOTE: Left out `gapo` gap open penalty
  #NOTE: Left out `gape` gap extension penalty
  #NOTE: Left out other arguments...
  aligner = local[os.path.join(aligner_base, "liu2013/cudasw")]
  retcode, stdout, stderr = aligner['-query', queryfile, '-db', databasefile, '-use_single', device] & TEE
  if retcode != 0:
    raise Exception("Crashed...")
  return ParseOutput(stdout)



aligners = [AlignerCUDASWpp3]
queryfile = ""
databasefile = ""

df = pd.DataFrame()

for aligner in aligners:
  aligner_out = aligner(queryfile, databasefile, device=0)
  aligner_out["name"] = aligner.__name__
  df = df.append(aligner_out, ignore_index = True)

df = df.set_index("name")
df.save_csv("output.csv")

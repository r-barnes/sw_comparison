#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from plumbum import local
from plumbum import FG, BG, TEE
import re
import sys

#Where the implementations are stored with respect to the working directory of
#this script
aligner_base  = "implementations/"


def ParseOutput(output):
  data_out = {}
  output = output.split("\n")
  for line in output:
    datum = re.match("^STATOUT ([a-zA-Z]+): ([\d\.]+)", line)
    if datum:
      stat_key           = datum.groups(1)[0]
      stat_val           = float(datum.groups(1)[1])
      data_out[stat_key] = stat_val
  return data_out



def ParsedCombiner(*args):
  stat_combiner = {"time:": sum, "gcups": min}
  baseline      = {"time": 0, "gcups": 0}
  for arg in args:
    for k, v in arg.items():
      if not k in baseline:
        raise Exception("No baseline for '{0}'!".format(k))
      baseline[k] = stat_combiner[k](baseline[k],v)
  return baseline



def AlignerLiu2013(queryfile, databasefile, device):
  #NOTE: Left out `mat` argument for substitution matrix
  #NOTE: Left out `gapo` gap open penalty
  #NOTE: Left out `gape` gap extension penalty
  #NOTE: Left out other arguments...
  aligner = local[os.path.join(aligner_base, "liu2013/cudasw")]
  retcode, stdout, stderr = aligner['-query', queryfile, '-db', databasefile, '-use_single', device] & TEE
  if retcode != 0:
    raise Exception("Crashed...")
  return ParseOutput(stdout)

def AlignerLiu2010(queryfile, databasefile, device):
  #NOTE: Left out `mat` argument for substitution matrix
  #NOTE: Left out `gapo` gap open penalty
  #NOTE: Left out `gape` gap extension penalty
  #NOTE: Left out other arguments...
  aligner = local[os.path.join(aligner_base, "liu2010/cudasw")]
  retcode, stdout, stderr = aligner['-query', queryfile, '-db', databasefile, '-use_single', device] & TEE
  if retcode != 0:
    raise Exception("Crashed...")
  return ParseOutput(stdout)

def AlignerOkada2015(queryfile, databasefile, device):
  aligner = local[os.path.join(aligner_base, "okada2015/bin/swsharpdb")]
  retcode, stdout, stderr = aligner['-i', queryfile, '-j', databasefile,'--nocache','--outfmt','light','--cards',device]
  if retcode!=0:
    raise Exception("Crashed...")
  return ParseOutput(stdout)

#NOTE: All query strings must be the same length. They may be padded with `N`.
def AlignerGupta2012(queryfile, databasefile, device):
  aligner = local[os.path.join(aligner_base, "pankaj2012/bin/swift")]
  retcode, stdout, stderr = aligner['-q', queryfile, '-r', databasefile,'-n','-1','-s','-1','-o','zout']
  if retcode!=0:
    raise Exception("Crashed...")
  return ParseOutput(stdout)

def AlignerKlus2012(queryfile, databasefile, device):
  barracuda = local[os.path.join(aligner_base, "klus2012/bin/barracuda")]
  i_retcode, i_stdout, i_stderr = barracuda['index',databasefile]
  if retcode!=0:
    raise Exception("Crashed...")
  a_retcode, a_stdout, a_stderr = barracuda['aln',databasefile,queryfile]
  if retcode!=0:
    raise Exception("Crashed...")  
  i_parsed = ParseOutput(i_stdout)
  a_parsed = ParseOutput(a_stdout)
  return ParsedCombiner(i_parsed,a_parsed)




all_aligners = [AlignerLiu2013,AlignerOkada2015,AlignerGupta2012]

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--list',          action='store_true',                      help='List available aligners')
parser.add_argument('queryfile',       nargs='?' if '--list' in sys.argv else 1, help='an integer for the accumulator')
parser.add_argument('databasefile',    nargs='?' if '--list' in sys.argv else 1, help='an integer for the accumulator')
parser.add_argument('-a','--aligners', action='store', default=all_aligners,     help='Comma delimited list of aligners to use. If not specified, all aligners are used.')
args = parser.parse_args()

if args.list:
  for aligner in args.aligners:
    print(aligner.__name__)
  sys.exit(0)

df = pd.DataFrame()

for aligner in args.aligners:
  aligner_out = aligner(args.queryfile, args.databasefile, device=0)
  aligner_out["name"] = aligner.__name__
  df = df.append(aligner_out, ignore_index = True)

df = df.set_index("name")
df.save_csv("output.csv")

pyPaSWAS
========
[![DOI](https://zenodo.org/badge/28648467.svg)](https://zenodo.org/badge/latestdoi/28648467)

Extented python version of PaSWAS. Original papers in PLOS ONE: 

http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0122524

http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190279


For DNA/RNA/protein sequence alignment and trimming. 

PaSWAS was developed in C and CUDA/OpenCL. This version uses the CUDA/OpenCL code from PaSWAS and integrates the sequence alignment software with Python. It supports:
- text output
- SAM output
- logging
- command line options and configuration files
- Several internal programs: aligner (default) and trimmer 
- Affine gap penalty method

Platforms supported:
- NVIDIA GPU using CUDA (compute capability 1.3 and higher) 
- NVIDIA GPU using OpenCL
- Intel CPU using OpenCL
- Intel Xeon Phi accelerator using OpenCL
- Other systems supporting OpenCL (AMD, Intel GPUs, etc) should be able to run the software, but are untested.

More information: https://github.com/swarris/pyPaSWAS/wiki

Docker
------

The pyPasWAS source contains several docker install files. Clone the repository: 

git clone https://github.com/swarris/pyPaSWAS.git

Then use one of the docker images in the _docker_ folder. For more information, see the [README](https://github.com/swarris/pyPaSWAS/tree/master/docker) 

Installation
------------
In most cases it is enough to clone the repository. After that, please install:
- pip (https://docs.python.org/2.7/installing/)
- numpy: sudo pip install numpy (or pip install --user numpy)
- BioPython: sudo pip install Biopython (or pip install --user Biopython)
- In some cases, the python development packages are required (Ubuntu: sudo apt-get install python-dev) 

Making use of the CUDA version (also recommended when using the OpenCL version on a NVIDIA GPU):
- Download CUDA sdk: https://developer.nvidia.com/cuda-downloads
- sudo pip install pyCuda (http://mathema.tician.de/software/pycuda/)

Making use of the OpenCL version:
- check dependencies or downloads for your system. See this wiki for some great pointers: http://wiki.tiker.net/OpenCLHowTo
- sudo pip install pyOpenCL

Getting pyPaSWAS:
- Clone the repository: git clone https://github.com/swarris/pyPaSWAS.git
- Or download a pre-packaged release: https://github.com/swarris/pyPaSWAS/releases

Notes on Mac OS X
--------------
Installing all the necessary packages on a Mac can be difficult:
- When confronted with the message that the 'intern' module is missing when running pyPaSWAS using openCL, the 'six' package is not installed correctly. This can be solved by upgrading it to the most recent version: 'sudo pip install --upgrade six'. In some cases this will not help.  
- Although Apple has native support for OpenCL, pyOpenCL might not run with this compiler. Please download the Intel drivers and compiler.
- NVIDIA CUDA through pyCuda is the best choice when you have a NVIDIA GPU in your Mac. 

Running the software
-------------------- 

The key command line options are given in Table 1. The two input files are mandatory. Through the options the user can specify the file types of the input files (default: fasta), an output file and a log file. When requested, PyPaSWAS will terminate if the output file already exists.

Run it by calling:
- *python pypaswas.py |options| file1 file2*

Help file:
- *python pypaswas.py --help*

Selection your device
---------------------
By default, pypaswas will use the first CPU device. This can be changed by using:
- *--device_type=[CPU|GPU]*
- *--platform_name=[Intel|NVIDIA]*
- *--framework=[opencl|CUDA]*
- *--device=[int]*

For example, this will select the CPU: --device_type=CPU --platform_name=Intel --framework=opencl

This will select the second NVIDIA GPU: --device_type=GPU --platform_name=NVIDIA --framework=CUDA --device=1


Examples
--------
See the github wiki at https://github.com/swarris/pyPaSWAS/wiki for more examples.

Use a fastq-file:
- *python pypaswas.py testSample.fastq adapters.fa -1 fastq -o out.txt --loglevel=DEBUG*

Output results in SAM format:
- *python pypaswas.py testSample.fastq adapters.fa -1 fastq -o out.sam --outputformat=SAM --loglevel=DEBUG*

Remove all matches from file 1. Useful from trimming sequences. Sequences with no hits will not be in output
- *python pypaswas.py testSample.fastq adapters.fa -1 fastq -o out.fa --outputformat=trimmedFasta -p trimmer -L /tmp/log.txt --loglevel=DEBUG*

Align protein sequences:
- *python pypaswas.py myAA.faa product.faa -M BLOSUM62 -o hits.txt -L /tmp/log.txt --loglevel=DEBUG*



Table 1. Key command line options

| Option	| Long version	| Description|
| --------- | ------------- | ---------- |
| -h| --help| This help|  
|-L	| --logfile	| Path to the log file| 
|	| --loglevel	| Specify the log level for log file output. Valid options are DEBUG, INFO, WARNING, ERROR and CRITICAL| 
|-o	| --output	| Path to the output file. Default ./output| 
|-O	| --overrideOutput	| When output file exists, override it (T/F). Default T (true) | 
|-1	| --filetype1	| File type of the first input file. See bioPython IO for available options. Default fasta| 
|-2	| --filetype2	| File type of the second input file. See bioPython IO for available options. Default fasta| 
|-G	| 	| Float value for the gap penalty. Default -5| 
|-g |   | Float value for the gap extension penalty. Memory usage will increase when setting this to anything other than zero.  Default 0 | 
|-q	| 	| Float value for a mismatch. Default -3| 
|-r	| 	| Float value for a match. Default 1| 
|	| --any	| Float value for an ambiguous nucleotide. Default 1| 
|	| --other	| Float value for an ambiguous nucleotide. Default 1| 
|	| --device	| Integer value indicating the device to use. Default 0 for the first device. | 
|-c	| 	| Option followed by the name of a configuration file. This option allows for short command line calls and administration of used settings. | 

For questions, e-mail s.warris@gmail.com

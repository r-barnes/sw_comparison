# MASA-Serial
<p align="justify">
The <b>MASA-Serial extension</b> is used with the <a href="https://github.com/edanssandes/MASA-Core">MASA architecture</a> to align DNA sequences of unrestricted size with the Smith-Waterman algorithm combined with Myers-Miller. 
This is a simple template that can be used as baseline of other extensions.
</p>

### Download

Latest Version: [masa-serial-1.0.1.1024.tar.gz](releases/masa-serial-1.0.1.1024.tar.gz?raw=true)

### Compiling (masa-serial)

```
tar -xvzf masa-serial-1.0.1.1024.tar.gz
cd masa-serial-1.0.1.1024
./configure
make
```


### Executing a MASA extension

```
./masa-serial [options] seq1.fasta seq2.fasta
```
All the command line arguments can be retrieved using the --help parameter. See the [wiki](https://github.com/edanssandes/MASA-Core/wiki/Command-line-examples) for a list of command line examples.


### License

MASA-Serial is an open source project with public license (GPLv3). A copy of the LICENSE is maintained in this repository.

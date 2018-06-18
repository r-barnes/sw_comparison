
/**
Raccolta di funzioni di vario uso e utilità. 
Ognuna viene spiegata singolarmente in basso
*/

#ifndef INOUT_H
#define INOUT_H

#include <string>
#include <stdlib.h>
#include <vector>
#include <map>
#include <functional>
#include <QtCore/QFile>

const unsigned MAXLIBSIZE = 2*256*1024*1024;
const unsigned MAXSTRUCTSIZE = MAXLIBSIZE / 4096;

/**
stabilisce il massimo numero allocabile di oggetti dbLengthIdx, che devono essere uno per sequenza del DB
*/
const unsigned MAX_DB_STRUCT_SIZE = 900000;

typedef std::map<std::string, int> SubstitutionMatrixMap;

struct SubstitutionMatrixArray {
	int values[255][255];
};

/// struttura usata nella lettura del DB per raccogliere le info sulle sequenze
struct dbLengthIdx {
	unsigned length;
	unsigned lengthWithPad;
	unsigned idx;
	unsigned offsets;
};

/// struttura usata nella lettura del DB per l'ordinamento delle sequenze
struct Eless : public std::binary_function<dbLengthIdx, dbLengthIdx, bool> {
	bool operator()(dbLengthIdx x, dbLengthIdx y) { return x.length < y.length; }
};

/// funzione che blinda il codice all'indirizzo MAC della macchina.
void verifyMAC();

void usage();

/// funzione per la gestione delle opzioni da riga di commando
/**
Legge le opzioni da riga di comando e restituisce in seqFile e libFile rispettivamente il nome del file delle query e del DB.
*/
unsigned commandLineManager(int argc, char** argv, std::string &seqFile, std::string &libFile);

/// funzione per la verifica del formato FASTA per il file il cui nome è passato come argomento
void fastaVerifier(std::string &fileName);

/// funzione per la lettura del DB e per il suo caricamento in memoria (non è usata)
unsigned readdb(std::string searched, const std::string &searchedName, const std::string &fname, char *seqlib, unsigned seqLibSize, unsigned *offsets, unsigned *sizes, std::vector<std::string> &seqNames );

/// funzione per la lettura ordinata del DB e per il suo caricamento in memoria
/**
const std::string &fname - nome del file in cui si trova il DB
char *seqlib - area di memoria che conterrà il DB
unsigned seqLibSize - max dimensione di tale area di memoria
unsigned *offsets - offset dell' i-esima sequenza in seqlib
unsigned *sizes - size dell'i-esima sequenza in seqlib
std::vector<std::string> &seqNames - nomi delle sequenze nella seqlib
std::vector<std::string> &seqNamesNotOrd - nomi ordinati per dimensione delle sequenze
unsigned *sizesWithPad - dimensione delle sequenze con padding a 65.
returna il numero di sequenze lette
*/
unsigned readdbOrdered( const std::string &fname, char *seqlib, unsigned seqLibSize, unsigned *offsets, unsigned *sizes, std::vector<std::string> &seqNames, std::vector<std::string> &seqNamesNotOrd, unsigned *sizesWithPad );

///funzioni non usate ma tenute per eventuali futuri utilizzi
void readSeq(const std::string &fname, std::string &seq, std::string &seqName);
void readSubstitutionMatrix(const std::string &fname, SubstitutionMatrixMap &mat);
void readSubstitutionMatrix( SubstitutionMatrixMap &mat);
void readSubstitutionMatrix( SubstitutionMatrixArray &mat);

#endif

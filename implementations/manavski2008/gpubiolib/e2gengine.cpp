
#include "e2gengine.h"

#include "e2g_consts.h"

#include <QtCore/QTime>

#include <iostream>
using namespace std;

int savedPairCmp( const void *a, const void *b )
{
	const EstPSavePair A = (const EstPSavePair)a;
	const EstPSavePair B = (const EstPSavePair)b;
	
	int n = A->col - B->col;
	if (n != 0)
		return n;
	else
		return (A->row - B->row);

}


///------begin----E2GBacktrackBuffer------------------------------------------------------------
E2GBacktrackBuffer::E2GBacktrackBuffer() : static_rpair(NULL), static_rpairs(0), static_rpair_size(0), static_rpairs_sorted(false)
{
	megabytes = 8;

	pairInit(megabytes*1024*1024);
}

E2GBacktrackBuffer::~E2GBacktrackBuffer()  {
	pairFree();
}

void E2GBacktrackBuffer::pairFree(void) {

	if(static_rpair)
		delete [] static_rpair;

	static_rpair = NULL;
	static_rpair_size = 0;
	static_rpairs = 0;
	static_rpairs_sorted = false;

}

void E2GBacktrackBuffer::memReset() {
	//cerr << "static_rpairs: " << static_rpairs << ", size: " << static_rpair_size << endl;
	
	static_rpairs = 0;
	static_rpairs_sorted = false;
}


void E2GBacktrackBuffer::pairInit(unsigned int max_bytes) {
	int maxpairs = max_bytes/sizeof(EstSSavePair);

	static_rpair_size = maxpairs;
	static_rpair = new EstSSavePair[static_rpair_size];

}


void E2GBacktrackBuffer::doNotForget( int col, int row )
{
	if( static_rpairs >= static_rpair_size ) {

		throw string("Memory limit exceeded in allocating space for rpairs");

	}
	static_rpair[static_rpairs].col = col;
	static_rpair[static_rpairs].row = row;

	static_rpairs++;
	static_rpairs_sorted = false;

}



/* ****************************************************************************
**
** Recall saved pair values for row and column
**
** @param [r] col [ajint] Current column
** @param [r] row [ajint] Current row
**
** @return Row number
******************************************************************************/
int E2GBacktrackBuffer::pairRemember( int col, int row )
{
	EstSSavePair rp;
	int left;
	int right;
	int middle;
	int d;
	int bad;

	if (!static_rpairs_sorted) {
		qsort(static_rpair, static_rpairs, sizeof(EstSSavePair), savedPairCmp);
		static_rpairs_sorted = true;
	}

	rp.col = col;
	rp.row = row;

	left = 0;
	right = static_rpairs-1;


    /*
	** MATHOG, changed flow somewhat, added "bad" variable, inverted some
	** tests ( PLEASE CHECK THEM!  I did this because the original version
	** had two ways to drop into the failure code, but one of them was only
	** evident after careful reading of the code.
	*/

	if((savedPairCmp(&static_rpair[left],&rp ) > 0 ) || (savedPairCmp(&static_rpair[right],&rp ) < 0 ))
		bad = 1;      /*MATHOG, preceding test's logic was inverted */
	else {
		bad = 0;
		while(right-left > 1) {
			/* binary check for row/col */
			middle = (left+right)/2;
			d = savedPairCmp( &static_rpair[middle], &rp );
			if( d < 0 )
				left = middle;
			else if( d >= 0 )
				right = middle;
		}

		/*
		** any of these fail indicates failure
		** MATHOG, next test's logic was inverted
		*/
		if ( savedPairCmp( &static_rpair[right], &rp ) < 0 || static_rpair[left].col != col || static_rpair[left].row >= row ) {
			bad = 2;
		}
	}

	/* error - this should never happen  */
	if(bad != 0)
		throw string("ERROR in estPairRemember()");

	return static_rpair[left].row;
}

///------end----E2GBacktrackBuffer------------------------------------------------------------


E2GEngine::E2GEngine()
{
}

E2GEngine::E2GEngine(int match, int mismatch, int gap, int neutral, char pad_char)
{
	matInit(match, mismatch, gap, neutral, pad_char);
}


E2GEngine::~E2GEngine()
{
}



/*************************************************************************
**
** Modified Smith-Waterman/Needleman to align an EST or mRNA to a Genomic
** sequence, allowing for introns.
******************************************************************************/
void E2GEngine::alignNonRecursiveBackTrack(EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, E2GBacktrackBuffer *btbuf)
{
	const char* splice_sites_str;
	unsigned char *path   = NULL;
	int *score1;
	int *score2;
	int *s1;
	int *s2;
	int *s3;
	int *best_intron_score;
	int gpos;
	int epos;
	int glen;
	int elen;
	int diagonal;
	int delete_genome;
	int delete_est;
	int intron;
	const char *gseq;
	const char *eseq;
	char g;
	int max;
	int is_acceptor;

	EstOCoord best_start;

	unsigned char diagonal_path[4]      = { 1, 4, 16, 64 };
	unsigned char delete_est_path[4]    = { 2, 8, 32, 128 };
	unsigned char delete_genome_path[4] = { 3, 12, 48, 192 };

    /*
     ** path is encoded as 2 bits per cell:
	**
	** 00 intron
	** 10 diagonal
	** 01 delete_est
	** 11 delete_genome
	** the backtrack path, packed 4 cells per byte
	*/

	best_start.right = 0;
	best_start.left  = 0;

	glen = pair->getSubject()->getSize();
	elen = pair->getQuery()->getSize();
	
	estFindSpliceSites(*pair);
	splice_sites_str = pair->splice_sites.c_str();
			
	btbuf->memReset();
	pair->allocPaths();

	score1 = new int[elen+1];
	fill(score1, score1+elen+1, 0);
	
	score2 = new int[elen+1];
	fill(score2, score2+elen+1, 0);

	s1 = score1+1;
	s2 = score2+1;

	best_intron_score = new int[elen+1];
	fill(best_intron_score, best_intron_score+elen+1, 0);

	gseq = pair->getSubject()->sequenceData();
	eseq = pair->getQuery()->sequenceData();
		
	QTime timer2; timer2.start();

	for (gpos=0; gpos<glen; gpos++) { /* loop thru GENOME sequence */
		// s1 <---> s2
		s3 = s1;
		s1 = s2;
		s2 = s3;

		g = gseq[gpos];

		path = pair->ppath[gpos];

		if( splice_sites_str[gpos] & ACCEPTOR )
			is_acceptor = 1; /* gpos is last base of putative intron */
		else
			is_acceptor = 0;

		/* initialisation */
		s1[-1] = 0;
		
		for (epos=0; epos<elen; epos++) { /* loop thru EST sequence */
		
			unsigned char cur_path_value;
			
			/* align est and genome */
			diagonal = s2[epos-1] + ali_lsimmat[(int)g][(int)eseq[epos]];

			/* single deletion in est */
			delete_est = s1[epos-1] - gap_penalty;

			/* single deletion in genome */
			delete_genome = s2[epos] - gap_penalty;

			if (delete_est >= delete_genome) {
				max = delete_est;
				cur_path_value = delete_est_path[epos%4];
			} else {
				max = delete_genome;
				cur_path_value = delete_genome_path[epos%4];
			}
			
			if (diagonal >= max) {
				max = diagonal;
				cur_path_value = diagonal_path[epos%4];
			}

			if (max < 0 )
				max = 0;
			
			/* intron in genome, possibly modified by
			** donor-acceptor splice sites
			*/
			int bic_epos = pair->best_intron_coord[epos];
			if (is_acceptor && (splice_sites_str[bic_epos] & DONOR))
				intron = best_intron_score[epos] - splice_penalty;
			else
				intron = best_intron_score[epos] - intron_penalty;

			if (intron > max) {
				max = intron;
				cur_path_value = 0;
			}

			s1[epos] = max;
			
			if (max > 0) {
				path[epos/4] =  (unsigned char)( (unsigned int) path[epos/4] | (unsigned int) cur_path_value );
			}
			
			if ( best_intron_score[epos] < max ) {
				/* if( intron > 0 ) */ /* will only need to store if this path is positive */
				btbuf->doNotForget(epos, pair->best_intron_coord[epos]);
				best_intron_score[epos] = max;
				pair->best_intron_coord[epos] = gpos;
			}

			if (pair->max_score < max) {
				pair->max_score = max;
				pair->emax = epos;
				pair->gmax = gpos;
			}
		} // for thru the EST sequence
	} // for thru the GENOME sequence

	int elapsed2 = timer2.elapsed();
	//cerr << "est2genome timer2: " << elapsed2 << endl;

	delete []score1;
	delete []score2;

	delete []best_intron_score;
}


void E2GEngine::constructAlignmentResult(EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, E2GBacktrackBuffer *btbuf) {
	const char* splice_sites_str;
	int pos;
	int gpos;
	int epos;
	int glen;
	int elen;
	int total = 0;
	int splice_type = 0;
	unsigned char direction;
	int *temp_path = NULL;
	const char *gseq;
	const char *eseq;

	unsigned char mask[4] = { 3, 12, 48, 192 };

	
	EstResultDetailed *ge_result = &(pair->alignmentResult);
	
	splice_sites_str = pair->splice_sites.c_str();
	
	glen = pair->getSubject()->getSize();
	elen = pair->getQuery()->getSize();
	
	temp_path = new int[glen+elen];
	fill(temp_path, temp_path+glen+elen, 0);
	
	/* back track */
	ge_result->subjectStop = pair->gmax;
	ge_result->queryStop = pair->emax;
	ge_result->score = pair->max_score;

	gseq = pair->getSubject()->sequenceData();
	eseq = pair->getQuery()->sequenceData();

	pos = 0;

	epos = ge_result->queryStop;
	gpos = ge_result->subjectStop;
	total = 0;

	/* determine the type of spliced intron (forward or reversed) */

	if( pair->isForward() )
		splice_type = FORWARD_SPLICED_INTRON;
	else 
		splice_type = REVERSE_SPLICED_INTRON;

	while (total < pair->max_score && epos >= 0 && gpos >= 0 ) {
		direction = (unsigned char) (((unsigned int)pair->ppath[gpos][epos/4] & (unsigned int)mask[epos%4] ) >> (2*(epos%4)));
		temp_path[pos++] = direction;
		if((unsigned int) direction == INTRON ) /* intron */
		{
			int gpos1;

			if( gpos - pair->best_intron_coord[epos]  <= 0 ) {
					//ajWarn("NEGATIVE intron gpos: %d %d\n", gpos, gpos-best_intron_coord[epos] );
				gpos1 = btbuf->pairRemember(epos, gpos );
			} else
				gpos1 = pair->best_intron_coord[epos];

			if(  (splice_sites_str[gpos] & ACCEPTOR ) && ( splice_sites_str[gpos1] & DONOR ) ) {
				total -= splice_penalty;
				temp_path[pos-1] = splice_type; 
				/* make note that this is a proper intron */
			} else {
				total -= intron_penalty;
			}
				
			temp_path[pos++] = gpos-gpos1; /* intron this far */
			gpos = gpos1;
				
		} else if((unsigned int) direction == DIAGONAL ) {
			/* diagonal */
			total += ali_lsimmat[(int)gseq[gpos]][(int)eseq[epos]];
			epos--;
			gpos--;
		} else if((unsigned int) direction == DELETE_EST ) {
			/* delete_est */
			total -= gap_penalty;
			epos--;
		} else {	
			/* delete_genome */
			total -= gap_penalty;
			gpos--;
		}
	}

	gpos++;
	epos++;

	ge_result->subjectStart = gpos;
	ge_result->queryStart = epos;
	ge_result->len    = pos;

		//AJCNEW(ge->align_path, ge->len);
	if (ge_result->len > 0) {
		ge_result->align_path.resize(ge_result->len);
		fill(ge_result->align_path.begin(), ge_result->align_path.end(), 0);
	}

	/* reverse the ge so it starts at the beginning of the sequences */
	int p;
	for (p=0; p < ge_result->len; p++) {
		if( temp_path[p] > INTRON ) /* can be INTRON or FORWARD_SPLICED_INTRON or REVERSE_SPLICED_INTRON */
			ge_result->align_path[pos-p-1] = temp_path[p];
		else {
			ge_result->align_path[pos-p-2] = temp_path[p];
			ge_result->align_path[pos-p-1] = temp_path[p+1];
			p++;
		}
	}

	delete []temp_path;

	pair->freePaths();
	
}


int E2GEngine::getMatrixValue(unsigned char a, unsigned char b) const
{
	return ali_lsimmat[a][b];
}

/***************************************************************************
**
** Finds all putative DONOR and ACCEPTOR splice sites in the genomic sequence.
**
** Returns a sequence object whose "dna" should be interpreted as an
** array indicating what kind (if any) of splice site can be found at
** each sequence position.
**
**     DONOR    sites are NNGTNN last position in exon
**
**     ACCEPTOR sites are NAGN last position in intron
**
**     if forward==1 then search fot GT/AG
**     else               search for CT/AC
**
**
** @return Sequence of bitmask codes for splice sites.
******************************************************************************/
void E2GEngine::estFindSpliceSites(EstAlignPair &pair)
{
	unsigned genomelen = pair.getSubject()->getSize();
	const char *s = pair.getSubject()->sequenceData();
	
	char *p_res_splice_sites = new char[genomelen];
	fill(p_res_splice_sites, p_res_splice_sites+genomelen, NOT_A_SITE);

	unsigned pos;
	
	if( pair.isForward() ) {
		
		/* gene is in forward direction -splice consensus is gt/ag */
		for (pos=1; pos<genomelen-2; pos++) {
			/* last position in exon */
			if ( (s[pos] == 'g' || s[pos] == 'G') && (s[pos+1] == 't' || s[pos+1] == 'T') ) /* donor */
				p_res_splice_sites[pos-1] =  (unsigned char) NOT_A_SITE | (unsigned char) DONOR;

			/* last position in intron */
			if ( (s[pos] == 'a' || s[pos] == 'A') && (s[pos+1] == 'g' || s[pos+1] == 'G') ) /* acceptor */
				p_res_splice_sites[pos+1] = (unsigned char) NOT_A_SITE |  (unsigned char) ACCEPTOR;
		}
	
	} else {
		
		/* gene is on reverse strand so splice consensus looks like ct/ac */
		for (pos=1; pos<genomelen-2; pos++) {
			/* last position in exon */
			if ( (s[pos] == 'c' || s[pos] == 'C') && (s[pos+1] == 't' || s[pos+1] == 'T') ) /* donor */
				p_res_splice_sites[pos-1] = (unsigned char) NOT_A_SITE |  (unsigned char) DONOR;

			/* last position in intron */
			if ( (s[pos] == 'a' || s[pos] == 'A') && (s[pos+1] == 'c' || s[pos+1] == 'C') ) /* acceptor */
				p_res_splice_sites[pos+1] = (unsigned char) NOT_A_SITE |  (unsigned char) ACCEPTOR;
		}
		
	}
	p_res_splice_sites[pos+1] = 0;
	pair.splice_sites = p_res_splice_sites;
	
	delete []p_res_splice_sites;

}

/* @func matInit ********************************************************
**
** Comparison matrix initialisation.
**
** @param [r] match [ajint] Match code
** @param [r] mismatch [ajint] Mismatch penalty
** @param [r] gap [ajint] Gap penalty
** @param [r] neutral [ajint] Score for ambiguous base positions.
** @param [r] pad_char [char] Pad character for gaps in input sequences
**
******************************************************************************/
void E2GEngine::matInit(int match, int mismatch, int gap, int neutral, char pad_char)
{
	int c1;
	int c2;

	for (c1=0; c1<256; c1++)
		for (c2=0; c2<256; c2++) {
		if( c1 == c2 ) {
			if( c1 != '*' && c1 != 'n' &&  c1 != 'N' && c1 != '-' )
				ali_lsimmat[c1][c2] = match;
			else
				ali_lsimmat[c1][c2] = 0;
		} else {
			if( c1 == pad_char || c2 == pad_char )
				ali_lsimmat[c1][c2] = ali_lsimmat[c2][c1] = -gap;
			else if( c1 == 'n' || c2 == 'n' || c1 == 'N' || c2 == 'N' )
				ali_lsimmat[c1][c2] = ali_lsimmat[c2][c1] = neutral;
			else
				ali_lsimmat[c1][c2] = ali_lsimmat[c2][c1] = -mismatch;
		}
		}

		for (c1=0; c1<256; c1++) {
			c2 = tolower(c1);
			ali_lsimmat[c1][c2] = ali_lsimmat[c1][c1];
			ali_lsimmat[c2][c1] = ali_lsimmat[c1][c1];
		}

}



///-----Est2GenomePrinter------------------------------------------------------------------------------------------------------------------------

Est2GenomePrinter::Est2GenomePrinter(const string &stroutfile, bool gffstyle) : isOpen(false), GFFOutputStyle(gffstyle), counter(1) {

	char buf[255]; strcpy(buf, stroutfile.c_str());

	outfile = fopen(buf, "w+");
	if (!outfile) {
		string tst("cannot open output file");
		throw tst+buf;
	}
	isOpen = true;
}

Est2GenomePrinter::~Est2GenomePrinter() 
{
	if (isOpen)
		fclose(outfile);
}

bool Est2GenomePrinter::est2genomeOutputGFF(const AlignPair &apair, const EstResultSummary &result, unsigned genStartPos, unsigned genStopPos)
{
	int gsub;
	int gpos;
	int esub;
	int epos;
	int goff;
	int eoff;

	if (!GFFOutputStyle) return false;

	char direction = '+';

	const BioSequence *genome = apair.getSubject();
	const BioSequence *est = apair.getQuery();
			
	gsub = gpos = result.subjectStart;
	esub = epos = result.queryStart;
	goff = 0; //genome->getOffset();
	eoff = 0; //est->getOffset();

	EstResultSummary rsum = result;
	if( outfile && (rsum.numGaps + rsum.numMismatches) < 4) {
		char buf[200];
		sprintf(buf, "%d", counter++);
		string matchid = genome->getName() + ":GEAGpu:"; matchid += est->getName(); matchid += ":"; matchid += buf;
		
		if(!result.forward) {
			direction = '-';
		}

		unsigned N = rsum.exons.size();

		unsigned totStart = genStartPos+goff+result.subjectStart+1;
		unsigned totStop = genStartPos+goff+result.subjectStop+1;

		fprintf( outfile, "\n%-12s\tGEAGpu\tmatch\t\t%9u\t%9u\t%5d\t%c\t.\tID=\"%s\";NAME=\"%s\";Note=\"SC:%d;G:%d;M:%d;RI:%u\"\n", genome->getName().c_str(), totStart, totStop, result.score, direction, matchid.c_str(), est->getName().c_str(), result.score, rsum.numGaps, rsum.numMismatches, N-1  );
		
		for (unsigned j=0; j<N; ++j) {
				totStart = genStartPos+rsum.exons[j].first;
				totStop = genStartPos+rsum.exons[j].second;

				fprintf( outfile, "%-12s\tGEAGpu\tmatch-part\t%9u\t%9u\t.\t%c\t.\tParent=\"%s\";Target=%s\n", genome->getName().c_str(), totStart, totStop, direction, matchid.c_str(), est->getName().c_str());

		}
		return true;
	}
	return false;
}

bool Est2GenomePrinter::est2genomeMakeOutput(Est2GenomePrinter::TITLE title, EstAlignPair *pair, unsigned genStartPos, unsigned genStopPos, E2GEngine *e2gEngine, int gap_penalty, int intron_penalty, int splice_penalty, int minscore, unsigned minscore_nointron, bool align, int width)
{
	QMutexLocker mlock(&m_makeOutput);
	
	if(pair->max_score >= minscore) {
		
		if (!GFFOutputStyle) {
			fprintf( outfile, "####%s %u %u query size: %u\n", pair->getSubject()->getName().c_str(), genStartPos, genStopPos, pair->getQuery()->getSize());
			if (title == FEFGRG)
				fprintf( outfile, "Note Best alignment is between forward est and forward genome, but splice sites imply REVERSED GENE\n");
			else if (title == FEFGFG)
				fprintf( outfile, "Note Best alignment is between forward est and forward genome, and splice sites imply forward gene\n");
			else if (title == REFGRG)
				fprintf( outfile, "Note Best alignment is between reversed est and forward genome, but splice sites imply REVERSED GENE\n");
			else if (title == REFGFG)
				fprintf( outfile, "Note Best alignment is between reversed est and forward genome, and splice sites imply forward gene\n");
			else if (title == FEFG)
				fprintf( outfile, "Note Best alignment is between forward est and forward genome\n");
		}
		
		bool bintrons = isThereIntron(&(pair->alignmentResult));
		if ( bintrons || (!bintrons && pair->max_score >=(int)minscore_nointron) ) {
			
			if (GFFOutputStyle) {
				return outGFFStyle(e2gEngine, pair, genStartPos, genStopPos, gap_penalty, intron_penalty, splice_penalty );
			} else {
				outBlastStyle(e2gEngine, pair, gap_penalty, intron_penalty, splice_penalty, 1, !pair->isForward() );
				fprintf( outfile, "\n");
				outBlastStyle(e2gEngine, pair, gap_penalty, intron_penalty, splice_penalty, 0, !pair->isForward());
			}
			if (align) {
				fprintf(outfile, "\n\n%s vs %s:\n", pair->getSubject()->getName().c_str(), pair->getQuery()->getName().c_str());
				printAlign(e2gEngine, pair, width);
			}
			return true;
			
		}
		
	}
	return false;
	
}

///-----Est2GenomePrinter---PRIVATE-----------------------------------------------------------------------------------------

bool Est2GenomePrinter::isThereIntron(const EstResultDetailed *ge) {
	unsigned len = ge->len;
	for (unsigned p=0; p<len; p++) {
		if(ge->align_path[p] <= INTRON) return true;
	}
	return false;
}


/* **************************************************************************
**
** Print the alignment
**
** @param [u] ofile [AjPFile] Output file
** @param [r] genome [const E2GSeq] Genomic sequence
** @param [r] est [const E2GSeq] EST sequence
** @param [r] ge [const EmbPEstAlign] Genomic EST alignment
** @param [r] width [ajint] Output width (in bases)
**
******************************************************************************/
void Est2GenomePrinter::printAlign(E2GEngine *e2gEngine,  EstAlignPair *pair, int width )
{
	int gpos;
	int epos;
	int pos;
	int len;
	int i;
	int j;
	int max;
	int m;
	char *gbuf;
	char *ebuf;
	char *sbuf;
	const char *genomeseq;
	const char *estseq;

	int *gcoord;
	int *ecoord;
	int namelen;

	char format[256];

	genomeseq = pair->getSubject()->sequenceData();
	estseq = pair->getQuery()->sequenceData();

	namelen = (pair->getSubject()->getName().size() > pair->getQuery()->getName().size()) ? pair->getSubject()->getName().size() : pair->getQuery()->getName().size() ;

	sprintf(format, "%%%ds %%6d ", namelen );
	if(outfile) {
		fprintf(outfile, "\n");
		len = pair->getSubject()->getSize() + pair->getQuery()->getSize() + 1;

		//AJCNEW(gbuf,len);
		gbuf = new char [len];
		
		//AJCNEW(ebuf,len);
		ebuf = new char [len];
		//AJCNEW(sbuf,len);
		sbuf = new char [len];

		//AJCNEW(gcoord,len);
		gcoord = new int [len];
		//AJCNEW(ecoord,len);
		ecoord = new int [len];

		gpos = pair->alignmentResult.subjectStart;
		epos = pair->alignmentResult.queryStart;
		len = 0;
		for (pos=0; pos<pair->alignmentResult.len; pos++) {
			
			int way = pair->alignmentResult.align_path[pos];
			if( way == DIAGONAL  ) {
				
				/* diagonal */
				gcoord[len] = gpos;
				ecoord[len] = epos;
				gbuf[len] = toupper(genomeseq[gpos++]);
				ebuf[len] = toupper(estseq[epos++]);
				m = e2gEngine->getMatrixValue((int)gbuf[len], (int)ebuf[len]);

		/*
				** MATHOG, the triple form promotes char to
				** arithmetic type, which
				** generates warnings as it might be able
				** to overflow the char type.  This is
				** equivalent but doesn't trigger any compiler noise
				** sbuf[len] = (char) ( m > 0 ? '|' : ' ' );
		*/

				if(m>0)
					sbuf[len] = '|';
				else
					sbuf[len] = ' ';

				len++;
			} else if(way == DELETE_EST) {
				gcoord[len] = gpos;
				ecoord[len] = epos;
				gbuf[len] = '-';
				ebuf[len] = toupper(estseq[epos++]);
				sbuf[len] = ' ';
				len++;
			} else if( way == DELETE_GENOME ) {
				gcoord[len] = gpos;
				ecoord[len] = epos;
				gbuf[len] = toupper(genomeseq[gpos++]);
				ebuf[len] = '-';
				sbuf[len] = ' ';
				len++;
			} else if( way <= INTRON ) {
		/*
				** want enough space to print the first 5 and last 5
				** bases of the intron, plus a string containing the
				** intron length
		*/
				int intron_width;
				int half_width;
				int g;
				char number[30];
				int numlen;

				intron_width = pair->alignmentResult.align_path[pos+1];
				g = gpos-1;
				half_width = intron_width > 10 ? 5 : intron_width/2;


				sprintf(number," %d ", intron_width );
				numlen = (int)strlen(number);

				for(j=len;j<len+half_width;j++) {
					g++;
					gcoord[j] = gpos-1;
					ecoord[j] = epos-1;
					gbuf[j] = genomeseq[g];
					ebuf[j] = '.';
					if(way == FORWARD_SPLICED_INTRON)
						sbuf[j] = '>';
					else if(way == REVERSE_SPLICED_INTRON)
						sbuf[j] = '<';
					else
						sbuf[j] = '?';
				}
				len = j;

				for(j=len;j<len+numlen;j++) {
					gcoord[j] = gpos-1;
					ecoord[j] = epos-1;
					gbuf[j] = '.';
					ebuf[j] = '.';
					sbuf[j] = number[j-len];
				}
				len = j;
				g = gpos + intron_width - half_width-1;

				for(j=len;j<len+half_width;j++) {
					g++;
					gcoord[j] = gpos-1;
					ecoord[j] = epos-1;
					gbuf[j] = genomeseq[g];
					ebuf[j] = '.';
					if(way == FORWARD_SPLICED_INTRON)
						sbuf[j] = '>';
					else if(way == REVERSE_SPLICED_INTRON)
						sbuf[j] = '<';
					else
						sbuf[j] = '?';
				}

				gpos += pair->alignmentResult.align_path[++pos];
				len = j;
			}
		}

		for(i=0;i<len;i+=width)
		{
			max = ( i+width > len ? len : i+width );

			fprintf(outfile, format, pair->getSubject()->getName().c_str(), gcoord[i]+1 );
			for(j=i;j<max;j++)
				fprintf(outfile, "%c",  gbuf[j]);
			fprintf(outfile," %6d\n", gcoord[j-1]+1 );

			for(j=0;j<namelen+8;j++)
				fprintf(outfile, " ");

			for(j=i;j<max;j++)
				fprintf(outfile,"%c", sbuf[j]);
			fprintf(outfile,  "\n");

			fprintf(outfile, format, pair->getQuery()->getName().c_str(), ecoord[i]+1 );
			for(j=i;j<max;j++)
				fprintf(outfile, "%c", ebuf[j]);
			fprintf(outfile," %6d\n\n", ecoord[j-1]+1 );
		}

		fprintf( outfile, "\nAlignment Score: %d\n", pair->max_score );

		delete [] gbuf;
		delete [] ebuf;
		delete [] sbuf;
		delete [] gcoord;
		delete [] ecoord;
	}

	return;
}



/*****************************************************************************
**
** write out the MSP (maximally scoring pair).
**
** @param [u] ofile [AjPFile] Output file
** @param [w] matches [ajint*] Number of matches found
** @param [w] len [ajint*] Length of alignment
** @param [w] tsub [ajint*] Score
** @param [r] genome [const E2GSeq] Genomic sequence
** @param [r] gsub [ajint] Genomic start position
** @param [r] gpos [ajint] Genomic end position
** @param [r] est [const E2GSeq] EST sequence
** @param [r] esub [ajint] EST start position
** @param [r] epos [ajint] EST end position
** @param [r] reverse [ajint] Boolean 1=reverse the EST sequence
** @param [r] gapped [ajint] Boolean 1=full gapped alignment
**                         0=display ungapped segment
**
******************************************************************************/
void Est2GenomePrinter::myestWriteMsp(int *matches, int *len, int *tsub, EstAlignPair *pair, int gsub, int gpos, int esub, int epos, int reverse, int gapped)
{
	float percent;
	int goff;
	int eoff;

	goff = 0; //genome->getOffset();
	eoff = 0; //est->getOffset();

	if( *len > 0 )
		percent = (*matches/(float)(*len))*(float)100.0;
	else
		percent = (float) 0.0;

	if( percent > 0 ) {
		if( gapped )
			fprintf( outfile, "Exon     " );
		else
			fprintf( outfile, "Segment  " );
		if( reverse )
			fprintf(outfile, "%5d %5.1f %5d %5d %-12s %5d %5d %-12s  %s\n",
					*tsub, percent, gsub+1, gpos, pair->getSubject()->getName().c_str(),
							eoff+pair->getQuery()->getSize()-esub, 
									eoff+pair->getQuery()->getSize()-epos+1,
											pair->getQuery()->getName().c_str(), "" ); //pair->query->getDescription().c_str()
		else
			fprintf(outfile, "%5d %5.1f %5d %5d %-12s %5d %5d %-12s  %s\n",
					*tsub, percent, goff+gsub+1, goff+gpos, pair->getSubject()->getName().c_str(),
							eoff+esub+1, eoff+epos, pair->getQuery()->getName().c_str(), "");
									//est->getDescription().c_str() );
	}

	*matches = *len = *tsub = 0;

}


/******************************************************************************
**
** output in blast style.
**
** @param [u] blast [AjPFile] Output file
** @param [r] genome [const E2GSeq] Genomic sequence
** @param [r] est [const E2GSeq] EST sequence
** @param [r] ge [const EmbPEstAlign] Genomic EST alignment
** @param [r] gap_penalty [ajint] Gap penalty
** @param [r] intron_penalty [ajint] Intron penalty
** @param [r] splice_penalty [ajint] Splice site penalty
** @param [r] gapped [ajint] Boolean. 1 = write a gapped alignment
** @param [r] reverse [ajint] Boolean. 1 = reverse alignment
**
** @return [void]
** @@
******************************************************************************/

void Est2GenomePrinter::outBlastStyle(E2GEngine *e2gEngine,  EstAlignPair *pair, int gap_penalty, int intron_penalty, int splice_penalty, int gapped, int reverse)
{

	int gsub;
	int gpos;
	int esub;
	int epos;
	int tsub;
	int p;
	int matches = 0;
	int len     = 0;
	int m;
	int total_matches = 0;
	int total_len     = 0;
	float percent;
	const char *genomestr;
	const char *eststr;
	int goff;
	int eoff;

	pair->getQuery()->toLower();
	BioSequence genome(*pair->getSubject());
	genome.toLower();
	
	genomestr = genome.sequenceData();
	eststr = pair->getQuery()->sequenceData();
	EstResultDetailed *ge = &pair->alignmentResult;
	
	gsub = gpos = ge->subjectStart;
	esub = epos = ge->queryStart;
	goff = 0; //genome->getOffset();
	eoff = 0; //est->getOffset();

	if( outfile ) {
		tsub = 0;
		for(p=0;p<ge->len;p++)
			if(ge->align_path[p] <= INTRON) {
				myestWriteMsp(&matches, &len, &tsub, pair, gsub, gpos, esub, epos, reverse, gapped);
				if (gapped) {
					if(ge->align_path[p] == INTRON) {
						fprintf(outfile, "?Intron  %5d %5.1f %5d %5d %-12s\n", -intron_penalty, (float) 0.0, goff+gpos+1, goff+gpos+ge->align_path[p+1], genome.getName().c_str());
					} else {	/* proper intron */
						if( ge->align_path[p] == FORWARD_SPLICED_INTRON )
							fprintf( outfile, "+Intron  %5d %5.1f %5d %5d %-12s\n", -splice_penalty, (float) 0.0, goff+gpos+1, goff+gpos+ge->align_path[p+1], genome.getName().c_str() );
						else
							fprintf( outfile, "-Intron  %5d %5.1f %5d %5d %-12s\n", -splice_penalty, (float) 0.0, goff+gpos+1, goff+gpos+ge->align_path[p+1], genome.getName().c_str() );
					}
				}

				gpos += ge->align_path[++p];
				esub = epos;
				gsub = gpos;
			
			} else if(ge->align_path[p] == DIAGONAL) {
				char gc = genomestr[gpos];
				m = e2gEngine->getMatrixValue((int)gc, (int)eststr[(int)epos]);
				tsub += m;
				if (m > 0) {
					matches++;
					total_matches++;
				}
				len++;
				total_len++;
				gpos++;
				epos++;
			
			} else if(ge->align_path[p] == DELETE_EST) {
				if (gapped) {
					tsub -= gap_penalty;
					epos++;
					len++;
					total_len++;
				} else {
					myestWriteMsp( &matches, &len, &tsub, pair, gsub, gpos, esub, epos, reverse, gapped);
					epos++;
					esub = epos;
					gsub = gpos;
				}
				
			} else if (ge->align_path[(int)p] == DELETE_GENOME) {
				if (gapped) {
					tsub -= gap_penalty;
					gpos++;
					total_len++;
					len++;
				} else {
					myestWriteMsp(&matches, &len, &tsub, pair, gsub, gpos, esub, epos, reverse, gapped);
					gpos++;
					esub = epos;
					gsub = gpos;
				}
			}
			
			myestWriteMsp(&matches, &len, &tsub, pair, gsub, gpos, esub, epos, reverse, gapped);

			if(gapped) {
				if(total_len > 0)
					percent = (total_matches/(float)(total_len))*(float)100.0;
				else
					percent = (float) 0.0;

				if(reverse)
					fprintf( outfile, "\nSpan     %5d %5.1f %5d %5d %-12s %5d %5d %-12s  %s\n", ge->score, percent, goff+ge->subjectStart+1, goff+ge->subjectStop+1, genome.getName().c_str(), eoff+pair->getQuery()->getSize()-ge->queryStart, eoff+pair->getQuery()->getSize()-ge->queryStop, pair->getQuery()->getName().c_str(), ""); //est->getDescription().c_str() );
				else
					fprintf( outfile, "\nSpan     %5d %5.1f %5d %5d %-12s %5d %5d %-12s  %s\n", ge->score, percent, goff+ge->subjectStart+1, goff+ge->subjectStop+1,genome.getName().c_str(), eoff+ge->queryStart+1, eoff+ge->queryStop+1, pair->getQuery()->getName().c_str(), "");
				//est->getDescription().c_str() );
			}

	}

	return;
}



/* @func outGFFStyle **************************************************
**
** output in GFF style.
**
******************************************************************************/
bool Est2GenomePrinter::outGFFStyle(E2GEngine *e2gEngine,  EstAlignPair *pair, unsigned genStartPos, unsigned genStopPos, int gap_penalty, int intron_penalty, int splice_penalty)
{
	EstResultSummary summary = pair->makeSummary(); 

	return est2genomeOutputGFF(*pair, summary, genStartPos, genStopPos);
}





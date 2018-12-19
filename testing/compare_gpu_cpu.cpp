#include <swsharp/swsharp.h>
#include <swsharp/evalue.h>
#include <ssw.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>

typedef std::vector<std::string> seqs_t;

class SSWAlign {
 private:
   const int match           = 1;
   const int mismatch        = 3;
   const int gap_open        = 5;
   const int gap_extension   = 2;
   const int ambiguity_score = 2;
   int8_t mat[25];  

   /* This table is used to transform nucleotide letters into numbers. */
   const int8_t nt_table[128] = {
       4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
       4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
       4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
       4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
     //   A     C            G
       4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
     //             T
       4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
     //   a     c            g
       4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
     //             t
       4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
   };   
 public:
  SSWAlign(){
    int k=0;
    for (int l=0; l < 4; ++l) {
      for (int m = 0; m < 4; ++m)
        mat[k++] = l == m ? match : - mismatch;   /* weight_match : -weight_mismatch */
      mat[k++] = -ambiguity_score; // ambiguous base: no penalty
    }
    for (int m = 0; m < 5; ++m)
      mat[k++] = -ambiguity_score;
  }

  void align(const std::string &query, const std::string &target){
    /* Use SSW to align as much as we can from this read (align from start of the read to allow misalignments at the beginning)*/
    int8_t *const query_num  = new int8_t[query.size() ];
    int8_t *const target_num = new int8_t[target.size()];

    for (unsigned int m = 0; m < query.size(); ++m)
      query_num[m] = (int8_t) nt_table[(int)query[m]];
    for (unsigned int m = 0; m < target.size(); ++m)
      target_num[m] = (int8_t) nt_table[(int)target[m]];

    const s_profile *const profile = ssw_init(query_num, query.size(), mat, 5, 2);

    const int min_len = (query.size()/2 >= 15) ? query.size() / 2 : 15;

    //TODO: CHeck to see if `target.size()` is correct thing for below
    const s_align *const r = ssw_align (profile, target_num, target.size(), gap_open, gap_extension, 8, 0, 0, min_len);

    delete[] query_num;
    delete[] target_num;
  }
};



class Okada2015 {
 private:
  Scorer* scorer;
  int cards[20];
  int cards_len;

 public:
  Okada2015(){
    // create a scorer object
    // match = 1
    // mismatch = -3
    // gap open = 5
    // gap extend = 2
    scorerCreateScalar(&scorer, 1, -3, 5, 2);    
    assert(scorer!=NULL);

    // use one CUDA card with index 0
    cards[0]  = 0;
    cards_len = 1;
  }

  ~Okada2015(){
    scorerDelete(scorer);
  }

  //TODO: has a mulitple-align option called `alignBest()`
  void align(const std::string &query, const std::string &target){
    const std::string query_name  = "query";
    const std::string target_name = "target";

    Chain *const queryc  = chainCreate(query_name.data(),  query_name.size(),  query.data(),  query.size() );
    Chain *const targetc = chainCreate(target_name.data(), target_name.size(), target.data(), target.size());
    
    // do the pairwise alignment, use Smith-Waterman algorithm
    Alignment* alignment;
    assert(scorer!=NULL);
    alignPair(&alignment, SW_ALIGN, queryc, targetc, scorer, cards, cards_len, NULL);
     
    // output the results in emboss stat-pair format
    //outputAlignment(alignment, NULL, SW_OUT_STAT_PAIR);
    
    alignmentDelete(alignment);

    chainDelete(queryc);
    chainDelete(targetc);
  }
};



class Okada2015Bulk {
 private:
  static void valueFunction(double* values, int* scores, Chain* query, 
    Chain** database, int databaseLen, int* cards, int cardsLen, void* param_ ) {
    
    EValueParams* eValueParams = (EValueParams*) param_;
    eValues(values, scores, query, database, databaseLen, cards, cardsLen, eValueParams);
  }

 public:
  int cards[20];
  int cards_len;
  const int gapOpen            = 10;
  const int gapExtend          = 1;
  const char* matrix           = "BLOSUM_62";
  Scorer* scorer               = NULL;
  ChainDatabase* chainDatabase = NULL;
  EValueParams*  eValueParams  = NULL;
  Chain**        database      = NULL;
  int            database_len  = 0;

  Okada2015Bulk(){
    scorerCreateMatrix(&scorer, matrix, gapOpen, gapExtend);
    assert(scorer!=NULL);
   
    // use one CUDA card with index 0
    cards[0]  = 0;
    cards_len = 1;
  }

  ~Okada2015Bulk(){
    scorerDelete(scorer);
    chainDatabaseDelete(chainDatabase);
    deleteEValueParams(eValueParams);
    deleteFastaChains(database, database_len);
  }

  void createDatabase(const seqs_t &dbvec){
    long long cells = 0;
    for(const auto &dstr: dbvec)
      cells += dstr.size();

    eValueParams = createEValueParams(cells, scorer);

    DbAlignment*** dbAlignments = NULL;
    int* dbAlignmentsLens       = NULL;

    const std::string db_chain_name = "no name";
    
    database     = new Chain*[dbvec.size()];
    database_len = dbvec.size();

    for(unsigned int i=0;i<dbvec.size();i++)
      database[i] = chainCreate(
        db_chain_name.data(),
        db_chain_name.size(),
        dbvec.at(i).data(),
        dbvec.at(i).size()
      );

    chainDatabase = chainDatabaseCreate(database, 0, dbvec.size(), cards, cards_len);
  }

  void align(const seqs_t &qvec){
    const int   maxAlignments = 10;
    const float maxEValue     = 10;

    const std::string q_chain_name = "no name";
    Chain**const queries = new Chain*[qvec.size()];
    for(unsigned int i=0;i<qvec.size();i++)
      queries[i] = chainCreate(q_chain_name.data(), q_chain_name.size(), qvec.at(i).data(), qvec.at(i).size());


    DbAlignment*** dbAlignments     = NULL;
    int*           dbAlignmentsLens = NULL;

    assert(chainDatabase!=NULL);
    shotgunDatabase(
      &dbAlignments, 
      &dbAlignmentsLens, 
      SW_ALIGN, 
      queries, 
      qvec.size(),
      chainDatabase,
      scorer,
      maxAlignments,
      Okada2015Bulk::valueFunction, 
      (void*) eValueParams,
      maxEValue,
      NULL,
      0, 
      cards,
      cards_len,
      NULL
    );

    // for (i = 0; i < queriesLen; ++i)
    // for (j = 0; j < dbAlignmentsLens[i]; ++j) {
    //   DbAlignment* dbAlignment = dbAlignments[i][j];
    //   int targetIdx = dbAlignmentGetTargetIdx(dbAlignment);

    //   usedMask[targetIdx] = 1;
    // }

    // outputShotgunDatabase(dbAlignments, dbAlignmentsLens, queriesLen, out, outFormat);
    deleteShotgunDatabase(dbAlignments, dbAlignmentsLens, qvec.size());

    deleteFastaChains(queries, qvec.size());
  }
};


std::string GenerateSequence(const int len){
  const char dna[5] = "ACTG";
  std::string temp;
  for(int i=0;i<len;i++)
    temp += dna[rand()%4];
  return temp;
}



std::vector<std::string> GenerateSequences(const int num, const int len){
  std::vector<std::string> temp;
  for(int i=0;i<num;i++)
    temp.push_back(GenerateSequence(len));
  return temp;
}



void MakeComparison(const int num_a, const int len_a, const int num_b, const int len_b){
  const seqs_t seqs_a = GenerateSequences(num_a, len_a);
  const seqs_t seqs_b = GenerateSequences(num_b, len_b);


  std::chrono::steady_clock::time_point ssw_begin = std::chrono::steady_clock::now();
  {
    SSWAlign  ssw;
   for(unsigned int i=0;i<seqs_a.size();i++)
   for(unsigned int j=0;j<seqs_b.size();j++)
     ssw.align(seqs_a[i], seqs_b[j]);
  }
  std::chrono::steady_clock::time_point ssw_end= std::chrono::steady_clock::now();

  std::chrono::steady_clock::time_point okada2015_begin = std::chrono::steady_clock::now();
  {
    Okada2015 okada2015;
   for(unsigned int i=0;i<seqs_a.size();i++)
   for(unsigned int j=0;j<seqs_b.size();j++)
     okada2015.align(seqs_a[i], seqs_b[j]);  
  }
  std::chrono::steady_clock::time_point okada2015_end= std::chrono::steady_clock::now();

  // std::chrono::steady_clock::time_point okada2015_bulk_begin = std::chrono::steady_clock::now();
  // {
  //   Okada2015Bulk okada2015_bulk;
  //   okada2015_bulk.createDatabase(seqs_b);
  //   okada2015_bulk.align(seqs_a);
  // }
  // std::chrono::steady_clock::time_point okada2015_bulk_end= std::chrono::steady_clock::now();



  std::cout <<"  num_a         = "<<std::setw(10) << num_a
            <<", len_a         = "<<std::setw(10) << len_a
            <<"  num_b         = "<<std::setw(10) << num_b
            <<", len_b         = "<<std::setw(10) << len_b            
            <<", SSW           = "<<std::setw(15) << std::chrono::duration_cast<std::chrono::milliseconds>(ssw_end - ssw_begin).count() << " ms"
            <<", Okada2015     = "<<std::setw(15) << std::chrono::duration_cast<std::chrono::milliseconds>(okada2015_end - okada2015_begin).count() << " ms"
            <<", Okada2015Bulk = "<<std::setw(15) << std::chrono::duration_cast<std::chrono::milliseconds>(okada2015_bulk_end - okada2015_bulk_begin).count() << " ms"
            <<std::endl;
}



int main(int argc, char **argv){
  std::vector<int> num_a_vals = {{10}};
  std::vector<int> len_a_vals = {{1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000}};
  std::vector<int> num_b_vals = {{10}};
  std::vector<int> len_b_vals = {{1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000}};

  for(const auto num_a: num_a_vals)
  for(const auto len_a: len_a_vals)
  for(const auto num_b: num_b_vals)
  for(const auto len_b: len_b_vals)
    MakeComparison(num_a, len_a, num_b, len_b);
}

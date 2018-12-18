#include <swsharp/swsharp.h>
#include <ssw.h>
#include <cstdlib>
#include <string>
#include <vector>

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
    Scorer* scorer;
    scorerCreateScalar(&scorer, 1, -3, 5, 2);    

    // use one CUDA card with index 0
    cards[0]  = 0;
    cards_len = 1;
  }

  ~Okada2015(){
    scorerDelete(scorer);
  }

  //TODO: has a mulitple-align option called `alignBest()`
  void align(const std::string &query, const std::string &target){
    Chain *const queryc  = chainCreate(NULL, 0,  query.data(),  query.size());
    Chain *const targetc = chainCreate(NULL, 0, target.data(), target.size());
    
    // do the pairwise alignment, use Smith-Waterman algorithm
    Alignment* alignment;
    alignPair(&alignment, SW_ALIGN, queryc, targetc, scorer, cards, cards_len, NULL);
     
    // output the results in emboss stat-pair format
    //outputAlignment(alignment, NULL, SW_OUT_STAT_PAIR);
    
    alignmentDelete(alignment);

    chainDelete(queryc);
    chainDelete(targetc);
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



void MakeComparison(const int num, const int len){
  const auto seqs = GenerateSequences(num, len);

  SSWAlign  ssw;
  Okada2015 okada2015;

  for(unsigned int i=0;i<seqs.size();i++)
  for(unsigned int j=i+1;j<seqs.size();j++)
    ssw.align(seqs[i], seqs[j]);

  for(unsigned int i=0;i<seqs.size();i++)
  for(unsigned int j=i+1;j<seqs.size();j++)
    okada2015.align(seqs[i], seqs[j]);  
}



int main(int argc, char **argv){
  MakeComparison(10,   1000); 
  MakeComparison(100,  1000);
  MakeComparison(10,  10000); 
  MakeComparison(10,  20000); 
}

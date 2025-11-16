#include "random_mdp.h"
#include <random>
#include <algorithm>

static int expected_nonzeros_per_action(Sparsity s, int S){
  switch(s){
    case Sparsity::VerySparse: return 2;   // ~1–2
    case Sparsity::Moderate:   return 8;   // ~5–10
    case Sparsity::Dense:      return std::max(1, S/2); // ~50%
  }
  return 2;
}

static void normalize(std::vector<double>& w){
  double z=0; for(double x:w) z+=x; if (z==0) return;
  for(double& x:w) x/=z;
}

void generate_random_mdp(const RandomMDPConfig& c, MDPCSR& out){
  std::mt19937 rng(c.seed);
  int S=c.S;
  bool varying = (c.A_fixed<=0);
  auto A_of = [&](int /*s*/){
    if (!varying) return c.A_fixed;
    std::uniform_int_distribution<int> u(c.A_min, c.A_max);
    return u(rng);
  };

  // First pass to determine total rows (sum A_s)
  std::vector<int> A_s(S);
  int total_rows=0, Amax=0;
  for(int s=0;s<S;++s){ A_s[s]=A_of(s); total_rows += A_s[s]; Amax=std::max(Amax,A_s[s]); }

  out = {};
  out.S=S; out.A=Amax; // keep A as the maximum for compatibility; rows = sum A_s
  CSR P, R; P.S=S; R.S=S; P.A=Amax; R.A=Amax;
  P.indptr.assign(total_rows+1,0);
  R.indptr.assign(total_rows+1,0);

  std::uniform_real_distribution<double> u01(0.,1.);
  std::uniform_int_distribution<int> uniS(0,S-1);

  // Build each row (s,a_local). We’ll map (s,a_local) to a global row index that packs varying actions.
  int row=0;
  std::vector<int> row_of_s_a;
  row_of_s_a.reserve(total_rows);

  // Count nnz and fill indptr
  int Pnnz=0, Rnnz=0;
  for(int s=0;s<S;++s){
    for(int aL=0;aL<A_s[s];++aL){
      int k = expected_nonzeros_per_action(c.sparsity, S);
      // sample k unique next states
      std::vector<int> cols; cols.reserve(k);
      while((int)cols.size()<k){ int j=uniS(rng); if (std::find(cols.begin(),cols.end(),j)==cols.end()) cols.push_back(j); }
      // random unnormalized probs
      std::vector<double> w(k); for(double& x:w) x=u01(rng)+1e-6; normalize(w);
      Pnnz += k;
      P.indptr[row+1]=Pnnz;

      // Rewards
      if (c.rtype==RType::UniformNeg1to1){
        Rnnz += k;
        R.indptr[row+1]=Rnnz;
      } else if (c.rtype==RType::Binary01 || c.rtype==RType::Binary0m1){
        Rnnz += k;
        R.indptr[row+1]=Rnnz;
      } else if (c.rtype==RType::PatternEveryK){
        // every Kth state gets +1, else 0
        for(int j:cols) if (j % std::max(1,c.pattern_k)==0) ++Rnnz;
        R.indptr[row+1]=Rnnz;
      }

      ++row;
    }
  }

  P.indices.resize(Pnnz); P.data.resize(Pnnz);
  R.indices.resize(Rnnz); R.data.resize(Rnnz);

  // Second pass: fill data
  row=0; int pi=0, ri=0;
  for(int s=0;s<S;++s){
    for(int aL=0;aL<A_s[s];++aL){
      int k = expected_nonzeros_per_action(c.sparsity, S);
      std::vector<int> cols; cols.reserve(k);
      while((int)cols.size()<k){ int j=uniS(rng); if (std::find(cols.begin(),cols.end(),j)==cols.end()) cols.push_back(j); }
      std::vector<double> w(k); for(double& x:w) x=u01(rng)+1e-6; normalize(w);

      for(int m=0;m<k;++m){ P.indices[pi]=cols[m]; P.data[pi]=w[m]; ++pi; }

      if (c.rtype==RType::UniformNeg1to1){
        std::uniform_real_distribution<double> ur(-1.0,1.0);
        for(int m=0;m<k;++m){ R.indices[ri]=cols[m]; R.data[ri]=ur(rng); ++ri; }
      } else if (c.rtype==RType::Binary01){
        for(int m=0;m<k;++m){ R.indices[ri]=cols[m]; R.data[ri]= (u01(rng)<0.5?0.0:1.0); ++ri; }
      } else if (c.rtype==RType::Binary0m1){
        for(int m=0;m<k;++m){ R.indices[ri]=cols[m]; R.data[ri]= (u01(rng)<0.5?0.0:-1.0); ++ri; }
      } else { // PatternEveryK
        for(int j:cols) if (j % std::max(1,c.pattern_k)==0){ R.indices[ri]=j; R.data[ri]=1.0; ++ri; }
      }
      ++row;
    }
  }

  out.P = std::move(P);
  out.R = std::move(R);
  out.gamma = 0.9;
}

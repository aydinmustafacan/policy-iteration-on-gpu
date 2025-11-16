#include "grid_gen.h"
#include <random>
#include <algorithm>

static inline int clamp(int x,int lo,int hi){ return std::max(lo,std::min(x,hi)); }

static int step_next(int i,int j,int a,int H,int W, Boundary b) {
  static const int di[4]={-1,1,0,0}, dj[4]={0,0,1,-1};
  int ni=i+di[a], nj=j+dj[a];
  bool off = (ni<0||ni>=H||nj<0||nj>=W);
  if (!off) return ni*W+nj;
  switch(b){
    case Boundary::Reflect: return i*W + j;
    case Boundary::Clamp:   return clamp(ni,0,H-1)*W + clamp(nj,0,W-1);
    case Boundary::Wrap:    return (( (ni%H)+H )%H)*W + (( (nj%W)+W )%W);
  }
  return i*W+j;
}

static void pick_goals(const GridConfig& c, std::vector<int>& goals, std::mt19937& rng){
  int S=c.H*c.W;
  if (c.goal_mode==GoalMode::Fixed){ goals={ (c.H-1)*c.W + (c.W-1) }; }
  else if (c.goal_mode==GoalMode::Random){
    std::uniform_int_distribution<int> uni(0,S-1);
    goals={ uni(rng) };
  } else { // Multi
    std::uniform_int_distribution<int> uni(0,S-1);
    goals.clear();
    for(int k=0;k<c.goals;k++) goals.push_back(uni(rng));
    std::sort(goals.begin(),goals.end());
    goals.erase(std::unique(goals.begin(),goals.end()),goals.end());
  }
}

void generate_gridworld_dense(const GridConfig& c, MDPDense& out){
  int H=c.H, W=c.W, S=H*W, A=4;
  out = {}; out.S=S; out.A=A; out.gamma=out.gamma?out.gamma:0.9;
  out.P.assign(S, std::vector<std::vector<double>>(A, std::vector<double>(S,0.0)));
  out.R.assign(S, std::vector<std::vector<double>>(A, std::vector<double>(S,0.0)));

  std::mt19937 rng(c.seed);
  // blocked walls
  std::vector<char> blocked(S,0);
  if (c.blocked_density>0){
    std::uniform_real_distribution<double> u(0.,1.);
    for(int s=0;s<S;++s) if (u(rng)<c.blocked_density) blocked[s]=1;
    blocked[0]=0; // avoid blocking trivial start
  }

  std::vector<int> goals; pick_goals(c, goals, rng);
  std::vector<char> is_goal(S,0); for(int g:goals) is_goal[g]=1;

  // obstacle (penalty) cells (enterable)
  std::vector<char> obstacle(S,0);
  if (c.obstacle_density>0){
    std::uniform_real_distribution<double> u(0.,1.);
    for(int s=0;s<S;++s) if(!is_goal[s] && u(rng)<c.obstacle_density) obstacle[s]=1;
  }

  auto add_prob = [&](int s,int a,int s2,double p){
    if (blocked[s]){ // if origin is blocked, treat as self-loop
      out.P[s][a][s]+=p;
    } else if (blocked[s2]){ // cannot enter wall → bounce back (reflect)
      out.P[s][a][s]+=p;
    } else {
      out.P[s][a][s2]+=p;
    }
  };

  for(int s=0;s<S;++s){
    int i=s/W, j=s%W;
    for(int a=0;a<A;++a){
      // Transition model
      if (c.trans==GridTrans::Deterministic){
        int s2=step_next(i,j,a,H,W,c.boundary);
        add_prob(s,a,s2,1.0);
      } else if (c.trans==GridTrans::Slip){
        double p_int=1.0 - c.slip;
        double p_slip=c.slip/2.0;
        int s_int = step_next(i,j,a,H,W,c.boundary);
        add_prob(s,a,s_int,p_int);
        int orth1=(a<2?2:0), orth2=(a<2?3:1);
        add_prob(s,a, step_next(i,j,orth1,H,W,c.boundary), p_slip);
        add_prob(s,a, step_next(i,j,orth2,H,W,c.boundary), p_slip);
      } else { // UniformNeighbors over {N,S,E,W} respecting boundary
        // Compute neighbors (including self if off-grid per boundary handler)
        int ns[4] = {
            step_next(i,j,0,H,W,c.boundary),
            step_next(i,j,1,H,W,c.boundary),
            step_next(i,j,2,H,W,c.boundary),
            step_next(i,j,3,H,W,c.boundary)
        };
        // equally likely next states independent of action
        for(int k=0;k<4;++k) add_prob(s,a,ns[k], 0.25);
      }

      // Reward model
      if (c.reward==RewardModel::Zero) {
        // nothing
      } else if (c.reward==RewardModel::SparseGoal){
        for(int g:goals) out.R[s][a][g] = 1.0;
      } else if (c.reward==RewardModel::ObstaclesNeg){
        for(int s2=0;s2<S;++s2) if (obstacle[s2]) out.R[s][a][s2] = c.obstacle_penalty;
      } else { // Mixed
        for(int g:goals) out.R[s][a][g] = 1.0;
        for(int s2=0;s2<S;++s2) if (obstacle[s2]) out.R[s][a][s2] = c.obstacle_penalty;
      }
    }
  }
}

void generate_gridworld_sparse(const GridConfig& c, MDPSparse& out){
  int H=c.H, W=c.W, S=H*W, A=4;
  out = {}; out.S=S; out.A=A;

  std::mt19937 rng(c.seed);
  std::vector<int> goals; pick_goals(c, goals, rng);
  std::vector<char> is_goal(S,0); for(int g:goals) is_goal[g]=1;

  std::vector<char> blocked(S,0), obstacle(S,0);
  if (c.blocked_density>0){
    std::uniform_real_distribution<double> u(0.,1.);
    for(int s=0;s<S;++s) if (u(rng)<c.blocked_density) blocked[s]=1;
    blocked[0]=0;
  }
  if (c.obstacle_density>0){
    std::uniform_real_distribution<double> u(0.,1.);
    for(int s=0;s<S;++s) if(!is_goal[s] && u(rng)<c.obstacle_density) obstacle[s]=1;
  }

//  out.P.assign(S, std::vector<std::vector<SparseTransition>>(A));
//  out.R.assign(S, std::vector<std::unordered_map<int,double>>(A));

  out.P.clear(); out.P.resize(S);
  for (auto& v : out.P) v.resize(A);
  out.R.clear(); out.R.resize(S);
  for (auto& v : out.R) v.resize(A);

  auto add = [&](int s,int a,int s2,double p){
    if (blocked[s] || blocked[s2]) s2 = s; // reflect at walls
    if (p<=0) return;
    auto& vec = out.P[s][a];
    // accumulate
    for(auto& pr:vec){ if (pr.first==s2){ pr.second += p; return; } }
    vec.emplace_back(s2,p);
  };

  for(int s=0;s<S;++s){
    int i=s/W, j=s%W;
    for(int a=0;a<A;++a){
      if (c.trans==GridTrans::Deterministic){
        add(s,a, step_next(i,j,a,H,W,c.boundary), 1.0);
      } else if (c.trans==GridTrans::Slip){
        double p_int=1.0 - c.slip, p_slip=c.slip/2.0;
        add(s,a, step_next(i,j,a,H,W,c.boundary), p_int);
        int orth1=(a<2?2:0), orth2=(a<2?3:1);
        add(s,a, step_next(i,j,orth1,H,W,c.boundary), p_slip);
        add(s,a, step_next(i,j,orth2,H,W,c.boundary), p_slip);
      } else {
        for(int k=0;k<4;++k) add(s,a, step_next(i,j,k,H,W,c.boundary), 0.25);
      }

      if (c.reward==RewardModel::SparseGoal || c.reward==RewardModel::Mixed){
        for(int g:goals) out.R[s][a][g] = 1.0;
      }
      if (c.reward==RewardModel::ObstaclesNeg || c.reward==RewardModel::Mixed){
        for(int s2=0;s2<S;++s2) if (obstacle[s2]) out.R[s][a][s2] = c.obstacle_penalty;
      }
    }
  }
}

CSR to_csr(const MDPSparse& sp, bool for_rewards){
  CSR csr; csr.S=sp.S; csr.A=sp.A;
  int rows = sp.S * sp.A;
  csr.indptr.assign(rows+1,0);
  // count nnz
  int nnz=0;
  for(int s=0;s<sp.S;++s)
    for(int a=0;a<sp.A;++a){
      int row = s*sp.A + a;
      if (!for_rewards){
        nnz += (int)sp.P[s][a].size();
      } else {
        nnz += (int)sp.R[s][a].size();
      }
      csr.indptr[row+1]=nnz;
    }
  csr.indices.resize(nnz);
  csr.data.resize(nnz);
  int k=0;
  for(int s=0;s<sp.S;++s)
    for(int a=0;a<sp.A;++a){
      if (!for_rewards){
        for(const auto& [j,p]: sp.P[s][a]){ csr.indices[k]=j; csr.data[k]=p; ++k; }
      } else {
        for(const auto& kv: sp.R[s][a])   { csr.indices[k]=kv.first; csr.data[k]=kv.second; ++k; }
      }
    }
  return csr;
}

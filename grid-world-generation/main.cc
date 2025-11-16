// main.cc
#include <getopt.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "policy_iteration.h"            // your dense solver/types
#include "policy_improvement_sparse.h"   // your sparse solver/types

// -----------------------------------------------------------------------------
// Helpers: enums & parsing
// -----------------------------------------------------------------------------
enum class Dataset {
  Grid, Random, Toy, Edge
};
enum class Storage {
  Dense, Sparse
};
enum class GridTrans {
  Deterministic, Slip, UniformNeighbors
};
enum class Boundary {
  Reflect, Clamp, Wrap
};
enum class GoalMode {
  Fixed, Random, Multi
};
enum class RewardModel {
  Zero, SparseGoal, ObstaclesNeg, Mixed
};
enum class Sparsity {
  VerySparse, Moderate, Dense
};
enum class RType {
  UniformNeg1to1, Binary01, Binary0m1, PatternEveryK
};
enum class EdgeKind {
  Single, Disconnected, Absorb0, Absorb1, ZeroReward, Cycle
};

// -----------------------------------------------------------------------------
// CSR container (rows = S*A, row = s*A + a) – used only for saving JSON fast.
// -----------------------------------------------------------------------------
struct CSR {
  int S{0}, A{0};
  std::vector<int> indptr;   // size S*A + 1
  std::vector<int> indices;  // next-state ids
  std::vector<double> data;     // probabilities or rewards
};

static void die_usage(const char *prog) {
  std::cerr <<
            "Usage: " << prog << " [OPTIONS]\n\n"
                                 "DATASET SELECTION\n"
                                 "  --dataset grid|random|toy|edge         (default: grid)\n"
                                 "  --storage dense|sparse                 (default: auto: dense<=1000 states)\n"
                                 "  --gen-only                             (only write mdp.json; skip solving)\n"
                                 "\nGRID OPTIONS\n"
                                 "  --height H  --width W                  (default: 50x50)\n"
                                 "  --trans det|slip|uniform               (default: slip)\n"
                                 "  --slip p                               (default: 0.1)\n"
                                 "  --boundary reflect|clamp|wrap          (default: reflect)\n"
                                 "  --reward zero|sparse_goal|obstacles_neg|mixed  (default: sparse_goal)\n"
                                 "  --blocked x                            (fraction of walls, default 0)\n"
                                 "  --obstacles x                          (fraction of penalty cells, default 0)\n"
                                 "  --obs-pen r                            (reward for obstacle cells, default -1)\n"
                                 "  --goals fixed|random|multi:K           (default: fixed)\n"
                                 "  --seed N                               (default: 42)\n"
                                 "\nRANDOM-SPARSE OPTIONS\n"
                                 "  --states S                             (default: 1000)\n"
                                 "  --actions fixed:N | vary:min:max       (default: fixed:4)\n"
                                 "  --sparsity very|moderate|dense         (default: very)\n"
                                 "  --rtype uniform|bin01|bin0m1|pattern   (default: uniform)\n"
                                 "  --pattern-k K                          (default: 10)\n"
                                 "  --seed N                               (default: 123)\n"
                                 "\nTOY OPTIONS\n"
                                 "  --size N                               (3|4|5, default: 4)\n"
                                 "\nEDGE OPTIONS\n"
                                 "  --edge single|disconnected|absorbing0|absorbing1|zero|cycle:K\n"
                                 "  --states S  --actions A                (some edge types use these)\n"
                                 "\nSOLVER/IO\n"
                                 "  --gamma g   --theta t  --max-iter n    (default: 0.9, 1e-8, 1000)\n"
                                 "  --out-mdp file.json                    (default: mdp.json; CSR format)\n"
                                 "  --out-policy policy.csv                (default: policy.csv)\n"
                                 "  --out-value  value.csv                 (default: value.csv)\n";
  std::exit(1);
}

static Dataset parse_dataset(const std::string &s) {
  if (s == "grid") return Dataset::Grid;
  if (s == "random") return Dataset::Random;
  if (s == "toy") return Dataset::Toy;
  if (s == "edge") return Dataset::Edge;
  die_usage("mdp_tool");
  return Dataset::Grid;
}

static Storage parse_storage(const std::string &s) {
  if (s == "dense") return Storage::Dense;
  if (s == "sparse") return Storage::Sparse;
  die_usage("mdp_tool");
  return Storage::Dense;
}

static GridTrans parse_trans(const std::string &s) {
  if (s == "det") return GridTrans::Deterministic;
  if (s == "slip") return GridTrans::Slip;
  if (s == "uniform") return GridTrans::UniformNeighbors;
  die_usage("mdp_tool");
  return GridTrans::Slip;
}

static Boundary parse_boundary(const std::string &s) {
  if (s == "reflect") return Boundary::Reflect;
  if (s == "clamp") return Boundary::Clamp;
  if (s == "wrap") return Boundary::Wrap;
  die_usage("mdp_tool");
  return Boundary::Reflect;
}

static RewardModel parse_reward_model(const std::string &s) {
  if (s == "zero") return RewardModel::Zero;
  if (s == "sparse_goal") return RewardModel::SparseGoal;
  if (s == "obstacles_neg") return RewardModel::ObstaclesNeg;
  if (s == "mixed") return RewardModel::Mixed;
  die_usage("mdp_tool");
  return RewardModel::SparseGoal;
}

static Sparsity parse_sparsity(const std::string &s) {
  if (s == "very") return Sparsity::VerySparse;
  if (s == "moderate") return Sparsity::Moderate;
  if (s == "dense") return Sparsity::Dense;
  die_usage("mdp_tool");
  return Sparsity::VerySparse;
}

static RType parse_rtype(const std::string &s) {
  if (s == "uniform") return RType::UniformNeg1to1;
  if (s == "bin01") return RType::Binary01;
  if (s == "bin0m1") return RType::Binary0m1;
  if (s == "pattern") return RType::PatternEveryK;
  die_usage("mdp_tool");
  return RType::UniformNeg1to1;
}

// -----------------------------------------------------------------------------
// Local utilities
// -----------------------------------------------------------------------------
static inline int clampi(int x, int lo, int hi) { return std::max(lo, std::min(x, hi)); }

static int step_next(int i, int j, int a, int H, int W, Boundary b) {
  static const int di[4] = {-1, 1, 0, 0}, dj[4] = {0, 0, 1, -1};
  int ni = i + di[a], nj = j + dj[a];
  bool off = (ni < 0 || ni >= H || nj < 0 || nj >= W);
  if (!off) return ni * W + nj;
  switch (b) {
    case Boundary::Reflect:
      return i * W + j;
    case Boundary::Clamp:
      return clampi(ni, 0, H - 1) * W + clampi(nj, 0, W - 1);
    case Boundary::Wrap:
      return (((ni % H) + H) % H) * W + (((nj % W) + W) % W);
  }
  return i * W + j;
}

static void pick_goals(int H, int W, GoalMode mode, int multi_k, unsigned seed, std::vector<int> &goals) {
  int S = H * W;
  if (mode == GoalMode::Fixed) {
    goals = {(H - 1) * W + (W - 1)};
    return;
  }
  std::mt19937 rng(seed);
  if (mode == GoalMode::Random) {
    std::uniform_int_distribution<int> uni(0, S - 1);
    goals = {uni(rng)};
    return;
  }
  // Multi
  std::uniform_int_distribution<int> uni(0, S - 1);
  goals.clear();
  for (int k = 0; k < multi_k; ++k) goals.push_back(uni(rng));
  std::sort(goals.begin(), goals.end());
  goals.erase(std::unique(goals.begin(), goals.end()), goals.end());
}

static int nnz_per_action(Sparsity sp, int S, std::mt19937 &rng) {
  std::uniform_int_distribution<int> u2(1, 2), u10(5, 10);
  switch (sp) {
    case Sparsity::VerySparse:
      return u2(rng);
    case Sparsity::Moderate:
      return u10(rng);
    case Sparsity::Dense:
      return std::max(1, S / 2);
  }
  return 2;
}

static void normalize(std::vector<double> &w) {
  double z = 0.0;
  for (double x: w) z += x;
  if (z <= 0) return;
  for (double &x: w) x /= z;
}

// -----------------------------------------------------------------------------
// CSR conversion & saving
// -----------------------------------------------------------------------------
static CSR to_csr_from_dense(const Matrix3D &P_or_R) {
  int S = (int) P_or_R.size();
  int A = (int) P_or_R[0].size();
  CSR csr;
  csr.S = S;
  csr.A = A;
  csr.indptr.assign(S * A + 1, 0);
  int nnz = 0, row = 0;
  for (int s = 0; s < S; ++s) {
    for (int a = 0; a < A; ++a) {
      const auto &rowvec = P_or_R[s][a];
      for (int j = 0; j < S; ++j) {
        double v = rowvec[j];
        if (std::fabs(v) > 0.0) {
          csr.indices.push_back(j);
          csr.data.push_back(v);
          ++nnz;
        }
      }
      csr.indptr[++row] = nnz;
    }
  }
  return csr;
}

static CSR to_csr_from_sparse(const SparseMatrix &P, const SparseReward &R, bool wantP) {
  int S = (int) P.size();
  int A = (int) P[0].size();
  CSR csr;
  csr.S = S;
  csr.A = A;
  csr.indptr.assign(S * A + 1, 0);
  int nnz = 0, row = 0;
  for (int s = 0; s < S; ++s) {
    for (int a = 0; a < A; ++a) {
      if (wantP) {
        for (const auto &pr: P[s][a]) {
          csr.indices.push_back(pr.first);
          csr.data.push_back(pr.second);
          ++nnz;
        }
      } else {
        for (const auto &kv: R[s][a]) {
          csr.indices.push_back(kv.first);
          csr.data.push_back(kv.second);
          ++nnz;
        }
      }
      csr.indptr[++row] = nnz;
    }
  }
  return csr;
}

static void save_csr_json(const CSR &Pcsr, const CSR &Rcsr, double gamma, const std::string &path) {
  std::ofstream f(path);
  if (!f) {
    std::cerr << "Failed to open " << path << " for write\n";
    return;
  }
  f << "{\n";
  f << "  \"S\": " << Pcsr.S << ",\n";
  f << "  \"A\": " << Pcsr.A << ",\n";
  f << "  \"gamma\": " << gamma << ",\n";
  f << "  \"format\": \"CSR\",\n";
  auto dump_arr_int = [&](const char *name, const std::vector<int> &v) {
    f << "    \"" << name << "\": [";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) f << ",";
      f << v[i];
    }
    f << "]";
  };
  auto dump_arr_double = [&](const char *name, const std::vector<double> &v) {
    f << "    \"" << name << "\": [";
    for (size_t i = 0; i < v.size(); ++i) {
      if (i) f << ",";
      f << std::setprecision(17) << v[i];
    }
    f << "]";
  };
  // P
  f << "  \"P\": {\n";
  dump_arr_int("indptr", Pcsr.indptr);
  f << ",\n";
  dump_arr_int("indices", Pcsr.indices);
  f << ",\n";
  dump_arr_double("data", Pcsr.data);
  f << "\n";
  f << "  },\n";
  // R
  f << "  \"R\": {\n";
  dump_arr_int("indptr", Rcsr.indptr);
  f << ",\n";
  dump_arr_int("indices", Rcsr.indices);
  f << ",\n";
  dump_arr_double("data", Rcsr.data);
  f << "\n";
  f << "  }\n";
  f << "}\n";
}

// -----------------------------------------------------------------------------
// GENERATORS
// -----------------------------------------------------------------------------

// ---- GRID WORLD (dense)
static void generate_gridworld_dense_general(
    int H, int W, GridTrans trans, Boundary boundary, RewardModel rmodel,
    double slip, double blocked_density, double obstacle_density, double obstacle_penalty,
    GoalMode gmode, int goals_k, unsigned seed,
    Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  const int S = H * W, A = 4;
  gamma_out = gamma_in;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<int> goals;
  pick_goals(H, W, gmode, goals_k, seed, goals);
  std::vector<char> is_goal(S, 0);
  for (int g: goals) if (g >= 0 && g < S) is_goal[g] = 1;

  // Walls (blocked cells) and penalty obstacles (enterable)
  std::vector<char> wall(S, 0), obstacle(S, 0);
  if (blocked_density > 0) {
    for (int s = 0; s < S; ++s) if (U(rng) < blocked_density) wall[s] = 1;
    wall[0] = 0; // keep start open
    for (int g: goals) wall[g] = 0;
  }
  if (obstacle_density > 0) {
    for (int s = 0; s < S; ++s) if (!wall[s] && !is_goal[s] && U(rng) < obstacle_density) obstacle[s] = 1;
  }

  auto add_prob = [&](int s, int a, int s2, double p) {
    if (wall[s] || wall[s2]) s2 = s; // reflect from/into a wall
    P[s][a][s2] += p;
  };

  for (int s = 0; s < S; ++s) {
    int i = s / W, j = s % W;
    for (int a = 0; a < A; ++a) {
      if (trans == GridTrans::Deterministic) {
        add_prob(s, a, step_next(i, j, a, H, W, boundary), 1.0);
      } else if (trans == GridTrans::Slip) {
        double p_int = 1.0 - slip, p_slip = slip / 2.0;
        add_prob(s, a, step_next(i, j, a, H, W, boundary), p_int);
        int orth1 = (a < 2 ? 2 : 0), orth2 = (a < 2 ? 3 : 1);
        add_prob(s, a, step_next(i, j, orth1, H, W, boundary), p_slip);
        add_prob(s, a, step_next(i, j, orth2, H, W, boundary), p_slip);
      } else { // UniformNeighbors
        for (int k = 0; k < 4; ++k) add_prob(s, a, step_next(i, j, k, H, W, boundary), 0.25);
      }

      if (rmodel == RewardModel::SparseGoal || rmodel == RewardModel::Mixed) {
        for (int g: goals) R[s][a][g] = 1.0;
      }
      if (rmodel == RewardModel::ObstaclesNeg || rmodel == RewardModel::Mixed) {
        for (int s2 = 0; s2 < S; ++s2) if (obstacle[s2]) R[s][a][s2] = obstacle_penalty;
      }
    }
  }
}

// ---- GRID WORLD (sparse)
static void generate_gridworld_sparse_general(
    int H, int W, GridTrans trans, Boundary boundary, RewardModel rmodel,
    double slip, double blocked_density, double obstacle_density, double obstacle_penalty,
    GoalMode gmode, int goals_k, unsigned seed,
    SparseMatrix &P, SparseReward &R, double &gamma_out, double gamma_in) {
  const int S = H * W, A = 4;
  gamma_out = gamma_in;
//  P.assign(S, std::vector<std::vector<SparseTransition>>(A));
//  R.assign(S, std::vector<std::unordered_map<int,double>>(A));

  P.clear();
  P.resize(S);
  for (auto &v: P) v.resize(A);
  R.clear();
  R.resize(S);
  for (auto &v: R) v.resize(A);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 1.0);

  std::vector<int> goals;
  pick_goals(H, W, gmode, goals_k, seed, goals);
  std::vector<char> is_goal(S, 0);
  for (int g: goals) if (g >= 0 && g < S) is_goal[g] = 1;

  std::vector<char> wall(S, 0), obstacle(S, 0);
  if (blocked_density > 0) {
    for (int s = 0; s < S; ++s) if (U(rng) < blocked_density) wall[s] = 1;
    wall[0] = 0;
    for (int g: goals) wall[g] = 0;
  }
  if (obstacle_density > 0) {
    for (int s = 0; s < S; ++s) if (!wall[s] && !is_goal[s] && U(rng) < obstacle_density) obstacle[s] = 1;
  }

  auto add = [&](int s, int a, int s2, double p) {
    if (wall[s] || wall[s2]) s2 = s;
    if (p <= 0) return;
    auto &v = P[s][a];
    for (auto &pr: v) {
      if (pr.first == s2) {
        pr.second += p;
        return;
      }
    }
    v.emplace_back(s2, p);
  };

  for (int s = 0; s < S; ++s) {
    int i = s / W, j = s % W;
    for (int a = 0; a < A; ++a) {
      if (trans == GridTrans::Deterministic) {
        add(s, a, step_next(i, j, a, H, W, boundary), 1.0);
      } else if (trans == GridTrans::Slip) {
        double p_int = 1.0 - slip, p_slip = slip / 2.0;
        add(s, a, step_next(i, j, a, H, W, boundary), p_int);
        int o1 = (a < 2 ? 2 : 0), o2 = (a < 2 ? 3 : 1);
        add(s, a, step_next(i, j, o1, H, W, boundary), p_slip);
        add(s, a, step_next(i, j, o2, H, W, boundary), p_slip);
      } else {
        for (int k = 0; k < 4; ++k) add(s, a, step_next(i, j, k, H, W, boundary), 0.25);
      }

      if (rmodel == RewardModel::SparseGoal || rmodel == RewardModel::Mixed) {
        for (int g: goals) R[s][a][g] = 1.0;
      }
      if (rmodel == RewardModel::ObstaclesNeg || rmodel == RewardModel::Mixed) {
        for (int s2 = 0; s2 < S; ++s2) if (obstacle[s2]) R[s][a][s2] = obstacle_penalty;
      }
    }
  }
}

// ---- RANDOM SPARSE MDP (always builds SparseMatrix/SparseReward)
static void generate_random_sparse_mdp(
    int S, int A_fixed, bool vary_actions, int A_min, int A_max,
    Sparsity sparsity, RType rtype, int pattern_k, unsigned seed,
    SparseMatrix &P, SparseReward &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  int A = vary_actions ? A_max : A_fixed;
//  P.assign(S, std::vector<std::vector<SparseTransition>>(A));
//  R.assign(S, std::vector<std::unordered_map<int,double>>(A));

  P.clear();
  P.resize(S);
  for (auto &v: P) v.resize(A);
  R.clear();
  R.resize(S);
  for (auto &v: R) v.resize(A);


  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  std::uniform_int_distribution<int> uniS(0, S - 1);
  auto choose_k = [&](int S_) { return nnz_per_action(sparsity, S_, rng); };

  for (int s = 0; s < S; ++s) {
    int A_s = vary_actions ? std::uniform_int_distribution<int>(A_min, A_max)(rng) : A_fixed;
    for (int a = 0; a < A_s; ++a) {
      int k = choose_k(S);
      // sample k unique next states
      std::vector<int> cols;
      cols.reserve(k);
      while ((int) cols.size() < k) {
        int j = uniS(rng);
        if (std::find(cols.begin(), cols.end(), j) == cols.end()) cols.push_back(j);
      }
      std::vector<double> w(k);
      for (double &x: w) x = u01(rng) + 1e-6;
      normalize(w);
      for (int m = 0; m < k; ++m) P[s][a].emplace_back(cols[m], w[m]);

      if (rtype == RType::UniformNeg1to1) {
        std::uniform_real_distribution<double> ur(-1.0, 1.0);
        for (int j: cols) R[s][a][j] = ur(rng);
      } else if (rtype == RType::Binary01) {
        for (int j: cols) R[s][a][j] = (u01(rng) < 0.5 ? 0.0 : 1.0);
      } else if (rtype == RType::Binary0m1) {
        for (int j: cols) R[s][a][j] = (u01(rng) < 0.5 ? 0.0 : -1.0);
      } else { // PatternEveryK
        for (int j: cols) if (pattern_k > 0 && (j % pattern_k) == 0) R[s][a][j] = 1.0;
      }
    }
    // Leave remaining actions (a >= A_s) empty → valid but disconnected actions
  }
}

// ---- TOY deterministic grid (N x N), reward only at goal = bottom-right
static void generate_toy_grid(int N, Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  int H = N, W = N, S = H * W, A = 4;
  gamma_out = gamma_in;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  int goal = (H - 1) * W + (W - 1);
  for (int s = 0; s < S; ++s) {
    int i = s / W, j = s % W;
    for (int a = 0; a < A; ++a) {
      int s2 = step_next(i, j, a, H, W, Boundary::Reflect);
      P[s][a][s2] = 1.0;
      R[s][a][goal] = 1.0;
    }
  }
}

// ---- EDGE cases (dense)
static void generate_single_state(Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  P.assign(1, std::vector < std::vector < double >> (1, std::vector<double>(1, 0.0)));
  R = P;
  P[0][0][0] = 1.0;
  R[0][0][0] = 0.0;
}

static void generate_disconnected(int S, int A, Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R = P; // all zeros; some rows intentionally all-zero (disconnected)
}

static void
generate_absorbing(int S, int A, bool reward1, Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  for (int s = 0; s < S; ++s)
    for (int a = 0; a < A; ++a) {
      P[s][a][s] = 1.0;
      if (reward1) R[s][a][s] = 1.0;
    }
}

static void generate_zero_reward(int S, int A, Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R = P;
  // simple self-loops
  for (int s = 0; s < S; ++s) for (int a = 0; a < A; ++a) P[s][a][s] = 1.0;
}

static void generate_cycle(int K, int A, Matrix3D &P, Matrix3D &R, double &gamma_out, double gamma_in) {
  gamma_out = gamma_in;
  int S = K;
  if (A < 1) A = 1;
  P.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  R.assign(S, std::vector < std::vector < double >> (A, std::vector<double>(S, 0.0)));
  for (int s = 0; s < S; ++s) {
    int s2 = (s + 1) % S;
    for (int a = 0; a < A; ++a) { P[s][a][s2] = 1.0; }
  }
  // zero rewards; stresses high-gamma convergence
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char **argv) {
  // Defaults
  Dataset dataset = Dataset::Grid;
  Storage storage = Storage::Dense;
  bool storage_set = false;
  bool gen_only = false;

  // Common
  double gamma = 0.9, theta = 1e-8;
  int max_iter = 1000;
  std::string out_mdp = "mdp.json";
  std::string out_policy = "policy.csv";
  std::string out_value = "value.csv";

  // Grid defaults
  int H = 50, W = 50;
  GridTrans trans = GridTrans::Slip;
  double slip = 0.1;
  Boundary boundary = Boundary::Reflect;
  RewardModel rmodel = RewardModel::SparseGoal;
  double blocked_density = 0.0, obstacle_density = 0.0, obstacle_penalty = -1.0;
  GoalMode gmode = GoalMode::Fixed;
  int goals_k = 1;
  unsigned grid_seed = 42;

  // Random MDP defaults
  int RS = 1000;
  bool vary_actions = false;
  int A_fixed = 4, A_min = 2, A_max = 6;
  Sparsity sparsity = Sparsity::VerySparse;
  RType rtype = RType::UniformNeg1to1;
  int pattern_k = 10;
  unsigned rnd_seed = 123;

  // Toy / Edge
  int toyN = 4;
  EdgeKind edge = EdgeKind::Single;
  int edge_S = 5, edge_A = 2, cycle_K = 10;

  // CLI options
  struct option long_opts[] = {
      {"dataset",    required_argument, nullptr, 1},
      {"storage",    required_argument, nullptr, 2},
      {"gen-only",   no_argument,       nullptr, 3},
      {"height",     required_argument, nullptr, 'H'},
      {"width",      required_argument, nullptr, 'W'},
      {"trans",      required_argument, nullptr, 4},
      {"slip",       required_argument, nullptr, 's'},
      {"boundary",   required_argument, nullptr, 5},
      {"reward",     required_argument, nullptr, 6},
      {"blocked",    required_argument, nullptr, 7},
      {"obstacles",  required_argument, nullptr, 8},
      {"obs-pen",    required_argument, nullptr, 9},
      {"goals",      required_argument, nullptr, 10},
      {"seed",       required_argument, nullptr, 11},

      {"states",     required_argument, nullptr, 12},
      {"actions",    required_argument, nullptr, 13},
      {"sparsity",   required_argument, nullptr, 14},
      {"rtype",      required_argument, nullptr, 15},
      {"pattern-k",  required_argument, nullptr, 16},

      {"size",       required_argument, nullptr, 17},
      {"edge",       required_argument, nullptr, 18},

      {"gamma",      required_argument, nullptr, 'g'},
      {"theta",      required_argument, nullptr, 't'},
      {"max-iter",   required_argument, nullptr, 'm'},
      {"out-mdp",    required_argument, nullptr, 19},
      {"out-policy", required_argument, nullptr, 'p'},
      {"out-value",  required_argument, nullptr, 'v'},
      {nullptr,      0,                 nullptr, 0}
  };

  int opt;
  while ((opt = getopt_long(argc, argv, "H:W:s:g:t:m:p:v:", long_opts, nullptr)) != -1) {
    switch (opt) {
      case 1:
        dataset = parse_dataset(optarg);
        break;
      case 2:
        storage = parse_storage(optarg);
        storage_set = true;
        break;
      case 3:
        gen_only = true;
        break;
      case 'H':
        H = std::stoi(optarg);
        break;
      case 'W':
        W = std::stoi(optarg);
        break;
      case 4:
        trans = parse_trans(optarg);
        break;
      case 's':
        slip = std::stod(optarg);
        break;
      case 5:
        boundary = parse_boundary(optarg);
        break;
      case 6:
        rmodel = parse_reward_model(optarg);
        break;
      case 7:
        blocked_density = std::stod(optarg);
        break;
      case 8:
        obstacle_density = std::stod(optarg);
        break;
      case 9:
        obstacle_penalty = std::stod(optarg);
        break;
      case 10: {
        std::string v(optarg);
        if (v == "fixed") gmode = GoalMode::Fixed;
        else if (v == "random") gmode = GoalMode::Random;
        else {
          if (v.rfind("multi:", 0) == 0) {
            gmode = GoalMode::Multi;
            goals_k = std::max(1, std::stoi(v.substr(6)));
          }
          else die_usage(argv[0]);
        }
      }
        break;
      case 11: {
        unsigned x = (unsigned) std::stoul(optarg);
        grid_seed = x;
        rnd_seed = x;
      }
        break;

      case 12:
        RS = std::stoi(optarg);
        break;
      case 13: {
        std::string v(optarg);
        if (v.rfind("fixed:", 0) == 0) {
          vary_actions = false;
          A_fixed = std::stoi(v.substr(6));
        }
        else if (v.rfind("vary:", 0) == 0) {
          vary_actions = true;
          auto rest = v.substr(5);
          auto p = rest.find(':');
          if (p == std::string::npos) die_usage(argv[0]);
          A_min = std::stoi(rest.substr(0, p));
          A_max = std::stoi(rest.substr(p + 1));
        } else die_usage(argv[0]);
      }
        break;
      case 14:
        sparsity = parse_sparsity(optarg);
        break;
      case 15:
        rtype = parse_rtype(optarg);
        break;
      case 16:
        pattern_k = std::stoi(optarg);
        break;

      case 17:
        toyN = std::stoi(optarg);
        break;
      case 18: {
        std::string v(optarg);
        if (v == "single") edge = EdgeKind::Single;
        else if (v == "disconnected") edge = EdgeKind::Disconnected;
        else if (v == "absorbing0") edge = EdgeKind::Absorb0;
        else if (v == "absorbing1") edge = EdgeKind::Absorb1;
        else if (v == "zero") edge = EdgeKind::ZeroReward;
        else if (v.rfind("cycle:", 0) == 0) {
          edge = EdgeKind::Cycle;
          cycle_K = std::max(1, std::stoi(v.substr(6)));
        }
        else die_usage(argv[0]);
      }
        break;

      case 'g':
        gamma = std::stod(optarg);
        break;
      case 't':
        theta = std::stod(optarg);
        break;
      case 'm':
        max_iter = std::stoi(optarg);
        break;
      case 19:
        out_mdp = optarg;
        break;
      case 'p':
        out_policy = optarg;
        break;
      case 'v':
        out_value = optarg;
        break;
      default:
        die_usage(argv[0]);
    }
  }

  // Decide default storage for grid/toy/edge if not set
  if (!storage_set && (dataset == Dataset::Grid || dataset == Dataset::Toy || dataset == Dataset::Edge)) {
    long long S = (dataset == Dataset::Grid) ? 1LL * H * W :
                  (dataset == Dataset::Toy) ? 1LL * toyN * toyN :
                  (edge == EdgeKind::Cycle) ? (long long) cycle_K : (long long) edge_S;
    storage = (S > 1000 ? Storage::Sparse : Storage::Dense);
  }

  Policy policy;
  Vector value;

  if (dataset == Dataset::Grid) {
    std::cout << "Generating GRID " << H << "x" << W << "  trans="
              << (trans == GridTrans::Deterministic ? "det" : (trans == GridTrans::Slip ? "slip" : "uniform"))
              << "  boundary="
              << (boundary == Boundary::Reflect ? "reflect" : boundary == Boundary::Clamp ? "clamp" : "wrap")
              << "  reward="
              << (rmodel == RewardModel::Zero ? "zero" : rmodel == RewardModel::SparseGoal ? "sparse_goal" : rmodel ==
                                                                                                             RewardModel::ObstaclesNeg
                                                                                                             ? "obstacles_neg"
                                                                                                             : "mixed")
              << "  storage=" << (storage == Storage::Dense ? "dense" : "sparse") << "\n";

    double gamma_used = gamma;

    if (storage == Storage::Dense) {
      Matrix3D P, R;
      generate_gridworld_dense_general(H, W, trans, boundary, rmodel, slip,
                                       blocked_density, obstacle_density, obstacle_penalty,
                                       gmode, goals_k, grid_seed, P, R, gamma_used, gamma);
      // save CSR snapshot
      save_csr_json(to_csr_from_dense(P), to_csr_from_dense(R), gamma_used, out_mdp);

      if (!gen_only) {
        auto [pol, val] = policy_iteration(P, R, gamma_used, theta, max_iter);
        policy = pol;
        value = val;
      }
    } else {
      SparseMatrix P;
      SparseReward R;
      generate_gridworld_sparse_general(H, W, trans, boundary, rmodel, slip,
                                        blocked_density, obstacle_density, obstacle_penalty,
                                        gmode, goals_k, grid_seed, P, R, gamma_used, gamma);
      // save CSR snapshot
      save_csr_json(to_csr_from_sparse(P, R, true), to_csr_from_sparse(P, R, false), gamma_used, out_mdp);

      if (!gen_only) {
        auto [pol, val] = policy_iteration_sparse(P, R, gamma_used, theta, max_iter);
        policy = pol;
        value = val;
      }
    }
  } else if (dataset == Dataset::Random) {
    std::cout << "Generating RANDOM-SPARSE S=" << RS
              << " actions=" << (vary_actions ? "vary" : "fixed")
              << " sparsity="
              << (sparsity == Sparsity::VerySparse ? "very" : sparsity == Sparsity::Moderate ? "moderate" : "dense")
              << " rtype="
              << (rtype == RType::UniformNeg1to1 ? "uniform" : rtype == RType::Binary01 ? "bin01" : rtype ==
                                                                                                    RType::Binary0m1
                                                                                                    ? "bin0m1"
                                                                                                    : "pattern")
              << "\n";
    // Always build sparse (your solver is sparse-friendly)
    SparseMatrix P;
    SparseReward R;
    double gamma_used = gamma;
    generate_random_sparse_mdp(RS, A_fixed, vary_actions, A_min, A_max,
                               sparsity, rtype, pattern_k, rnd_seed,
                               P, R, gamma_used, gamma);
    // Save CSR
    save_csr_json(to_csr_from_sparse(P, R, true), to_csr_from_sparse(P, R, false), gamma_used, out_mdp);

    if (!gen_only) {
      auto [pol, val] = policy_iteration_sparse(P, R, gamma_used, theta, max_iter);
      policy = pol;
      value = val;
    }
  } else if (dataset == Dataset::Toy) {
    std::cout << "Generating TOY grid " << toyN << "x" << toyN << " (deterministic)\n";
    Matrix3D P, R;
    double gamma_used = gamma;
    generate_toy_grid(toyN, P, R, gamma_used, gamma);
    save_csr_json(to_csr_from_dense(P), to_csr_from_dense(R), gamma_used, out_mdp);
    if (!gen_only) {
      auto [pol, val] = policy_iteration(P, R, gamma_used, theta, max_iter);
      policy = pol;
      value = val;
    }
  } else { // Edge
    std::cout << "Generating EDGE case\n";
    Matrix3D P, R;
    double gamma_used = gamma;
    switch (edge) {
      case EdgeKind::Single:
        generate_single_state(P, R, gamma_used, gamma);
        break;
      case EdgeKind::Disconnected:
        generate_disconnected(edge_S, edge_A, P, R, gamma_used, gamma);
        break;
      case EdgeKind::Absorb0:
        generate_absorbing(edge_S, edge_A, false, P, R, gamma_used, gamma);
        break;
      case EdgeKind::Absorb1:
        generate_absorbing(edge_S, edge_A, true, P, R, gamma_used, gamma);
        break;
      case EdgeKind::ZeroReward:
        generate_zero_reward(edge_S, edge_A, P, R, gamma_used, gamma);
        break;
      case EdgeKind::Cycle:
        generate_cycle(cycle_K, edge_A, P, R, gamma_used, gamma);
        break;
    }
    save_csr_json(to_csr_from_dense(P), to_csr_from_dense(R), gamma_used, out_mdp);
    if (!gen_only) {
      auto [pol, val] = policy_iteration(P, R, gamma_used, theta, max_iter);
      policy = pol;
      value = val;
    }
  }

  // Write policy/value (if solved)
  if (!gen_only && !policy.empty()) {
    {
      std::ofstream pf(out_policy);
      if (!pf) {
        std::cerr << "Failed to open " << out_policy << "\n";
        return 1;
      }
      pf << "state,action\n";
      for (size_t s = 0; s < policy.size(); ++s) pf << s << "," << policy[s] << "\n";
    }
    {
      std::ofstream vf(out_value);
      if (!vf) {
        std::cerr << "Failed to open " << out_value << "\n";
        return 1;
      }
      vf << "state,value\n";
      for (size_t s = 0; s < value.size(); ++s) vf << s << "," << std::setprecision(17) << value[s] << "\n";
    }
    std::cout << "Wrote policy to " << out_policy << " and value to " << out_value << "\n";
  }
  std::cout << "Wrote MDP to " << out_mdp << " (CSR JSON)\n";
  return 0;
}

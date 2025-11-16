// mdp.h
#pragma once
#include <vector>
#include <utility>
#include <unordered_map>
#include <string>

using Vector   = std::vector<double>;
using Policy   = std::vector<int>;
using Matrix3D = std::vector<std::vector<std::vector<double>>>; // [S][A][S]

// Sparse per (s,a): list of (next_state, prob)
using SparseTransition = std::pair<int,double>;
using SparseMatrixSA   = std::vector<std::vector<std::vector<SparseTransition>>>; // [S][A][]

// CSR over rows = S*A
struct CSR {
  int S{0}, A{0};
  // Row i corresponds to (s = i / A, a = i % A)
  std::vector<int> indptr;   // size S*A + 1
  std::vector<int> indices;  // next states
  std::vector<double> data;  // probs
};

enum class GridTrans { Deterministic, Slip, UniformNeighbors };
enum class Boundary  { Reflect, Clamp, Wrap };
enum class GoalMode  { Fixed, Random, Multi };
enum class RewardModel { Zero, SparseGoal, ObstaclesNeg, Mixed };
enum class Storage   { Dense, CSR_ };
enum class Dataset   { Grid, RandomSparse, Toy, Edge };
enum class Sparsity  { VerySparse, Moderate, Dense };
enum class RType     { UniformNeg1to1, Binary01, Binary0m1, PatternEveryK };

struct GridConfig {
  int H=4, W=4, goals=1;
  double slip=0.1;
  GridTrans trans=GridTrans::Slip;
  Boundary boundary=Boundary::Reflect;
  RewardModel reward=RewardModel::SparseGoal;
  GoalMode goal_mode=GoalMode::Fixed;
  double blocked_density=0.0;     // fraction of blocked cells (walls)
  double obstacle_density=0.0;    // fraction of penalty cells (enterable)
  double obstacle_penalty=-1.0;   // reward for obstacle cells
  int    pattern_k=10;            // for pattern rewards
  unsigned seed=42;
};

struct RandomMDPConfig {
  int S=1000;
  int A_fixed=4;                 // if >0 used, else vary
  int A_min=2, A_max=6;          // used when varying
  Sparsity sparsity=Sparsity::VerySparse;
  RType rtype=RType::UniformNeg1to1;
  int    pattern_k=10;
  Storage storage=Storage::CSR_;
  unsigned seed=123;
};

struct MDPDense { int S=0, A=0; Matrix3D P, R; double gamma=0.9; };
struct MDPSparse { int S=0, A=0; SparseMatrixSA P; std::vector<std::vector<std::unordered_map<int,double>>> R; double gamma=0.9; };
struct MDPCSR { int S=0, A=0; CSR P; CSR R; double gamma=0.9; };

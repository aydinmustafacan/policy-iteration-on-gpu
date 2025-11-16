#include "policy_improvement_sparse.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>
#include <unordered_map>

void generate_gridworld_sparse(int H, int W, double slip,
                               SparseMatrix& P,
                               SparseReward& R)
{
  int S = H * W;
  const int A = 4;
  double p_intended = 1.0 - slip;
  double p_unintended = slip / 2.0;

  P.resize(S, std::vector<SparseTransition>(A));
  R.resize(S, std::vector<std::unordered_map<int, double>>(A));

  // Direction vectors: N, S, E, W
  const std::vector<std::pair<int,int>> dir = {
      {-1,  0}, {1,  0}, {0,  1}, {0, -1}
  };
  const std::vector<std::vector<int>> slip_dirs = {
      {2, 3}, {2, 3}, {0, 1}, {0, 1}
  };

  int goal = (H - 1) * W + (W - 1);

  for (int s = 0; s < S; ++s) {
    int i = s / W, j = s % W;
    for (int a = 0; a < A; ++a) {
      std::unordered_map<int, double> transitions;

      // Intended move
      int ni = i + dir[a].first;
      int nj = j + dir[a].second;
      int s_int = (ni >= 0 && ni < H && nj >= 0 && nj < W) ? (ni * W + nj) : s;
      transitions[s_int] += p_intended;

      // Slip moves
      for (int up : slip_dirs[a]) {
        int ui = i + dir[up].first;
        int uj = j + dir[up].second;
        int s_slip = (ui >= 0 && ui < H && uj >= 0 && uj < W) ? (ui * W + uj) : s;
        transitions[s_slip] += p_unintended;
      }

      // Convert to sparse format
      for (const auto& [next_state, prob] : transitions) {
        if (prob > 1e-10) {
          P[s][a].emplace_back(next_state, prob);
        }
      }

      // Set reward
      R[s][a][goal] = 1.0;
    }
  }
}

Vector policy_evaluation_sparse(const Policy& policy,
                                const SparseMatrix& P,
                                const SparseReward& R,
                                double gamma,
                                double theta,
                                int max_iters)
{
  int S = policy.size();
  Vector V(S, 0.0);

  for (int iter = 0; iter < max_iters; ++iter) {
    double delta = 0.0;
    for (int s = 0; s < S; ++s) {
      int a = policy[s];
      double v_old = V[s];
      double v_new = 0.0;

      // Only iterate over non-zero transitions
      for (const auto& [s2, prob] : P[s][a]) {
        double reward = 0.0;
        auto it = R[s][a].find(s2);
        if (it != R[s][a].end()) {
          reward = it->second;
        }
        v_new += prob * (reward + gamma * V[s2]);
      }

      V[s] = v_new;
      delta = std::max(delta, std::fabs(v_old - v_new));
    }
    if (delta < theta) break;
  }

  return V;
}

Policy policy_improvement_sparse(const Vector& V,
                                 const SparseMatrix& P,
                                 const SparseReward& R,
                                 double gamma)
{
  int S = V.size();
  int A = P[0].size();
  Policy new_policy(S, 0);

  for (int s = 0; s < S; ++s) {
    double best_q = -std::numeric_limits<double>::infinity();
    int best_a = 0;
    for (int a = 0; a < A; ++a) {
      double q = 0.0;
      for (const auto& [s2, prob] : P[s][a]) {
        double reward = 0.0;
        auto it = R[s][a].find(s2);
        if (it != R[s][a].end()) {
          reward = it->second;
        }
        q += prob * (reward + gamma * V[s2]);
      }
      if (q > best_q) {
        best_q = q;
        best_a = a;
      }
    }
    new_policy[s] = best_a;
  }

  return new_policy;
}

std::pair<Policy, Vector> policy_iteration_sparse(const SparseMatrix& P,
                                                  const SparseReward& R,
                                                  double gamma,
                                                  double theta,
                                                  int max_iters)
{
  int S = P.size();
  Policy policy(S, 0);
  Vector V(S, 0.0);

  for (int it = 0; it < max_iters; ++it) {
    V = policy_evaluation_sparse(policy, P, R, gamma, theta);
    Policy new_policy = policy_improvement_sparse(V, P, R, gamma);
    if (new_policy == policy) {
      std::cout << "Policy converged after " << it+1 << " iterations.\n";
      break;
    }
    policy = std::move(new_policy);
  }
  return {policy, V};
}
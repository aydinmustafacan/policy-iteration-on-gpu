#include "policy_iteration.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <utility>

void generate_gridworld(int H, int W, double slip,
                        Matrix3D& P,
                        Matrix3D& R)
{
  int S = H * W;
  const int A = 4;
  double p_intended   = 1.0 - slip;
  double p_unintended = slip / 2.0;

  // Direction vectors: N, S, E, W
  const std::vector<std::pair<int,int>> dir = {
      {-1,  0},  // North
      { 1,  0},  // South
      { 0,  1},  // East
      { 0, -1}   // West
  };
  // For N/S, slip goes E/W; for E/W, slip goes N/S
  const std::vector<std::vector<int>> slip_dirs = {
      {2, 3},  // from North
      {2, 3},  // from South
      {0, 1},  // from East
      {0, 1}   // from West
  };

  int goal = (H - 1) * W + (W - 1);

  // Fill P and R
  for (int s = 0; s < S; ++s) {
    int i = s / W, j = s % W;
    for (int a = 0; a < A; ++a) {
      // zero out
      std::fill(P[s][a].begin(), P[s][a].end(), 0.0);
      std::fill(R[s][a].begin(), R[s][a].end(), 0.0);

      // intended move
      int ni = i + dir[a].first;
      int nj = j + dir[a].second;
      int s_int = (ni>=0 && ni<H && nj>=0 && nj<W) ? (ni*W + nj) : s;
      P[s][a][s_int] += p_intended;

      // slips
      for (int up : slip_dirs[a]) {
        int ui = i + dir[up].first;
        int uj = j + dir[up].second;
        int s_slip = (ui>=0 && ui<H && uj>=0 && uj<W) ? (ui*W + uj) : s;
        P[s][a][s_slip] += p_unintended;
      }

      // reward: +1 on reaching goal
      R[s][a][goal] = 1.0;
    }
  }
}

Vector policy_evaluation(const Policy& policy,
                         const Matrix3D& P,
                         const Matrix3D& R,
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
      for (int s2 = 0; s2 < S; ++s2) {
        v_new += P[s][a][s2] * (R[s][a][s2] + gamma * V[s2]);
      }
      V[s] = v_new;
      delta = std::max(delta, std::fabs(v_old - v_new));
    }
    if (delta < theta) break;
  }

  return V;
}

Policy policy_improvement(const Vector& V,
                          const Matrix3D& P,
                          const Matrix3D& R,
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
      for (int s2 = 0; s2 < S; ++s2) {
        q += P[s][a][s2] * (R[s][a][s2] + gamma * V[s2]);
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

std::pair<Policy,Vector> policy_iteration(const Matrix3D& P,
                                          const Matrix3D& R,
                                          double gamma,
                                          double theta,
                                          int max_iters)
{
  int S = P.size();
  Policy policy(S, 0);
  Vector V(S, 0.0);

  for (int it = 0; it < max_iters; ++it) {
    V = policy_evaluation(policy, P, R, gamma, theta);
    Policy new_policy = policy_improvement(V, P, R, gamma);
    if (new_policy == policy) {
      std::cout << "Policy converged after " << it+1 << " iterations.\n";
      break;
    }
    policy = std::move(new_policy);
  }
  return {policy, V};
}
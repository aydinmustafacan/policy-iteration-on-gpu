/**
 * @file sparse_policy_iteration.cc
 * @brief Implementation of sparse matrix policy iteration solver
 * 
 * This file provides the complete implementation of the SparsePolicyIteration class,
 * including all core algorithms for policy evaluation, policy improvement, and
 * sparse matrix operations optimized for large-scale MDP solving.
 * 
 * @author Mustafa Can Aydin
 * @date 2025-08-06
 * @version 1.0
 */

#include "sparse_policy_iteration.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unordered_map>

using namespace std;

/**
 * @brief Constructor implementation
 * 
 * Initializes the sparse policy iteration solver with MDP parameters and
 * CSR matrices. Performs basic validation and sets up internal data structures.
 * 
 * @param mdp_data Complete MDP specification including dimensions and sparse matrices
 */
SparsePolicyIteration::SparsePolicyIteration(const JSONReader::MDPData &mdp_data)
  : S(mdp_data.S), A(mdp_data.A), gamma(mdp_data.gamma),
    P_csr(mdp_data.P), R_csr(mdp_data.R) {
  cout << "Initialized sparse solver for " << S << " states, " << A << " actions" << endl;

  // Pre-align rewards to P's CSR structure for fast lookup during Q computation.
  // Build R_aligned_data_ same length/order as P_csr.data.
  const int rows = static_cast<int>(P_csr.indptr.size() - 1);
  R_aligned_data_.assign(P_csr.data.size(), 0.0);
  if (!R_csr.indptr.empty() && !R_csr.indices.empty()) {
    for (int row = 0; row < rows; ++row) {
      int p_beg = P_csr.indptr[row];
      int p_end = P_csr.indptr[row + 1];
      int r_beg = (row < static_cast<int>(R_csr.indptr.size() - 1)) ? R_csr.indptr[row] : 0;
      int r_end = (row < static_cast<int>(R_csr.indptr.size() - 1)) ? R_csr.indptr[row + 1] : 0;
      // Map next_state -> reward for this row
      std::unordered_map<int, double> rmap;
      rmap.reserve(static_cast<size_t>(r_end - r_beg));
      for (int j = r_beg; j < r_end; ++j) rmap.emplace(R_csr.indices[j], R_csr.data[j]);
      // Fill aligned rewards in P's column order
      for (int j = p_beg; j < p_end; ++j) {
        int col = P_csr.indices[j];
        auto it = rmap.find(col);
        R_aligned_data_[j] = (it == rmap.end()) ? 0.0 : it->second;
      }
    }
  } else if (R_csr.indptr.empty() && R_csr.indices.empty() && !R_csr.data.empty()) {
    // R provided as data-only, must match P.nnz
    if (R_csr.data.size() == P_csr.data.size()) {
      R_aligned_data_ = R_csr.data;
    } else {
      std::cerr << "[Warn] R.data length (" << R_csr.data.size()
          << ") does not match P.nnz (" << P_csr.data.size() << ")\n";
    }
  }
}

/**
 * @brief Main policy iteration algorithm implementation
 *
 * Implements the complete policy iteration algorithm with convergence detection.
 * The algorithm alternates between policy evaluation (computing values for current policy)
 * and policy improvement (finding better policy) until convergence.
 *
 * Convergence is detected when the policy no longer changes between iterations,
 * which guarantees that the optimal policy has been found (for finite MDPs).
 *
 * @param max_iters Maximum number of policy iterations allowed
 * @return Tuple containing (optimal_policy, optimal_values, iterations_used)
 */
tuple<Policy, Vector, int> SparsePolicyIteration::solve(int max_iters) {
  cout << "Starting sparse policy iteration..." << endl;

  // Initialize policy (all states start with action 0)
  // This is a common initialization strategy, though any valid policy works
  Policy policy(S, 0);

  // Initialize value function to zero
  // Values will be computed properly in first policy evaluation
  Vector V(S, 0.0); // warm-start across outer iterations

  // Track number of iterations for performance analysis
  int final_iterations = 0;
  bool pi_converged = false;

  // Metrics
  int policy_improvements = 0;
  long long eval_sweeps_total = 0;
  int eval_sweeps_max = 0;
  // no residual tracking

  const double tie_eps = 1e-20;

  // Main policy iteration loop
  for (int iter = 0; iter < max_iters; ++iter) {
    // cout << "Policy iteration " << iter + 1 << "..." << endl;

    int sweeps = 0;
    V = policy_evaluation_sparse(
      policy,
      /*theta*/ 1e-5,
      /*max_iters*/ 1000,
      &sweeps);
    eval_sweeps_total += sweeps;
    eval_sweeps_max = std::max(eval_sweeps_max, sweeps);

    // STEP 2: Policy Improvement
    // Compute greedy policy π'(s) = argmax_a Q^π(s,a) where
    // Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
    Policy new_policy = policy_improvement_sparse(V);

    // STEP 3: Convergence Check
    // If policy didn't change, we've found the optimal policy
    // This is the standard convergence criterion for policy iteration
    // Stabilize on ties: keep current action unless the new action improves Q by > tie_eps
    int changes = 0;
    for (int s = 0; s < S; ++s) {
      // current action of the existing policy
      int a_curr = policy[s];
      // new action from the improved policy
      int a_new = new_policy[s];
      // compare the new action with the current one
      // If the new action is different, we need to check if it improves Q-value
      // If it does not improve by more than tie_eps, keep the current action
      // This prevents unnecessary policy oscillations on tiny improvements
      if (a_new != a_curr) {
        double q_curr = compute_state_value(s, a_curr, V);
        double q_new = compute_state_value(s, a_new, V);
        if (q_new <= q_curr + tie_eps) {
          new_policy[s] = a_curr; // don’t flip on tiny/zero improvement
        }
      }
      if (new_policy[s] != policy[s]) ++changes;
    }

    if (changes == 0) {
      // policy stable
      final_iterations = iter + 1;
      pi_converged = true;
      break;
    }

    // Update policy for next iteration
    policy = std::move(new_policy);
    final_iterations = iter + 1;
    policy_improvements = final_iterations;
  }

  if (!pi_converged) {
    std::cout << "[Warn] Reached max policy iterations (" << max_iters
        << ") without policy stability (π' != π)." << std::endl;
  }

  // Store metrics
  last_policy_improvements_ = policy_improvements;
  last_eval_sweeps_total_ = eval_sweeps_total;
  last_eval_sweeps_max_ = eval_sweeps_max;
  // no residual stored

  return {policy, V, final_iterations};
}

/**
 * @brief Policy evaluation using sparse Jacobi iteration
 *
 * Solves the linear system V^π = R^π + γP^π V^π iteratively to find the value
 * function for a given policy. Uses the Jacobi method for numerical stability.
 *
 * The Jacobi iteration formula is:
 * V^(k+1)(s) = Σ_s' P^π(s'|s)[R^π(s,s') + γV^(k)(s')]
 *
 * where P^π(s'|s) = P(s'|s,π(s)) and R^π(s,s') = R(s,π(s),s')
 *
 * @param policy Current policy π(s) for which to evaluate values
 * @param theta Convergence tolerance - stops when ||V^(k+1) - V^(k)||_∞ < theta
 * @param max_iters Maximum number of Jacobi iterations
 * @return Vector containing the value function V^π(s) for the given policy
 */
Vector SparsePolicyIteration::policy_evaluation_sparse(const Policy &policy, double theta, int max_iters,
                                                       int *sweeps_out) {
  Vector V_old(S, 0.0);
  Vector V_new(S, 0.0);

  int iter = 0;
  for (; iter < max_iters; ++iter) {
    double delta = 0.0;
    for (int s = 0; s < S; ++s) {
      V_new[s] = compute_state_value(s, policy[s], V_old);
      delta = std::max(delta, std::abs(V_new[s] - V_old[s]));
    }
    if (delta < theta) break;
    V_old.swap(V_new);
  }
  // If we broke early (converged), V_new holds the latest values.
  if (iter < max_iters) {
    V_old.swap(V_new);
  }
  if (sweeps_out) *sweeps_out = iter + 1;
  return V_old;
}

/**
 * @brief Policy improvement using sparse Q-value computation
 *
 * Computes the greedy policy improvement step by evaluating Q-values for all
 * state-action pairs and selecting the action with maximum Q-value in each state.
 *
 * For each state s, computes:
 * Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
 * π'(s) = argmax_a Q(s,a)
 *
 * @param V Current value function V(s) from policy evaluation
 * @return Improved policy π'(s) that is greedy with respect to V
 */
Policy SparsePolicyIteration::policy_improvement_sparse(const Vector &V) {
  Policy new_policy(S, 0); // Initialize new policy (default action 0 for all states)

  // For each state, find the action that maximizes Q-value
  for (int s = 0; s < S; ++s) {
    double best_value = -numeric_limits<double>::infinity(); // Track best Q-value found
    int best_action = 0; // Track action that achieves best Q-value

    // Evaluate all possible actions in state s
    for (int a = 0; a < A; ++a) {
      // Compute Q-value: Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
      // This uses sparse matrix operations to avoid zero entries
      double q_value = compute_state_value(s, a, V);

      // GREEDY SELECTION: Choose action with highest Q-value
      // Ties are broken by selecting the first (lowest index) action encountered
      if (q_value > best_value) {
        best_value = q_value;
        best_action = a;
      }
    }

    // Set the greedy action for this state
    new_policy[s] = best_action;
  }

  return new_policy;
}

/**
 * @brief Core sparse matrix computation for state-action values
 *
 * This is the computational kernel that efficiently computes Q-values using
 * the sparse CSR matrix representation. It implements the Bellman equation:
 *
 * Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
 *
 * Key optimizations:
 * - Uses CSR indptr to locate relevant matrix entries for (s,a) pair
 * - Only iterates over non-zero transition probabilities
 * - Separately looks up rewards from sparse reward matrix
 * - Avoids O(S) iteration by using sparsity structure
 *
 * @param state State index s (0 ≤ s < S)
 * @param action Action index a (0 ≤ a < A)
 * @param V Value function V(s') for all states s'
 * @return Q-value Q(s,a) computed using sparse operations
 */
double SparsePolicyIteration::compute_state_value(int state, int action, const Vector &V) {
  double value = 0.0; // Accumulator for Q-value computation

  // COMPUTE STATE-ACTION INDEX in flattened CSR representation
  // CSR matrices store entries for all (state, action) pairs
  // Index mapping: (s,a) → s*A + a
  int state_action_idx = state * A + action;

  // BOUNDARY CHECK: Ensure state-action pair exists in transition matrix
  // If index is out of bounds, no transitions are defined (absorbing state-action)
  if (state_action_idx >= static_cast<int>(P_csr.indptr.size() - 1)) {
    return 0.0; // No transitions defined, return zero value
  }

  // GET TRANSITION RANGE from CSR indptr array
  // P_csr.indptr[i] gives start index for row i in the data/indices arrays
  // P_csr.indptr[i+1] gives end index (exclusive) for row i
  int p_start_idx = P_csr.indptr[state_action_idx];
  int p_end_idx = P_csr.indptr[state_action_idx + 1];

  // ITERATE OVER NON-ZERO TRANSITIONS for efficient computation
  // Only process entries where P(s'|s,a) > 0
  for (int p_idx = p_start_idx; p_idx < p_end_idx; ++p_idx) {
    // Extract transition information from CSR arrays
    int next_state = P_csr.indices[p_idx]; // Next state s'
    double probability = P_csr.data[p_idx]; // Transition probability P(s'|s,a)

    // Use pre-aligned reward corresponding to this P entry if available
    double reward = (!R_aligned_data_.empty()) ? R_aligned_data_[p_idx] : 0.0;

    // ACCUMULATE BELLMAN EQUATION TERM
    // Add P(s'|s,a)[R(s,a,s') + γV(s')] to the Q-value
    value += probability * (reward + gamma * V[next_state]);
  }

  return value;
}

/**
 * @brief Extract transition information for debugging and analysis
 *
 * Utility function that extracts all transition information for a given
 * state-action pair. Useful for debugging, policy analysis, and visualization.
 *
 * @param state Source state s
 * @param action Action a taken in state s
 * @param[out] next_states Vector of reachable next states s'
 * @param[out] probabilities Vector of transition probabilities P(s'|s,a)
 * @param[out] rewards Vector of immediate rewards R(s,a,s')
 */
void SparsePolicyIteration::get_state_action_transitions(int state, int action,
                                                         vector<int> &next_states,
                                                         vector<double> &probabilities,
                                                         vector<double> &rewards) {
  // Clear output vectors to ensure clean state
  next_states.clear();
  probabilities.clear();
  rewards.clear();

  // Compute state-action index for CSR matrix lookup
  int state_action_idx = state * A + action;

  // Check if state-action pair has any transitions defined
  if (state_action_idx >= static_cast<int>(P_csr.indptr.size() - 1)) {
    return; // No transitions defined, return empty vectors
  }

  // Get CSR index ranges for transitions and rewards
  int p_start_idx = P_csr.indptr[state_action_idx];
  int p_end_idx = P_csr.indptr[state_action_idx + 1];

  int r_start_idx = (state_action_idx < static_cast<int>(R_csr.indptr.size() - 1)) ? R_csr.indptr[state_action_idx] : 0;
  int r_end_idx = (state_action_idx < static_cast<int>(R_csr.indptr.size() - 1))
                    ? R_csr.indptr[state_action_idx + 1]
                    : 0;

  // Extract all transitions for this state-action pair
  for (int p_idx = p_start_idx; p_idx < p_end_idx; ++p_idx) {
    int next_state = P_csr.indices[p_idx];
    next_states.push_back(next_state);
    probabilities.push_back(P_csr.data[p_idx]);

    // Find corresponding reward for this specific transition
    double reward = 0.0;
    for (int r_idx = r_start_idx; r_idx < r_end_idx; ++r_idx) {
      if (R_csr.indices[r_idx] == next_state) {
        reward = R_csr.data[r_idx];
        break;
      }
    }
    rewards.push_back(reward);
  }
}

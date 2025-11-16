/**
 * @file sparse_policy_iteration.h
 * @brief Sparse Matrix Policy Iteration Solver for Markov Decision Processes
 * 
 * This file implements a high-performance policy iteration algorithm specifically
 * optimized for sparse MDPs using Compressed Sparse Row (CSR) matrix format.
 * The solver is designed to handle large-scale problems efficiently by leveraging
 * sparsity patterns in transition and reward matrices.
 * 
 * @author Mustafa Can Aydin
 */

#ifndef SPARSE_POLICY_ITERATION_H
#define SPARSE_POLICY_ITERATION_H

#include <vector>
#include <tuple>
#include <limits>
#include "json_reader.h"

/**
 * @brief Type alias for value function representation
 * 
 * Represents the value function V(s) for all states s in the MDP.
 * Each element V[s] contains the expected discounted cumulative reward
 * when starting from state s and following the optimal policy.
 */
using Vector = std::vector<double>;

/**
 * @brief Type alias for policy representation
 * 
 * Represents a deterministic policy π(s) for all states s in the MDP.
 * Each element Policy[s] contains the action index to take in state s.
 * Actions are represented as integers (0, 1, 2, ..., A-1).
 */
using Policy = std::vector<int>;

/**
 * @class SparsePolicyIteration
 * @brief High-performance sparse matrix policy iteration solver for MDPs
 * 
 * This class implements the policy iteration algorithm optimized for sparse
 * Markov Decision Processes. It uses Compressed Sparse Row (CSR) format for
 * efficient storage and computation of large, sparse transition and reward matrices.
 * 
 * The algorithm alternates between:
 * 1. Policy Evaluation: Computing value function V^π for current policy π
 * 2. Policy Improvement: Finding greedy policy π' with respect to V^π
 * 
 * Key optimizations:
 * - CSR matrix format for O(nnz) operations instead of O(S²A)
 * - Jacobi iteration for policy evaluation with configurable convergence
 * - Efficient sparse matrix-vector operations
 * - Memory-efficient storage for large state spaces
 * 
 * @note This solver is specifically designed for problems where the transition
 *       and reward matrices are sparse (many zero entries), such as gridworld
 *       navigation, network routing, or resource allocation problems.
 */
class SparsePolicyIteration {
public:
  /**
   * @brief Construct sparse policy iteration solver from MDP data
   *
   * Initializes the solver with pre-parsed MDP data in CSR format.
   * The constructor copies the CSR matrices and problem parameters
   * for efficient access during solving.
   *
   * @param mdp_data Structured MDP data containing:
   *                 - S: Number of states
   *                 - A: Number of actions
   *                 - gamma: Discount factor (0 < gamma <= 1)
   *                 - P: Transition probability matrix in CSR format
   *                 - R: Reward matrix in CSR format
   *
   * @pre mdp_data.S > 0 && mdp_data.A > 0
   * @pre 0 < mdp_data.gamma <= 1.0
   * @pre P and R matrices must have consistent dimensions
   *
   * @post Solver is ready to solve the MDP via solve() method
   */
  SparsePolicyIteration(const JSONReader::MDPData &mdp_data);

  /**
   * @brief Solve the MDP using policy iteration algorithm
   *
   * Executes the complete policy iteration algorithm to find the optimal
   * policy and value function. The algorithm guarantees convergence to
   * the optimal solution for finite MDPs with proper discount factors.
   *
   * Algorithm outline:
   * 1. Initialize policy π₀ (all actions = 0)
   * 2. Repeat until convergence:
   *
   *    a. Policy Evaluation: Solve
   *    \f[
   *     V^{\pi} = R^{\pi} + \gamma P^{\pi} V^{\pi}
   *    \f]
   *    b. Policy Improvement: 
   *    \f[
   *     \pi'(s) = \arg\max_{a} \sum_{s'} P(s' \mid s, a)\,\bigl[ R(s,a,s') + \gamma \, V^{\pi}(s') \bigr]
   *    \f]
   *    c. If \f$ \pi^{'} = \pi \f$, then convergence achieved
   *
   * @param max_iters Maximum number of policy iterations (default: 5000)
   *
   * @return std::tuple containing:
   *         - Policy: Optimal policy \f$\pi^{*}(s)\f$ for each state
   *         - Vector: Optimal value function \f$V^{*}(s)\f$ for each state
   *         - int: Number of iterations until convergence
   *
   * @pre max_iters > 0
   * @post Returned policy is optimal (\f$\epsilon\f$-optimal within numerical precision)
   *
   * @par Complexity Time: O(iterations × nnz × eval_iterations)
   *            Space: \f$O(S + nnz)\f$
   *            where nnz = number of non-zero entries in \f$P\f$ and \f$R\f$ matrices
   *
   * @note For well-conditioned problems, typically converges in 10-50 iterations
   */
  std::tuple<Policy, Vector, int> solve(int max_iters = 5000);

  // Metrics captured from the last solve()
  int last_policy_improvements() const { return last_policy_improvements_; }
  long long last_eval_sweeps_total() const { return last_eval_sweeps_total_; }
  int last_eval_sweeps_max() const { return last_eval_sweeps_max_; }

  /**
   * @brief Get number of states in the MDP
   * @return Number of states S
   */
  int getNumStates() const { return S; }

  /**
   * @brief Get number of actions in the MDP
   * @return Number of actions A
   */
  int getNumActions() const { return A; }

  /**
   * @brief Get discount factor of the MDP
   * @return Discount factor γ (gamma)
   */
  double getGamma() const { return gamma; }

private:
  // Problem parameters
  int S; ///< Number of states in the MDP
  int A; ///< Number of actions available in each state
  double gamma; ///< Discount factor (0 < γ <= 1)

  // CSR matrices for efficient sparse operations
  JSONReader::CSRMatrix P_csr; ///< Transition probability matrix P(s'|s,a) in CSR format
  JSONReader::CSRMatrix R_csr; ///< Reward matrix R(s,a,s') in CSR format

  /**
   * @brief Perform policy evaluation using sparse Jacobi iteration
   *
   * Solves the linear system V^π = R^π + γP^π V^π iteratively using
   * the Jacobi method and returns the converged value function for
   * the given policy.
   *
   * The Jacobi update rule is:
   * V^(k+1)(s) = Σ_a π(s,a) Σ_s' P(s'|s,a)[R(s,a,s') + γV^(k)(s')]
   *
   * @param policy Current policy π(s) for all states
   * @param theta  Convergence tolerance (default: 1e-5)
   * @param max_iters Maximum Jacobi iterations (default: 50000)
   *
   * @pre policy.size() == S
   * @pre theta > 0
   * @pre max_iters > 0
   *
   * @return Value function V^π(s) for the given policy
   */
  // Policy evaluation with optional sweep count output
  Vector policy_evaluation_sparse(const Policy &policy,
                                  double theta = 1e-5,
                                  int max_iters = 50000,
                                  int *sweeps_out = nullptr);

  /**
   * @brief Perform policy improvement using sparse Q-value computation
   *
   * Computes the greedy policy with respect to the current value function
   * by evaluating Q-values for all state-action pairs and selecting the
   * action with maximum Q-value in each state.
   *
   * Q-value computation:
   * Q^π(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV^π(s')]
   *
   * Policy improvement:
   * π'(s) = argmax_a Q^π(s,a)
   *
   * @param V Current value function V^π(s)
   *
   * @return Improved policy π'(s) that is greedy with respect to V
   *
   * @pre V.size() == S
   * @post π'(s) = argmax_a Q^π(s,a) for all states s
   *
   * @complexity Time: O(S × A × avg_transitions_per_state)
   *            Space: O(S)
   *
   * @note Ties in Q-values are broken by selecting the first (lowest index) action
   */
  Policy policy_improvement_sparse(const Vector &V);

  /**
   * @brief Compute state-action value using sparse matrix operations
   *
   * Efficiently computes the Q-value Q(s,a) = Σ_s' P(s'|s,a)[R(s,a,s') + γV(s')]
   * using CSR matrix format to avoid iterating over zero entries.
   *
   * This is the core computational kernel that leverages sparsity:
   * - Uses CSR indptr to find relevant transitions for (s,a)
   * - Only processes non-zero transition probabilities
   * - Looks up corresponding rewards in sparse reward matrix
   *
   * @param state State index s (0 ≤ s < S)
   * @param action Action index a (0 ≤ a < A)
   * @param V Value function V(s') for all states s'
   *
   * @return Q-value Q(s,a) = expected discounted reward for taking action a in state s
   *
   * @pre 0 <= state < S
   * @pre 0 <= action < A
   * @pre V.size() == S
   *
   * @complexity Time: O(transitions_from_state_action)
   *            Space: O(1)
   *
   * @note Returns 0.0 if no transitions are defined for the state-action pair
   */
  double compute_state_value(int state, int action, const Vector &V);

  /**
   * @brief Extract transition information for a state-action pair
   *
   * Retrieves all possible transitions from a given state-action pair,
   * including next states, transition probabilities, and immediate rewards.
   * Used for debugging, analysis, and policy visualization.
   *
   * @param state Source state s
   * @param action Action a taken in state s
   * @param[out] next_states Vector of reachable next states s'
   * @param[out] probabilities Vector of transition probabilities P(s'|s,a)
   * @param[out] rewards Vector of immediate rewards R(s,a,s')
   *
   * @pre 0 <= state < S
   * @pre 0 <= action < A
   * @post next_states.size() == probabilities.size() == rewards.size()
   * @post Σ probabilities ≈ 1.0 (stochastic constraint)
   *
   * @note Output vectors are cleared before populating
   * @note If no transitions exist, all output vectors will be empty
   */
  void get_state_action_transitions(int state, int action,
                                    std::vector<int> &next_states,
                                    std::vector<double> &probabilities,
                                    std::vector<double> &rewards);

private:
  // Metrics
  int last_policy_improvements_ = 0;
  long long last_eval_sweeps_total_ = 0;
  int last_eval_sweeps_max_ = 0;
  // (No residual stored; using default evaluation params inside implementation)

  // Reward data aligned to P's CSR structure (size equals P.nnz)
  std::vector<double> R_aligned_data_;
};

#endif // SPARSE_POLICY_ITERATION_H

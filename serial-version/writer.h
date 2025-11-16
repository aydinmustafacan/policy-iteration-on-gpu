/**
 * @file writer.h
 * @brief CSV and text file output utilities for MDP solutions and analysis
 * 
 * This header defines utilities for writing Markov Decision Process (MDP)
 * solutions, analysis results, and performance metrics to various file formats.
 * Supports both human-readable and machine-readable output formats for
 * result validation, comparison, and further analysis.
 * 
 * Output Formats Supported:
 * - Policy CSV: "state,action" format for deterministic policies
 * - Value CSV: "state,value" format for value functions
 * - Combined CSV: "state,action,value" format for complete solutions
 * - MDP CSV: Separate files for transitions and rewards in sparse format
 * - Summary TXT: Human-readable analysis reports with statistics
 * - Metrics TXT: Performance measurements and convergence data
 * 
 * Key Features:
 * - Consistent CSV formatting with headers
 * - High-precision floating point output for numerical accuracy
 * - Automatic file handling with error reporting
 * - Sparse format output for large MDPs
 * - Performance metrics tracking and reporting
 * 
 * Usage Patterns:
 * 1. Save optimization results for comparison: write_policy_and_values_to_csv()
 * 2. Export MDP specifications: write_mdp_to_csv()
 * 3. Generate analysis reports: write_results_summary()
 * 4. Track performance metrics: write_performance_metrics()
 * 
 * @version 1.0
 */

#ifndef PROJECT_CODE_WRITER_H
#define PROJECT_CODE_WRITER_H

#include <vector>
#include <string>

/**
 * @brief Type alias for 3D transition/reward matrices
 * 
 * Matrix3D[s][a][s'] represents either:
 * - Transition probabilities P(s'|s,a) when used for transitions
 * - Immediate rewards R(s,a,s') when used for rewards
 * 
 * Dimensions: [num_states][num_actions][num_states]
 */
using Matrix3D = std::vector<std::vector<std::vector<double> > >;

/**
 * @brief Type alias for value function vectors
 * 
 * Vector[s] represents the value V(s) for state s
 * Dimension: [num_states]
 */
using Vector = std::vector<double>;

/**
 * @brief Type alias for deterministic policy vectors
 * 
 * Policy[s] represents the action π(s) selected in state s
 * Dimension: [num_states]
 */
using Policy = std::vector<int>;

/**
 * @brief CSV and text file writer for MDP solutions and analysis
 * 
 * Static utility class providing functions to write various MDP components
 * and analysis results to files. Handles automatic formatting, precision
 * control, and error reporting for robust file output operations.
 * 
 * Design Philosophy:
 * - All methods are static for utility-style usage
 * - Consistent CSV formatting across all output functions
 * - High precision floating point output for numerical accuracy
 * - Comprehensive error handling with boolean return values
 * - Human-readable formats for analysis and debugging
 * 
 * Output Standards:
 * - CSV files include descriptive headers
 * - Floating point values use sufficient precision for reproducibility
 * - Sparse format output for efficiency with large MDPs
 * - Text files use clear formatting for readability
 */
class writer {
public:
  /**
   * @brief Write deterministic policy to CSV file
   * 
   * Outputs a policy specification in standard CSV format:
   * state,action
   * 0,1
   * 1,0
   * 2,2
   * ...
   * 
   * Each row represents the action selected by the policy for that state.
   * States are output in ascending order from 0 to |S|-1.
   * 
   * @param policy Policy vector to write
   * @param filename Output CSV file path
   * @return true if write successful, false on error
   * 
   * @complexity Time: O(|S|) for writing all states
   * @complexity Space: O(1) additional space (streaming output)
   */
  static bool write_policy_to_csv(const Policy &policy, const std::string &filename);

  /**
   * @brief Write value function to CSV file
   * 
   * Outputs state values in standard CSV format:
   * state,value
   * 0,10.5
   * 1,8.3
   * 2,15.7
   * ...
   * 
   * Values are output with high precision to ensure numerical accuracy
   * for subsequent analysis and comparison.
   * 
   * @param values Value vector to write
   * @param filename Output CSV file path
   * @return true if write successful, false on error
   * 
   * @complexity Time: O(|S|) for writing all state values
   * @complexity Space: O(1) additional space (streaming output)
   */
  static bool write_values_to_csv(const Vector &values, const std::string &filename);

  /**
   * @brief Write complete MDP solution to single CSV file
   * 
   * Combines policy and value function in unified format:
   * state,action,value
   * 0,1,10.5
   * 1,0,8.3
   * 2,2,15.7
   * ...
   * 
   * This format is convenient for analysis tools that need both
   * policy and value information together.
   * 
   * @param policy Policy vector to write
   * @param values Value vector to write (must have same size as policy)
   * @param filename Output CSV file path
   * @return true if write successful, false on error
   * 
   * @complexity Time: O(|S|) for writing all states
   * @complexity Space: O(1) additional space (streaming output)
   */
  static bool write_policy_and_values_to_csv(const Policy &policy, const Vector &values,
                                             const std::string &filename);

  /**
   * @brief Write MDP specification to separate CSV files
   * 
   * Exports transition and reward matrices in sparse CSV format suitable
   * for re-importing or sharing MDP specifications. Only non-zero entries
   * are written to minimize file size.
   * 
   * Transition File Format:
   * state,action,next_state,probability
   * 0,0,0,0.8
   * 0,0,1,0.2
   * ...
   * 
   * Reward File Format:
   * state,action,next_state,reward
   * 0,0,0,-1.0
   * 0,0,1,10.0
   * ...
   * 
   * @param P Transition probability matrix to export
   * @param R Immediate reward matrix to export
   * @param transition_file Path for transition CSV output
   * @param reward_file Path for reward CSV output
   * @return true if both files written successfully, false on any error
   * 
   * @complexity Time: O(|S|²|A|) for scanning all matrix entries
   * @complexity Space: O(1) additional space (streaming output)
   */
  static bool write_mdp_to_csv(const Matrix3D &P, const Matrix3D &R,
                               const std::string &transition_file,
                               const std::string &reward_file);

  /**
   * @brief Write comprehensive results summary to text file
   * 
   * Generates a human-readable analysis report containing:
   * - MDP problem dimensions and parameters
   * - Optimal policy and value function
   * - Convergence statistics and performance metrics
   * - Summary statistics (max/min/mean values)
   * 
   * This format is ideal for documentation, reports, and debugging.
   * 
   * @param policy Optimal policy found
   * @param values Optimal value function
   * @param gamma Discount factor used
   * @param iterations Number of policy iterations required
   * @param elapsed_ms Computation time in milliseconds
   * @param filename Output text file path
   * @return true if write successful, false on error
   * 
   * @complexity Time: O(|S|) for computing statistics and writing results
   * @complexity Space: O(1) additional space (streaming output)
   */
  static bool write_results_summary(const Policy &policy, const Vector &values,
                                    double gamma, int iterations, double elapsed_ms,
                                    const std::string &filename);

  /**
   * @brief Write performance metrics to separate file
   * 
   * Outputs structured performance data for benchmarking and comparison:
   * - Algorithm parameters (discount factor)
   * - Convergence metrics (iterations, time)
   * - Derived metrics (time per iteration, convergence rate)
   * 
   * Format is suitable for automated processing and statistical analysis.
   * 
   * @param gamma Discount factor used in optimization
   * @param iterations Number of policy iterations required
   * @param elapsed_ms Total computation time in milliseconds
   * @param filename Output metrics file path
   * @return true if write successful, false on error
   * 
   * @complexity Time: O(1) for computing and writing metrics
   * @complexity Space: O(1) additional space
   */
  static bool write_performance_metrics(double gamma, int iterations, double elapsed_ms,
                                        const std::string &filename);
};

#endif //PROJECT_CODE_WRITER_H

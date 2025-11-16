/**
 * @file writer.cc
 * @brief Implementation of CSV and text file output utilities for MDP solutions
 * 
 * This file implements the file output functionality defined in writer.h.
 * Provides comprehensive output capabilities for MDP solutions, analysis results,
 * and performance metrics in various formats suitable for analysis and comparison.
 * 
 * Implementation Features:
 * - High-precision floating point output for numerical accuracy
 * - Automatic file handling with comprehensive error checking
 * - Sparse format output for efficient large MDP storage
 * - Consistent CSV formatting with descriptive headers
 * - Human-readable summary reports for analysis
 * 
 * Output Standards:
 * - CSV files use standard comma-separated format with headers
 * - Floating point values output with 6 decimal places for reproducibility
 * - Sparse matrices only output non-zero entries for efficiency
 * - Text files use clear formatting for readability and parsing
 * 
 * @version 1.0
 */

#include "writer.h"
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;

/**
 * @brief Write deterministic policy to CSV file with standard formatting
 * 
 * Outputs a policy specification in standard two-column CSV format with
 * descriptive header. Each row represents the action selected by the policy
 * for the corresponding state index.
 * 
 * Output Format:
 * state,action
 * 0,1
 * 1,0
 * 2,2
 * ...
 * 
 * Error Handling:
 * - Validates file can be opened for writing
 * - Reports success/failure with informative messages
 * - Properly closes file resources
 * 
 * @param policy Policy vector to write to file
 * @param filename Output CSV file path
 * @return true if write operation successful, false on any error
 */
bool writer::write_policy_to_csv(const Policy &policy, const string &filename) {
  ofstream file(filename);

  // FILE VALIDATION: Check if file can be opened for writing
  if (!file.is_open()) {
    cerr << "Error: Could not open file " << filename << " for writing" << endl;
    return false;
  }

  // HEADER OUTPUT: Write descriptive column headers
  file << "state,action\n";

  // DATA OUTPUT: Write each state-action pair
  for (int s = 0; s < static_cast<int>(policy.size()); ++s) {
    file << s << "," << policy[s] << "\n";
  }

  file.close();
  cout << "Successfully wrote policy to " << filename << endl;
  return true;
}

/**
 * @brief Write value function to CSV file with high precision
 *
 * Outputs state values in standard two-column CSV format with fixed precision
 * to ensure numerical accuracy for subsequent analysis and comparison.
 * Uses 6 decimal places to balance precision with readability.
 *
 * Output Format:
 * state,value
 * 0,10.500000
 * 1,8.300000
 * 2,15.700000
 * ...
 *
 * @param values Value vector to write to file
 * @param filename Output CSV file path
 * @return true if write operation successful, false on any error
 */
bool writer::write_values_to_csv(const Vector &values, const string &filename) {
  ofstream file(filename);

  // FILE VALIDATION: Check if file can be opened for writing
  if (!file.is_open()) {
    cerr << "Error: Could not open file " << filename << " for writing" << endl;
    return false;
  }

  // HEADER OUTPUT: Write descriptive column headers
  file << "state,value\n";

  // PRECISION SETTING: Configure high-precision floating point output
  file << fixed << setprecision(6);

  // DATA OUTPUT: Write each state-value pair with fixed precision
  for (int s = 0; s < static_cast<int>(values.size()); ++s) {
    file << s << "," << values[s] << "\n";
  }

  file.close();
  cout << "Successfully wrote values to " << filename << endl;
  return true;
}

/**
 * @brief Write complete MDP solution to unified CSV file
 *
 * Combines policy and value function information in a single three-column
 * CSV file. This format is convenient for analysis tools that need both
 * policy and value information together.
 *
 * Validation:
 * - Ensures policy and value vectors have identical sizes
 * - Reports size mismatch errors clearly
 *
 * Output Format:
 * state,action,value
 * 0,1,10.500000
 * 1,0,8.300000
 * 2,2,15.700000
 * ...
 *
 * @param policy Policy vector to write (must match values size)
 * @param values Value vector to write (must match policy size)
 * @param filename Output CSV file path
 * @return true if write operation successful, false on any error
 */
bool writer::write_policy_and_values_to_csv(const Policy &policy, const Vector &values,
                                            const string &filename) {
  // SIZE VALIDATION: Ensure policy and values have consistent dimensions
  if (policy.size() != values.size()) {
    cerr << "Error: Policy and values must have the same size" << endl;
    return false;
  }

  ofstream file(filename);

  // FILE VALIDATION: Check if file can be opened for writing
  if (!file.is_open()) {
    cerr << "Error: Could not open file " << filename << " for writing" << endl;
    return false;
  }

  // HEADER OUTPUT: Write descriptive column headers
  file << "state,action,value\n";

  // PRECISION SETTING: Configure high-precision floating point output
  file << fixed << setprecision(6);

  // DATA OUTPUT: Write each state with both action and value
  for (int s = 0; s < static_cast<int>(policy.size()); ++s) {
    file << s << "," << policy[s] << "," << values[s] << "\n";
  }

  file.close();
  cout << "Successfully wrote policy and values to " << filename << endl;
  return true;
}

/**
 * @brief Write MDP specification to separate CSV files in sparse format
 *
 * Exports transition and reward matrices to separate CSV files using
 * sparse representation. Only non-zero entries are written to minimize
 * file size and improve efficiency for large sparse MDPs.
 *
 * Transition File Output:
 * state,action,next_state,probability
 * 0,0,0,0.800000
 * 0,0,1,0.200000
 * ...
 *
 * Reward File Output:
 * state,action,next_state,reward
 * 0,0,0,-1.000000
 * 0,0,1,10.000000
 * ...
 *
 * Sparse Optimization:
 * - Transitions: Only entries with probability > 0.0 are written
 * - Rewards: Only entries with reward ≠ 0.0 are written
 * - Significant file size reduction for sparse problems
 *
 * @param P Transition probability matrix to export
 * @param R Immediate reward matrix to export
 * @param transition_file Path for transition CSV output
 * @param reward_file Path for reward CSV output
 * @return true if both files written successfully, false on any error
 */
bool writer::write_mdp_to_csv(const Matrix3D &P, const Matrix3D &R,
                              const string &transition_file,
                              const string &reward_file) {
  // TRANSITION MATRIX OUTPUT
  ofstream trans_file(transition_file);

  // FILE VALIDATION: Check if transition file can be opened
  if (!trans_file.is_open()) {
    cerr << "Error: Could not open file " << transition_file << " for writing" << endl;
    return false;
  }

  // TRANSITION HEADER: Write descriptive column headers
  trans_file << "state,action,next_state,probability\n";
  trans_file << fixed << setprecision(6);

  // SPARSE TRANSITION OUTPUT: Only write non-zero probabilities
  for (int s = 0; s < static_cast<int>(P.size()); ++s) {
    for (int a = 0; a < static_cast<int>(P[s].size()); ++a) {
      for (int s2 = 0; s2 < static_cast<int>(P[s][a].size()); ++s2) {
        if (P[s][a][s2] > 0.0) {
          // Sparse optimization: skip zero probabilities
          trans_file << s << "," << a << "," << s2 << "," << P[s][a][s2] << "\n";
        }
      }
    }
  }
  trans_file.close();

  // REWARD MATRIX OUTPUT
  ofstream reward_file_stream(reward_file);

  // FILE VALIDATION: Check if reward file can be opened
  if (!reward_file_stream.is_open()) {
    cerr << "Error: Could not open file " << reward_file << " for writing" << endl;
    return false;
  }

  // REWARD HEADER: Write descriptive column headers
  reward_file_stream << "state,action,next_state,reward\n";
  reward_file_stream << fixed << setprecision(6);

  // SPARSE REWARD OUTPUT: Only write non-zero rewards
  for (int s = 0; s < static_cast<int>(R.size()); ++s) {
    for (int a = 0; a < static_cast<int>(R[s].size()); ++a) {
      for (int s2 = 0; s2 < static_cast<int>(R[s][a].size()); ++s2) {
        if (R[s][a][s2] != 0.0) {
          // Sparse optimization: skip zero rewards
          reward_file_stream << s << "," << a << "," << s2 << "," << R[s][a][s2] << "\n";
        }
      }
    }
  }
  reward_file_stream.close();

  cout << "Successfully wrote MDP to " << transition_file << " and " << reward_file << endl;
  return true;
}

/**
 * @brief Write comprehensive results summary in CUDA-compatible format
 *
 * Generates a human-readable summary report containing the optimal policy
 * and value function in a format that matches the CUDA implementation output.
 * This ensures compatibility with comparison tools and maintains consistent
 * formatting across different solvers.
 *
 * Output Format:
 * Optimal policy: 1 0 2 1 0 ...
 * Optimal value:  10.500000 8.300000 15.700000 ...
 *
 * Format Compatibility:
 * - Matches CUDA version output format exactly
 * - Space-separated values for easy parsing
 * - Fixed precision for numerical consistency
 *
 * @param policy Optimal policy found by algorithm
 * @param values Optimal value function computed
 * @param gamma Discount factor used (included for completeness)
 * @param iterations Number of policy iterations required
 * @param elapsed_ms Computation time in milliseconds
 * @param filename Output text file path
 * @return true if write operation successful, false on any error
 */
bool writer::write_results_summary(const Policy &policy, const Vector &values,
                                   double gamma, int iterations, double elapsed_ms,
                                   const string &filename) {
  ofstream file(filename);

  // FILE VALIDATION: Check if file can be opened for writing
  if (!file.is_open()) {
    cerr << "Error: Could not open file " << filename << " for writing" << endl;
    return false;
  }

  // POLICY OUTPUT: Write in CUDA-compatible format with space separation
  file << "Optimal policy: ";
  for (int a: policy) file << a << " ";
  file << "\nOptimal value:  ";

  // VALUE OUTPUT: Write with fixed precision for numerical consistency
  file << fixed << setprecision(6);
  for (double v: values) file << v << " ";
  file << "\n";

  file.close();
  cout << "Successfully wrote results summary to " << filename << endl;
  return true;
}

/**
 * @brief Write performance metrics for benchmarking and analysis
 *
 * Outputs structured performance data including timing information,
 * convergence metrics, and algorithm parameters. This format is suitable
 * for automated processing, statistical analysis, and performance comparison.
 *
 * Output Format:
 * --- Performance Metrics ---
 * Data source: External files
 * Gamma (discount factor): 0.900000
 * Policy iterations completed: 15
 * Serial computation time: 125.340000 ms
 * Total CPU time: 125.340000 ms
 * GPU speedup factor: 1.0x
 *
 * Metrics Included:
 * - Algorithm parameters (discount factor)
 * - Convergence information (iterations required)
 * - Timing measurements (computation time)
 * - Comparative metrics (speedup factors)
 *
 * @param gamma Discount factor used in optimization
 * @param iterations Number of policy iterations required for convergence
 * @param elapsed_ms Total computation time in milliseconds
 * @param filename Output metrics file path
 * @return true if write operation successful, false on any error
 */
bool writer::write_performance_metrics(double gamma, int iterations, double elapsed_ms, const string &filename) {
  ofstream file(filename);

  // FILE VALIDATION: Check if file can be opened for writing
  if (!file.is_open()) {
    cerr << "Error: Could not open file " << filename << " for writing" << endl;
    return false;
  }

  // STRUCTURED METRICS OUTPUT: Write performance data in organized format
  file << "--- Performance Metrics ---\n";
  file << "Data source: External files\n";
  file << "Gamma (discount factor): " << gamma << "\n";
  file << "Policy iterations completed: " << iterations << "\n";
  file << "Serial computation time: " << elapsed_ms << " ms\n";
  file << "Total CPU time: " << elapsed_ms << " ms\n";
  file << "GPU speedup factor: 1.0x\n"; // Serial baseline reference

  file.close();
  cout << "Successfully wrote performance metrics to " << filename << endl;
  return true;
}

/**
 * @file main.cc
 * @brief Main entry point for serial MDP policy iteration solver
 * 
 * This program implements a comprehensive Markov Decision Process (MDP) solver
 * using policy iteration algorithms. It supports multiple input formats and
 * automatically selects between dense and sparse solvers based on problem size.
 * 
 * Program Features:
 * - JSON file input with automatic format detection
 * - CSV file input for transitions and rewards
 * - Hardcoded example MDP for testing
 * - Automatic solver selection (dense vs sparse) based on problem size
 * - Comprehensive output including results, metrics, and timing
 * - File output for result analysis and comparison
 * 
 * Command Line Usage:
 * 1. JSON input: ./program_name input.json
 * 2. Default example: ./program_name (no arguments)
 * 
 * Solver Selection Logic:
 * - Problems with > 100 states: Use sparse CSR-based solver
 * - Problems with ≤ 100 states: Use dense matrix solver
 * 
 * Output Files Generated:
 * - optimal_policy.csv: Optimal policy in CSV format
 * - optimal_values.csv: Optimal value function in CSV format
 * - combined_results.csv: Combined policy and values
 * - {name}-serial.txt: Human-readable results summary
 * - {name}-serial-metrics.txt: Performance metrics
 * 
 * @version 1.0
 */

#include "main.h"
#include <iostream>
#include <string>
#include <chrono>
#include <fstream>
#include <iterator>
#include "writer.h"
#include "json_reader.h"
#include "sparse_policy_iteration.h"

using namespace std;

/**
 * @brief Extract meaningful name from file path for output naming
 * 
 * Utility function that extracts a descriptive name from an input file path
 * to use for naming output files. Attempts to extract the parent directory
 * name first (useful for organized dataset structures), falling back to
 * the filename without extension.
 * 
 * Examples:
 * - "/path/to/gridworld/input.json" → "gridworld"
 * - "/data/maze.json" → "maze"
 * - "simple.csv" → "simple"
 * 
 * @param filepath Full path to input file
 * @return Extracted name suitable for output file naming
 */
string extract_name_from_path(const string &filepath) {
  // DIRECTORY EXTRACTION: Try to get parent directory name for organized datasets
  size_t last_slash = filepath.find_last_of('/');
  if (last_slash != string::npos && last_slash > 0) {
    // Get the parent directory name (more descriptive than filename)
    size_t second_last_slash = filepath.find_last_of('/', last_slash - 1);
    if (second_last_slash != string::npos) {
      return filepath.substr(second_last_slash + 1, last_slash - second_last_slash - 1);
    }
  }

  // FILENAME FALLBACK: Extract filename without extension
  string filename = filepath.substr(last_slash + 1);
  size_t dot = filename.find_last_of('.');
  if (dot != string::npos) {
    return filename.substr(0, dot);
  }
  return filename;
}

/**
 * @brief Main program entry point for MDP policy iteration solver
 *
 * Implements comprehensive MDP solving with multiple input formats and
 * automatic solver selection. The program analyzes command line arguments
 * to determine input type and problem characteristics, then selects the
 * most appropriate solving algorithm.
 *
 * Algorithm Flow:
 * 1. ARGUMENT PARSING: Determine input format (JSON, CSV, or default)
 * 2. DATA LOADING: Load MDP specification from appropriate source
 * 3. SOLVER SELECTION: Choose dense or sparse solver based on problem size
 * 4. OPTIMIZATION: Run policy iteration to find optimal policy
 * 5. OUTPUT GENERATION: Save results in multiple formats for analysis
 *
 * Performance Considerations:
 * - Sparse solver for large problems (>100 states) to handle memory efficiently
 * - Dense solver for small problems for simplicity and speed
 * - High-precision timing for performance analysis
 * - Structured output for automated comparison tools
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return 0 on successful completion, 1 on error
 */
int main(int argc, char *argv[]) {
  string output_name = "default";

  if (argc == 2) {
    string filename = argv[1];
    if (filename.substr(filename.find_last_of(".") + 1) == "json") {
      cout << "Reading MDP from JSON file: " << filename << endl;
      ifstream file(filename);
      if (!file.is_open()) {
        cerr << "Could not open file: " << filename << endl;
        return 1;
      }
      string json_content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
      file.close();
      JSONReader::MDPData mdp_data = JSONReader::parse_json(json_content);
      output_name = extract_name_from_path(filename);

      // Initialize sparse solver with MDP data
      SparsePolicyIteration sparse_solver(mdp_data);

      auto start_time = chrono::high_resolution_clock::now();
      auto [policy, optimal_value, iterations] = sparse_solver.solve(1000);
      auto end_time = chrono::high_resolution_clock::now();
      auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
      double elapsed_ms = duration.count() / 1000.0;

      string output_filename = "results/" + output_name + "-serial.txt";
      string metrics_filename = "results/" + output_name + "-serial-metrics.txt";
      writer::write_policy_to_csv(policy, "results/optimal_policy.csv");
      writer::write_values_to_csv(optimal_value, "results/optimal_values.csv");
      writer::write_policy_and_values_to_csv(policy, optimal_value, "results/combined_results.csv");
      writer::write_results_summary(policy, optimal_value, mdp_data.gamma, iterations, elapsed_ms,
                                    output_filename);
      writer::write_performance_metrics(mdp_data.gamma, iterations, elapsed_ms, metrics_filename);

      cout << "\n--- Performance Metrics ---\n";
      cout << "Data source: External files\n";
      cout << "MDP dimensions: " << mdp_data.S << " states, " << mdp_data.A << " actions\n";
      cout << "Gamma (discount factor): " << mdp_data.gamma << "\n";
      cout << "Policy iterations completed: " << iterations << "\n";
      cout << "Policy improvements (outer iters): " << sparse_solver.last_policy_improvements() << "\n";
      cout << "Evaluation sweeps (total/max): "
          << sparse_solver.last_eval_sweeps_total() << " / "
          << sparse_solver.last_eval_sweeps_max() << "\n";
      cout << "Serial computation time: " << elapsed_ms << " ms\n";
      cout << "Serial computation time: " << elapsed_ms / 1000.0 << " seconds\n";
      cout << "Time per iteration: " << elapsed_ms / iterations << " ms\n";
      return 0;
    } else {
      cerr << "Single file provided but not JSON format. Please provide a JSON file." << endl;
      return 1;
    }
  } else {
    cerr << "Please provide a JSON CSR file.\n";
    return 1;
  }
}

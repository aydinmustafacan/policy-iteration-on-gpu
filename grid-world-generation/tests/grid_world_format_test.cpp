#include "grid_gen.h"
#include "io.h"
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

class GridWorldFormatTest : public ::testing::Test {
protected:
  void SetUp() override {
    temp_dir = std::filesystem::temp_directory_path() / "grid_test";
    std::filesystem::create_directories(temp_dir);
  }

  void TearDown() override { std::filesystem::remove_all(temp_dir); }

  std::filesystem::path temp_dir;

  // Helper to validate JSON structure
  bool validate_csr_json(const std::string &filepath) {
    std::ifstream f(filepath);
    if (!f)
      return false;

    nlohmann::json j;
    f >> j;

    // Check required fields
    if (!j.contains("S") || !j.contains("A") || !j.contains("gamma") ||
        !j.contains("format") || !j.contains("P") || !j.contains("R")) {
      return false;
    }

    if (j["format"] != "CSR")
      return false;

    // Check P structure
    auto &P = j["P"];
    if (!P.contains("indptr") || !P.contains("indices") ||
        !P.contains("data")) {
      return false;
    }

    // Check R structure
    auto &R = j["R"];
    if (!R.contains("indptr") || !R.contains("indices") ||
        !R.contains("data")) {
      return false;
    }

    int S = j["S"];
    int A = j["A"];

    // Validate indptr sizes
    if (P["indptr"].size() != S * A + 1 || R["indptr"].size() != S * A + 1) {
      return false;
    }

    return true;
  }
};

TEST_F(GridWorldFormatTest, CSRJSONFormatValidation) {
  GridConfig config;
  config.H = 3;
  config.W = 3;
  config.trans = GridTrans::Deterministic;
  config.boundary = Boundary::Reflect;
  config.reward = RewardModel::SparseGoal;
  config.seed = 42;

  // Generate sparse MDP
  MDPSparse mdp;
  generate_gridworld_sparse(config, mdp);

  // Convert to CSR and save
  CSR P_csr = to_csr(mdp, false);
  CSR R_csr = to_csr(mdp, true);

  std::string json_path = temp_dir / "test_mdp.json";

  // Create a simple CSR JSON writer for testing
  std::ofstream f(json_path);
  f << "{\n";
  f << "  \"S\": " << P_csr.S << ",\n";
  f << "  \"A\": " << P_csr.A << ",\n";
  f << "  \"gamma\": 0.9,\n";
  f << "  \"format\": \"CSR\",\n";
  f << "  \"P\": {\n";
  f << "    \"indptr\": [";
  for (size_t i = 0; i < P_csr.indptr.size(); ++i) {
    if (i > 0)
      f << ",";
    f << P_csr.indptr[i];
  }
  f << "],\n";
  f << "    \"indices\": [";
  for (size_t i = 0; i < P_csr.indices.size(); ++i) {
    if (i > 0)
      f << ",";
    f << P_csr.indices[i];
  }
  f << "],\n";
  f << "    \"data\": [";
  for (size_t i = 0; i < P_csr.data.size(); ++i) {
    if (i > 0)
      f << ",";
    f << std::setprecision(17) << P_csr.data[i];
  }
  f << "]\n";
  f << "  },\n";
  f << "  \"R\": {\n";
  f << "    \"indptr\": [";
  for (size_t i = 0; i < R_csr.indptr.size(); ++i) {
    if (i > 0)
      f << ",";
    f << R_csr.indptr[i];
  }
  f << "],\n";
  f << "    \"indices\": [";
  for (size_t i = 0; i < R_csr.indices.size(); ++i) {
    if (i > 0)
      f << ",";
    f << R_csr.indices[i];
  }
  f << "],\n";
  f << "    \"data\": [";
  for (size_t i = 0; i < R_csr.data.size(); ++i) {
    if (i > 0)
      f << ",";
    f << std::setprecision(17) << R_csr.data[i];
  }
  f << "]\n";
  f << "  }\n";
  f << "}\n";
  f.close();

  // Validate the JSON format
  EXPECT_TRUE(validate_csr_json(json_path));
}

TEST_F(GridWorldFormatTest, CSRDataIntegrityValidation) {
  GridConfig config;
  config.H = 4;
  config.W = 4;
  config.trans = GridTrans::Slip;
  config.slip = 0.1;
  config.seed = 123;

  MDPSparse mdp;
  generate_gridworld_sparse(config, mdp);

  CSR P_csr = to_csr(mdp, false);

  // Validate CSR structure integrity
  EXPECT_EQ(P_csr.S, 16);
  EXPECT_EQ(P_csr.A, 4);
  EXPECT_EQ(P_csr.indptr.size(), 16 * 4 + 1);

  // Check that indptr is non-decreasing
  for (size_t i = 1; i < P_csr.indptr.size(); ++i) {
    EXPECT_GE(P_csr.indptr[i], P_csr.indptr[i - 1]);
  }

  // Check that indices are valid state indices
  for (int idx : P_csr.indices) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, 16);
  }

  // Check that data values are valid probabilities
  for (double prob : P_csr.data) {
    EXPECT_GE(prob, 0.0);
    EXPECT_LE(prob, 1.0);
  }

  // Verify each row sums to 1.0
  for (int row = 0; row < 16 * 4; ++row) {
    int start = P_csr.indptr[row];
    int end = P_csr.indptr[row + 1];

    double sum = 0.0;
    for (int k = start; k < end; ++k) {
      sum += P_csr.data[k];
    }

    EXPECT_NEAR(sum, 1.0, 1e-10);
  }
}

TEST_F(GridWorldFormatTest, RoundTripConsistencyValidation) {
  // Test that converting sparse -> CSR -> back produces consistent results
  GridConfig config;
  config.H = 3;
  config.W = 3;
  config.trans = GridTrans::Deterministic;
  config.seed = 42;

  MDPSparse original;
  generate_gridworld_sparse(config, original);

  // Convert to CSR
  CSR P_csr = to_csr(original, false);

  // Convert back to sparse-like representation for comparison
  for (int s = 0; s < original.S; ++s) {
    for (int a = 0; a < original.A; ++a) {
      int row = s * original.A + a;
      int start = P_csr.indptr[row];
      int end = P_csr.indptr[row + 1];

      // Collect CSR data for this (s,a)
      std::map<int, double> csr_transitions;
      for (int k = start; k < end; ++k) {
        csr_transitions[P_csr.indices[k]] = P_csr.data[k];
      }

      // Collect original sparse data
      std::map<int, double> sparse_transitions;
      for (const auto &[state, prob] : original.P[s][a]) {
        sparse_transitions[state] = prob;
      }

      // Compare
      EXPECT_EQ(csr_transitions.size(), sparse_transitions.size());
      for (const auto &[state, prob] : sparse_transitions) {
        EXPECT_TRUE(csr_transitions.count(state) > 0);
        EXPECT_NEAR(csr_transitions[state], prob, 1e-15);
      }
    }
  }
}
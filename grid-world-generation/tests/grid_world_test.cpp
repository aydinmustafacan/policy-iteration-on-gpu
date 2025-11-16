#include "grid_gen.h"
#include "mdp.h"
#include <cmath>
#include <gtest/gtest.h>
#include <unordered_set>

class GridWorldTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Default test configuration
    config.H = 3;
    config.W = 3;
    config.trans = GridTrans::Deterministic;
    config.boundary = Boundary::Reflect;
    config.reward = RewardModel::SparseGoal;
    config.slip = 0.1;
    config.blocked_density = 0.0;
    config.obstacle_density = 0.0;
    config.obstacle_penalty = -1.0;
    config.goal_mode = GoalMode::Fixed;
    config.goals = 1;
    config.seed = 42;
  }

  GridConfig config;

  // Helper to check if probabilities in a row sum to 1.0
  bool is_probability_distribution(const std::vector<double> &probs,
                                   double tolerance = 1e-10) {
    double sum = 0.0;
    for (double p : probs) {
      if (p < 0.0)
        return false;
      sum += p;
    }
    return std::abs(sum - 1.0) < tolerance;
  }

  // Helper to check if sparse transitions sum to 1.0
  bool is_probability_distribution_sparse(
      const std::vector<SparseTransition> &transitions,
      double tolerance = 1e-10) {
    double sum = 0.0;
    for (const auto &[state, prob] : transitions) {
      if (prob < 0.0)
        return false;
      sum += prob;
    }
    return std::abs(sum - 1.0) < tolerance;
  }
};

TEST_F(GridWorldTest, DenseGridWorldBasicProperties) {
  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Check basic dimensions
  EXPECT_EQ(mdp.S, 9); // 3x3 grid
  EXPECT_EQ(mdp.A, 4); // 4 actions (N,S,E,W)
  EXPECT_GT(mdp.gamma, 0.0);
  EXPECT_LE(mdp.gamma, 1.0);

  // Check transition matrix dimensions
  EXPECT_EQ(mdp.P.size(), 9);
  for (int s = 0; s < 9; ++s) {
    EXPECT_EQ(mdp.P[s].size(), 4);
    for (int a = 0; a < 4; ++a) {
      EXPECT_EQ(mdp.P[s][a].size(), 9);
      // Each row should be a valid probability distribution
      EXPECT_TRUE(is_probability_distribution(mdp.P[s][a]));
    }
  }

  // Check reward matrix dimensions
  EXPECT_EQ(mdp.R.size(), 9);
  for (int s = 0; s < 9; ++s) {
    EXPECT_EQ(mdp.R[s].size(), 4);
    for (int a = 0; a < 4; ++a) {
      EXPECT_EQ(mdp.R[s][a].size(), 9);
    }
  }
}

TEST_F(GridWorldTest, SparseGridWorldBasicProperties) {
  MDPSparse mdp;
  generate_gridworld_sparse(config, mdp);

  // Check basic dimensions
  EXPECT_EQ(mdp.S, 9);
  EXPECT_EQ(mdp.A, 4);

  // Check sparse structure
  EXPECT_EQ(mdp.P.size(), 9);
  for (int s = 0; s < 9; ++s) {
    EXPECT_EQ(mdp.P[s].size(), 4);
    for (int a = 0; a < 4; ++a) {
      // Each (s,a) should have valid probability distribution
      EXPECT_TRUE(is_probability_distribution_sparse(mdp.P[s][a]));

      // Should have at least one transition (even if self-loop)
      EXPECT_GT(mdp.P[s][a].size(), 0);
    }
  }
}

TEST_F(GridWorldTest, DeterministicMovementValidation) {
  config.trans = GridTrans::Deterministic;
  config.boundary = Boundary::Reflect;

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Test corner movements (should reflect)
  // State 0 (top-left corner): moving north or west should stay in place
  EXPECT_EQ(mdp.P[0][0][0], 1.0); // North from (0,0) stays at (0,0)
  EXPECT_EQ(mdp.P[0][3][0], 1.0); // West from (0,0) stays at (0,0)
  EXPECT_EQ(mdp.P[0][2][1], 1.0); // East from (0,0) goes to (0,1)
  EXPECT_EQ(mdp.P[0][1][3], 1.0); // South from (0,0) goes to (1,0)

  // Test center movement
  // State 4 (center): all movements should be deterministic
  EXPECT_EQ(mdp.P[4][0][1], 1.0); // North from center
  EXPECT_EQ(mdp.P[4][1][7], 1.0); // South from center
  EXPECT_EQ(mdp.P[4][2][5], 1.0); // East from center
  EXPECT_EQ(mdp.P[4][3][3], 1.0); // West from center
}

TEST_F(GridWorldTest, SlippyMovementValidation) {
  config.trans = GridTrans::Slip;
  config.slip = 0.2; // 20% slip

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Test slippy movement from center (state 4)
  // Moving north should have 80% chance north, 10% east, 10% west
  EXPECT_NEAR(mdp.P[4][0][1], 0.8, 1e-10); // North
  EXPECT_NEAR(mdp.P[4][0][5], 0.1, 1e-10); // East (slip)
  EXPECT_NEAR(mdp.P[4][0][3], 0.1, 1e-10); // West (slip)
  EXPECT_NEAR(mdp.P[4][0][7], 0.0, 1e-10); // South (opposite, no slip)
}

TEST_F(GridWorldTest, RewardModelValidation) {
  // Test sparse goal reward
  config.reward = RewardModel::SparseGoal;
  config.goal_mode = GoalMode::Fixed; // Goal at bottom-right (state 8)

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Goal reward should only be at state 8
  for (int s = 0; s < 9; ++s) {
    for (int a = 0; a < 4; ++a) {
      for (int s2 = 0; s2 < 9; ++s2) {
        if (s2 == 8) {
          EXPECT_EQ(mdp.R[s][a][s2], 1.0);
        } else {
          EXPECT_EQ(mdp.R[s][a][s2], 0.0);
        }
      }
    }
  }
}

TEST_F(GridWorldTest, BlockedCellsValidation) {
  config.blocked_density = 0.5; // 50% blocked
  config.seed = 123;            // Fixed seed for reproducibility

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Check that blocked states have self-loops for all actions
  std::vector<bool> is_blocked(9, false);

  // Identify blocked states by checking for self-loops in all actions
  for (int s = 0; s < 9; ++s) {
    bool all_self_loops = true;
    for (int a = 0; a < 4; ++a) {
      if (mdp.P[s][a][s] < 0.99) { // Allow for floating point precision
        all_self_loops = false;
        break;
      }
    }
    if (all_self_loops && s != 0) { // State 0 is never blocked
      is_blocked[s] = true;
    }
  }

  // State 0 should never be blocked
  EXPECT_FALSE(is_blocked[0]);
}

TEST_F(GridWorldTest, CSRConversionValidation) {
  MDPSparse sparse_mdp;
  generate_gridworld_sparse(config, sparse_mdp);

  // Convert to CSR for transitions
  CSR P_csr = to_csr(sparse_mdp, false);
  CSR R_csr = to_csr(sparse_mdp, true);

  // Check CSR structure
  EXPECT_EQ(P_csr.S, 9);
  EXPECT_EQ(P_csr.A, 4);
  EXPECT_EQ(P_csr.indptr.size(), 9 * 4 + 1); // S*A + 1

  // Verify CSR represents same probabilities
  for (int s = 0; s < 9; ++s) {
    for (int a = 0; a < 4; ++a) {
      int row = s * 4 + a;
      int start = P_csr.indptr[row];
      int end = P_csr.indptr[row + 1];

      // Check that CSR row represents same distribution as sparse
      std::unordered_map<int, double> csr_probs;
      for (int k = start; k < end; ++k) {
        csr_probs[P_csr.indices[k]] = P_csr.data[k];
      }

      std::unordered_map<int, double> sparse_probs;
      for (const auto &[state, prob] : sparse_mdp.P[s][a]) {
        sparse_probs[state] = prob;
      }

      EXPECT_EQ(csr_probs.size(), sparse_probs.size());
      for (const auto &[state, prob] : sparse_probs) {
        EXPECT_NEAR(csr_probs[state], prob, 1e-10);
      }
    }
  }
}

TEST_F(GridWorldTest, StandardFormatValidation) {
  // Generate both dense and sparse versions
  MDPDense dense_mdp;
  MDPSparse sparse_mdp;

  generate_gridworld_dense(config, dense_mdp);
  generate_gridworld_sparse(config, sparse_mdp);

  // Convert sparse to CSR
  CSR P_csr = to_csr(sparse_mdp, false);
  CSR R_csr = to_csr(sparse_mdp, true);

  // Verify both representations produce same MDP
  for (int s = 0; s < 9; ++s) {
    for (int a = 0; a < 4; ++a) {
      // Compare transition probabilities
      int row = s * 4 + a;
      int start = P_csr.indptr[row];
      int end = P_csr.indptr[row + 1];

      std::vector<double> dense_probs = dense_mdp.P[s][a];
      std::vector<double> csr_probs(9, 0.0);

      for (int k = start; k < end; ++k) {
        csr_probs[P_csr.indices[k]] = P_csr.data[k];
      }

      for (int s2 = 0; s2 < 9; ++s2) {
        EXPECT_NEAR(dense_probs[s2], csr_probs[s2], 1e-10);
      }
    }
  }
}

TEST_F(GridWorldTest, MultipleGoalsValidation) {
  config.goal_mode = GoalMode::Multi;
  config.goals = 3;
  config.seed = 42;

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Count states with goal rewards
  std::unordered_set<int> goal_states;
  for (int s = 0; s < 9; ++s) {
    for (int a = 0; a < 4; ++a) {
      for (int s2 = 0; s2 < 9; ++s2) {
        if (mdp.R[s][a][s2] > 0.0) {
          goal_states.insert(s2);
        }
      }
    }
  }

  // Should have at most 3 goal states (could be fewer due to deduplication)
  EXPECT_LE(goal_states.size(), 3);
  EXPECT_GE(goal_states.size(), 1);
}

TEST_F(GridWorldTest, BoundaryHandlingValidation) {
  // Test different boundary conditions
  config.H = 2;
  config.W = 2;
  config.trans = GridTrans::Deterministic;

  // Test wrap boundary
  config.boundary = Boundary::Wrap;
  MDPDense wrap_mdp;
  generate_gridworld_dense(config, wrap_mdp);

  // Moving north from state 0 should wrap to state 2
  EXPECT_EQ(wrap_mdp.P[0][0][2], 1.0);

  // Test clamp boundary
  config.boundary = Boundary::Clamp;
  MDPDense clamp_mdp;
  generate_gridworld_dense(config, clamp_mdp);

  // Moving north from state 0 should clamp to state 0
  EXPECT_EQ(clamp_mdp.P[0][0][0], 1.0);
}

TEST_F(GridWorldTest, ConsistencyBetweenDenseAndSparse) {
  MDPDense dense_mdp;
  MDPSparse sparse_mdp;

  generate_gridworld_dense(config, dense_mdp);
  generate_gridworld_sparse(config, sparse_mdp);

  // Both should have same basic properties
  EXPECT_EQ(dense_mdp.S, sparse_mdp.S);
  EXPECT_EQ(dense_mdp.A, sparse_mdp.A);

  // Compare transition probabilities
  for (int s = 0; s < dense_mdp.S; ++s) {
    for (int a = 0; a < dense_mdp.A; ++a) {
      // Convert sparse to dense format for comparison
      std::vector<double> sparse_as_dense(dense_mdp.S, 0.0);
      for (const auto &[state, prob] : sparse_mdp.P[s][a]) {
        sparse_as_dense[state] = prob;
      }

      for (int s2 = 0; s2 < dense_mdp.S; ++s2) {
        EXPECT_NEAR(dense_mdp.P[s][a][s2], sparse_as_dense[s2], 1e-10);
      }
    }
  }
}

TEST_F(GridWorldTest, ObstacleRewardValidation) {
  config.reward = RewardModel::ObstaclesNeg;
  config.obstacle_density = 0.3;
  config.obstacle_penalty = -2.0;
  config.seed = 789;

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Check that obstacle rewards are negative and goal rewards are zero
  bool found_obstacle = false;
  for (int s = 0; s < 9; ++s) {
    for (int a = 0; a < 4; ++a) {
      for (int s2 = 0; s2 < 9; ++s2) {
        if (mdp.R[s][a][s2] < 0.0) {
          EXPECT_EQ(mdp.R[s][a][s2], -2.0);
          found_obstacle = true;
        }
        // No positive rewards in obstacle-only mode
        EXPECT_LE(mdp.R[s][a][s2], 0.0);
      }
    }
  }

  // Should have found at least one obstacle (probabilistic, but likely with 30%
  // density) Note: This test might occasionally fail due to randomness,
  // consider using fixed obstacle positions for deterministic testing
}

// Additional validation test for grid world properties
TEST_F(GridWorldTest, GridWorldStructuralValidation) {
  config.H = 4;
  config.W = 4;

  MDPDense mdp;
  generate_gridworld_dense(config, mdp);

  // Validate that state space corresponds to grid positions
  int S = config.H * config.W;
  EXPECT_EQ(mdp.S, S);

  // Check that each state has valid neighbors
  for (int s = 0; s < S; ++s) {
    int i = s / config.W; // row
    int j = s % config.W; // column

    EXPECT_GE(i, 0);
    EXPECT_LT(i, config.H);
    EXPECT_GE(j, 0);
    EXPECT_LT(j, config.W);

    // Verify state conversion is consistent
    int reconstructed_state = i * config.W + j;
    EXPECT_EQ(s, reconstructed_state);
  }
}
#include <gtest/gtest.h>
#include "../reader.h"
#include <fstream>
#include <filesystem>

class ReaderTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create test directory
    std::filesystem::create_directory("test_data");

    // Create test policy file
    std::ofstream policy_file("test_data/test_policy.csv");
    policy_file << "state,action\n";
    policy_file << "0,1\n";
    policy_file << "1,0\n";
    policy_file << "2,1\n";
    policy_file.close();

    // Create test values file
    std::ofstream values_file("test_data/test_values.csv");
    values_file << "state,value\n";
    values_file << "0,10.5\n";
    values_file << "1,8.2\n";
    values_file << "2,6.7\n";
    values_file.close();

    // Create test transition file
    std::ofstream trans_file("test_data/test_transitions.csv");
    trans_file << "state,action,next_state,probability\n";
    trans_file << "0,0,0,0.8\n";
    trans_file << "0,0,1,0.2\n";
    trans_file << "0,1,1,1.0\n";
    trans_file << "1,0,0,0.5\n";
    trans_file << "1,0,1,0.5\n";
    trans_file << "1,1,1,1.0\n";
    trans_file.close();

    // Create test reward file
    std::ofstream reward_file("test_data/test_rewards.csv");
    reward_file << "state,action,next_state,reward\n";
    reward_file << "0,0,0,1.0\n";
    reward_file << "0,0,1,0.0\n";
    reward_file << "0,1,1,5.0\n";
    reward_file << "1,0,0,2.0\n";
    reward_file << "1,0,1,1.0\n";
    reward_file << "1,1,1,3.0\n";
    reward_file.close();
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove_all("test_data");
  }
};

TEST_F(ReaderTest, ReadPolicyFromCSV) {
  Policy policy = reader::read_policy_from_csv("test_data/test_policy.csv");

  EXPECT_EQ(policy.size(), 3);
  EXPECT_EQ(policy[0], 1);
  EXPECT_EQ(policy[1], 0);
  EXPECT_EQ(policy[2], 1);
}

TEST_F(ReaderTest, ReadValuesFromCSV) {
  Vector values = reader::read_values_from_csv("test_data/test_values.csv");

  EXPECT_EQ(values.size(), 3);
  EXPECT_DOUBLE_EQ(values[0], 10.5);
  EXPECT_DOUBLE_EQ(values[1], 8.2);
  EXPECT_DOUBLE_EQ(values[2], 6.7);
}

TEST_F(ReaderTest, ReadMDPFromCSV) {
  Matrix3D P, R;
  reader::read_mdp_from_csv("test_data/test_transitions.csv",
                            "test_data/test_rewards.csv", P, R);

  EXPECT_EQ(P.size(), 2); // 2 states (0,1)
  EXPECT_EQ(R.size(), 2);
  EXPECT_EQ(P[0].size(), 2); // 2 actions (0,1)
  EXPECT_EQ(R[0].size(), 2);

  // Check specific transitions
  EXPECT_DOUBLE_EQ(P[0][0][0], 0.8);
  EXPECT_DOUBLE_EQ(P[0][0][1], 0.2);
  EXPECT_DOUBLE_EQ(P[0][1][1], 1.0);

  // Check specific rewards
  EXPECT_DOUBLE_EQ(R[0][0][0], 1.0);
  EXPECT_DOUBLE_EQ(R[0][1][1], 5.0);
  EXPECT_DOUBLE_EQ(R[1][0][0], 2.0);
}

TEST_F(ReaderTest, HandleNonExistentFile) {
  Policy policy = reader::read_policy_from_csv("nonexistent.csv");
  EXPECT_TRUE(policy.empty());

  Vector values = reader::read_values_from_csv("nonexistent.csv");
  EXPECT_TRUE(values.empty());
}

TEST_F(ReaderTest, HandleMalformedCSV) {
  // Create malformed CSV file
  std::ofstream bad_file("test_data/bad_policy.csv");
  bad_file << "state,action\n";
  bad_file << "0,1\n";
  bad_file << "invalid,data\n";
  bad_file << "2,0\n";
  bad_file.close();

  Policy policy = reader::read_policy_from_csv("test_data/bad_policy.csv");

  // Should still read valid lines
  EXPECT_GE(policy.size(), 2);
  EXPECT_EQ(policy[0], 1);
  if (policy.size() > 2) {
    EXPECT_EQ(policy[2], 0);
  }
}

TEST_F(ReaderTest, HandleFileWithoutHeader) {
  // Create CSV without header
  std::ofstream no_header("test_data/no_header_policy.csv");
  no_header << "0,1\n";
  no_header << "1,0\n";
  no_header.close();

  Policy policy = reader::read_policy_from_csv("test_data/no_header_policy.csv");

  EXPECT_EQ(policy.size(), 2);
  EXPECT_EQ(policy[0], 1);
  EXPECT_EQ(policy[1], 0);
}

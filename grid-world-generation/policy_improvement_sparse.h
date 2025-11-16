#ifndef GRID_WORLD_GENERATION_POLICY_IMPROVEMENT_SPARSE_H
#define GRID_WORLD_GENERATION_POLICY_IMPROVEMENT_SPARSE_H

#include <vector>
#include <utility>
#include <unordered_map>

// Type aliases for sparse representation
using SparseTransition = std::vector<std::pair<int, double>>;
using SparseMatrix = std::vector<std::vector<SparseTransition>>;
using SparseReward = std::vector<std::vector<std::unordered_map<int, double>>>;
using Vector = std::vector<double>;
using Policy = std::vector<int>;

// Function declarations
void generate_gridworld_sparse(int H, int W, double slip,
                               SparseMatrix& P,
                               SparseReward& R);

Vector policy_evaluation_sparse(const Policy& policy,
                                const SparseMatrix& P,
                                const SparseReward& R,
                                double gamma,
                                double theta = 1e-8,
                                int max_iters = 10000);

Policy policy_improvement_sparse(const Vector& V,
                                 const SparseMatrix& P,
                                 const SparseReward& R,
                                 double gamma);

std::pair<Policy, Vector> policy_iteration_sparse(const SparseMatrix& P,
                                                  const SparseReward& R,
                                                  double gamma,
                                                  double theta = 1e-8,
                                                  int max_iters = 1000);

#endif //GRID_WORLD_GENERATION_POLICY_IMPROVEMENT_SPARSE_H
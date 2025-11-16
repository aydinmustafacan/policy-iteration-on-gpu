#ifndef GRID_WORLD_GENERATION_POLICY_ITERATION_H
#define GRID_WORLD_GENERATION_POLICY_ITERATION_H

#include <vector>
#include <utility>

// Type aliases
using Matrix3D = std::vector<std::vector<std::vector<double>>>;
using Vector   = std::vector<double>;
using Policy   = std::vector<int>;

// Function declarations
void generate_gridworld(int H, int W, double slip,
                        Matrix3D& P,
                        Matrix3D& R);

Vector policy_evaluation(const Policy& policy,
                         const Matrix3D& P,
                         const Matrix3D& R,
                         double gamma,
                         double theta = 1e-8,
                         int max_iters = 10000);

Policy policy_improvement(const Vector& V,
                          const Matrix3D& P,
                          const Matrix3D& R,
                          double gamma);

std::pair<Policy,Vector> policy_iteration(const Matrix3D& P,
                                          const Matrix3D& R,
                                          double gamma,
                                          double theta = 1e-8,
                                          int max_iters = 1000);

#endif //GRID_WORLD_GENERATION_POLICY_ITERATION_H
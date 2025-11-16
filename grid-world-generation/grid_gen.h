#pragma once
#include "mdp.h"

void generate_gridworld_dense(const GridConfig& cfg, MDPDense& out);
void generate_gridworld_sparse(const GridConfig& cfg, MDPSparse& out);

// Utility: convert sparse-per-(s,a) to CSR
CSR to_csr(const MDPSparse& spP, bool for_rewards);

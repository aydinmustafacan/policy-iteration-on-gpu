#pragma once
#include "mdp.h"

void generate_random_mdp(const RandomMDPConfig& cfg, MDPCSR& out);     // CSR native
void generate_random_mdp_dense(const RandomMDPConfig& cfg, MDPDense&); // optional dense

#pragma once
#include "mdp.h"
#include <string>

// Save (S,A,P,R,gamma). For dense, write P/R as CSV triplets or nested arrays in JSON.
// For CSR, write {"rows":S*A,"indptr":[...],"indices":[...],"data":[...]}.
void save_dense_json(const MDPDense& mdp, const std::string& path);
void save_csr_json(const MDPCSR& mdp, const std::string& path);
void save_policy_csv(const Policy& pi, const std::string& path);
void save_value_csv(const Vector& V, const std::string& path);

// Optional fast binary dumps (little endian)
void save_csr_bin(const CSR& csr, const std::string& path);

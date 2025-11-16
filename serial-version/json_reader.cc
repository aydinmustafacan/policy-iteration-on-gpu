#include "json_reader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

// Simple JSON parser (basic implementation for the specific format)
JSONReader::MDPData JSONReader::parse_json(const std::string &json_content) {
  MDPData data;

  // Find S, A, gamma
  size_t s_pos = json_content.find("\"S\":");
  if (s_pos != std::string::npos) {
    size_t start = json_content.find_first_of("0123456789", s_pos);
    size_t end = json_content.find_first_of(",}", start);
    data.S = std::stoi(json_content.substr(start, end - start));
  }

  size_t a_pos = json_content.find("\"A\":");
  if (a_pos != std::string::npos) {
    size_t start = json_content.find_first_of("0123456789", a_pos);
    size_t end = json_content.find_first_of(",}", start);
    data.A = std::stoi(json_content.substr(start, end - start));
  }

  size_t gamma_pos = json_content.find("\"gamma\":");
  if (gamma_pos != std::string::npos) {
    size_t start = json_content.find_first_of("0123456789.", gamma_pos);
    size_t end = json_content.find_first_of(",}", start);
    data.gamma = std::stod(json_content.substr(start, end - start));
  }

  // Parse P matrix
  size_t p_start = json_content.find("\"P\":");
  if (p_start != std::string::npos) {
    size_t p_block_start = json_content.find("{", p_start);
    size_t p_block_end = json_content.find("}", p_block_start);
    std::string p_block = json_content.substr(p_block_start + 1, p_block_end - p_block_start - 1);

    // Parse indptr
    size_t indptr_pos = p_block.find("\"indptr\":");
    if (indptr_pos != std::string::npos) {
      size_t array_start = p_block.find("[", indptr_pos);
      size_t array_end = p_block.find("]", array_start);
      std::string array_str = p_block.substr(array_start + 1, array_end - array_start - 1);
      data.P.indptr = parse_int_array(array_str);
    }

    // Parse indices
    size_t indices_pos = p_block.find("\"indices\":");
    if (indices_pos != std::string::npos) {
      size_t array_start = p_block.find("[", indices_pos);
      size_t array_end = p_block.find("]", array_start);
      std::string array_str = p_block.substr(array_start + 1, array_end - array_start - 1);
      data.P.indices = parse_int_array(array_str);
    }

    // Parse data
    size_t data_pos = p_block.find("\"data\":");
    if (data_pos != std::string::npos) {
      size_t array_start = p_block.find("[", data_pos);
      size_t array_end = p_block.find("]", array_start);
      std::string array_str = p_block.substr(array_start + 1, array_end - array_start - 1);
      data.P.data = parse_double_array(array_str);
    }
  }

  // Parse R matrix (similar to P)
  size_t r_start = json_content.find("\"R\":");
  if (r_start != std::string::npos) {
    size_t r_block_start = json_content.find("{", r_start);
    size_t r_block_end = json_content.find("}", r_block_start);
    std::string r_block = json_content.substr(r_block_start + 1, r_block_end - r_block_start - 1);

    // Parse indptr
    size_t indptr_pos = r_block.find("\"indptr\":");
    if (indptr_pos != std::string::npos) {
      size_t array_start = r_block.find("[", indptr_pos);
      size_t array_end = r_block.find("]", array_start);
      std::string array_str = r_block.substr(array_start + 1, array_end - array_start - 1);
      data.R.indptr = parse_int_array(array_str);
    }

    // Parse indices
    size_t indices_pos = r_block.find("\"indices\":");
    if (indices_pos != std::string::npos) {
      size_t array_start = r_block.find("[", indices_pos);
      size_t array_end = r_block.find("]", array_start);
      std::string array_str = r_block.substr(array_start + 1, array_end - array_start - 1);
      data.R.indices = parse_int_array(array_str);
    }

    // Parse data
    size_t data_pos = r_block.find("\"data\":");
    if (data_pos != std::string::npos) {
      size_t array_start = r_block.find("[", data_pos);
      size_t array_end = r_block.find("]", array_start);
      std::string array_str = r_block.substr(array_start + 1, array_end - array_start - 1);
      data.R.data = parse_double_array(array_str);
    }
  }

  // Align R to P if R is provided as data-only (no CSR structure), mirroring CUDA logic
  const bool R_has_csr = !data.R.indptr.empty() && !data.R.indices.empty();
  const bool R_has_only_data = data.R.indptr.empty() && data.R.indices.empty() && !data.R.data.empty();
  if (R_has_only_data) {
    if (data.R.data.size() != data.P.data.size()) {
      std::cerr << "R.data length " << data.R.data.size()
          << " does not match P.nnz " << data.P.data.size() << "\n";
    } else {
      std::cout << "Non-CSR R matrix aligned to P's structure." << std::endl;
      data.R.indptr = data.P.indptr;
      data.R.indices = data.P.indices;
      // data.R.data already aligned 1:1 with P.data
    }
  }

  return data;
}

void JSONReader::csr_to_matrix3d(const CSRMatrix &csr_matrix, Matrix3D &matrix3d,
                                 int S, int A) {
  matrix3d.assign(S, std::vector<std::vector<double> >(A, std::vector<double>(S, 0.0)));

  // Convert CSR format to 3D matrix
  for (int state_action = 0; state_action < static_cast<int>(csr_matrix.indptr.size() - 1); ++state_action) {
    int state = state_action / A;
    int action = state_action % A;

    int start_idx = csr_matrix.indptr[state_action];
    int end_idx = csr_matrix.indptr[state_action + 1];

    for (int idx = start_idx; idx < end_idx; ++idx) {
      int next_state = csr_matrix.indices[idx];
      double value = csr_matrix.data[idx];

      if (state < S && action < A && next_state < S) {
        matrix3d[state][action][next_state] = value;
      }
    }
  }
}

bool JSONReader::read_mdp_from_json(const std::string &json_file,
                                    Matrix3D &P, Matrix3D &R,
                                    double &gamma) {
  // Read JSON file
  std::ifstream file(json_file);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open JSON file " << json_file << std::endl;
    return false;
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string json_content = buffer.str();
  file.close();

  // Parse JSON
  MDPData data = parse_json(json_content);

  int S = data.S;
  int A = data.A;
  gamma = data.gamma;

  // Convert CSR matrices to 3D matrices
  csr_to_matrix3d(data.P, P, S, A);
  csr_to_matrix3d(data.R, R, S, A);

  std::cout << "Successfully read MDP from JSON with " << S << " states and " << A << " actions" << std::endl;
  std::cout << "Gamma: " << gamma << std::endl;

  return true;
}

std::string JSONReader::trim(const std::string &str) {
  size_t start = str.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) return "";
  size_t end = str.find_last_not_of(" \t\r\n");
  return str.substr(start, end - start + 1);
}

std::vector<int> JSONReader::parse_int_array(const std::string &array_str) {
  std::vector<int> result;
  std::stringstream ss(array_str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (!token.empty()) {
      result.push_back(std::stoi(token));
    }
  }
  return result;
}

std::vector<double> JSONReader::parse_double_array(const std::string &array_str) {
  std::vector<double> result;
  std::stringstream ss(array_str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    token = trim(token);
    if (!token.empty()) {
      result.push_back(std::stod(token));
    }
  }
  return result;
}

#ifndef JSON_READER_H
#define JSON_READER_H

#include <string>
#include <vector>

// Type aliases for clarity (matching serial version)
using Matrix3D = std::vector<std::vector<std::vector<double> > >;

// Simple JSON reader for MDP data
class JSONReader {
public:
  struct CSRMatrix {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
  };

  struct MDPData {
    int S; // number of states
    int A; // number of actions
    double gamma;
    CSRMatrix P; // transition probabilities
    CSRMatrix R; // rewards
  };

  // Read MDP from JSON file and convert to 3D matrices for serial version
  static bool read_mdp_from_json(const std::string &json_file,
                                 Matrix3D &P, Matrix3D &R,
                                 double &gamma);

  // Parse JSON content directly to CSR format (for sparse solver)
  static MDPData parse_json(const std::string &json_content);

  // Convert CSR to dense format (for small problems)
  static void csr_to_matrix3d(const CSRMatrix &csr_matrix, Matrix3D &matrix3d,
                              int S, int A);

private:
  static std::string trim(const std::string &str);

  static std::vector<int> parse_int_array(const std::string &array_str);

  static std::vector<double> parse_double_array(const std::string &array_str);
};

#endif // JSON_READER_H

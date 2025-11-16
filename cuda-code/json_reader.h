#ifndef JSON_READER_H
#define JSON_READER_H

#include <string>
#include <vector>

// Simple JSON reader for MDP data
class JSONReader {
public:
    struct CSRMatrix {
        std::vector<int> indptr;
        std::vector<int> indices;
        std::vector<double> data;
    };
    
    struct MDPData {
        int S;  // number of states
        int A;  // number of actions
        double gamma;
        CSRMatrix P;  // transition probabilities
        CSRMatrix R;  // rewards
    };
    
    // Read MDP from JSON file and convert to flat arrays for CUDA
    static bool read_mdp_from_json(const std::string& json_file,
                                   std::vector<double>& P_flat,
                                   std::vector<double>& R_flat,
                                   int& S, int& A, double& gamma);
    
private:
    static MDPData parse_json(const std::string& json_content);
    static void csr_to_flat(const CSRMatrix& csr_matrix, std::vector<double>& flat_matrix, 
                           int S, int A);
    static std::string trim(const std::string& str);
    static std::vector<int> parse_int_array(const std::string& array_str);
    static std::vector<double> parse_double_array(const std::string& array_str);
};

#endif // JSON_READER_H

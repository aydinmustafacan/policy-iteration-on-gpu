// main.cu — CUDA Policy Iteration using CSR sparse matrices (shared sparsity for P and R)
// Build: nvcc -O3 -std=c++17 main.cu -o cuda_pi_csr
//
// Usage:
//   1) JSON (CSR):   ./cuda_pi_csr mdp.json
//   2) CSV:          ./cuda_pi_csr transitions.csv rewards.csv
//   3) Default demo: ./cuda_pi_csr
//
// JSON format (supports two variants for R):
// {
//   "S": 9, "A": 4, "gamma": 0.9,
//   "P": { "indptr":[...], "indices":[...], "data":[...] },
//   // Variant A (aligned): R.data length == P.data length, shares ordering
//   "R": { "data":[...] }
//   // Variant B (separate CSR): will be aligned to P row-by-row by s' column
//   // "R": { "indptr":[...], "indices":[...], "data":[...] }
// }

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <limits>
#include <cmath>
#include <fstream>
#include <chrono>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <string>
#include <cctype>
#include <cstdlib>

// Precision switch: define USE_FP32 at compile time to use float on GPU
#ifdef USE_FP32
using Real = float;
#else
using Real = double;
#endif

// MACROS 
// ---------- CUDA error check ----------
#define CUDA_CHECK(err) if ((err) != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(1); \
}

// ---------- Tiny string helpers ----------
static inline void trim_inplace(std::string& s) {
    auto not_space = [](int ch){ return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        trim_inplace(tok);
        tokens.push_back(tok);
    }
    return tokens;
}

bool skip_header_if_present(std::ifstream& file, const std::string& indicator = "state") {
    std::streampos start_pos = file.tellg();
    std::string line;
    if (std::getline(file, line)) {
        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find(indicator) != std::string::npos) {
            return true; // consumed header line
        } else {
            file.clear();
            file.seekg(start_pos);
            return false;
        }
    }
    return false;
}

// ---------- CSV → CSR loader (shared sparsity) ----------
struct PairPR { double p=0.0, r=0.0; };

bool read_mdp_from_csv_csr(
    const std::string& transition_file,
    const std::string& reward_file,
    std::vector<int>& indptr,
    std::vector<int>& indices,
    std::vector<double>& P_data,
    std::vector<double>& R_data,
    int& S, int& A)
{
    using Row = std::unordered_map<int, PairPR>; // s' -> (p,r), per (s,a) row
    std::vector<Row> rows;

    // Pass 1: determine S, A
    {
        std::ifstream f(transition_file);
        if (!f) { std::cerr << "Cannot open " << transition_file << "\n"; return false; }
        skip_header_if_present(f);
    std::string line;
    int max_s=-1, max_a=-1;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = parse_csv_line(line);
            if (t.size() < 4) continue;
            int s  = std::stoi(t[0]);
            int a  = std::stoi(t[1]);
            int s2 = std::stoi(t[2]);
            max_s  = std::max({max_s, s, s2});
            max_a  = std::max(max_a, a);
        }
        S = max_s + 1;
        A = max_a + 1;
        rows.assign(S * A, Row{});
    }

    // Pass 2: read transitions
    {
        std::ifstream f(transition_file);
        if (!f) { std::cerr << "Cannot open " << transition_file << "\n"; return false; }
        skip_header_if_present(f);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = parse_csv_line(line);
            if (t.size() < 4) continue;
            int s  = std::stoi(t[0]);
            int a  = std::stoi(t[1]);
            int s2 = std::stoi(t[2]);
            double p = std::stod(t[3]);
            auto& cell = rows[s*A + a][s2];
            cell.p += p; // sum duplicates
        }
    }

    // Rewards (overlay on existing transitions to share sparsity)
    {
        std::ifstream f(reward_file);
        if (!f) { std::cerr << "Cannot open " << reward_file << "\n"; return false; }
        skip_header_if_present(f);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            auto t = parse_csv_line(line);
            if (t.size() < 4) continue;
            int s  = std::stoi(t[0]);
            int a  = std::stoi(t[1]);
            int s2 = std::stoi(t[2]);
            double r = std::stod(t[3]);
            auto& row = rows[s*A + a];
            auto it = row.find(s2);
            if (it != row.end()) it->second.r += r; // ignore rewards on zero-prob edges
        }
    }

    // Build CSR
    indptr.assign(S*A + 1, 0);
    for (int row = 0; row < S*A; ++row) indptr[row + 1] = indptr[row] + (int)rows[row].size();
    int nnz = indptr.back();
    indices.resize(nnz);
    P_data.resize(nnz);
    R_data.resize(nnz);

    // Fill sorted by column (s') for determinism
    for (int row = 0; row < S*A; ++row) {
        int start = indptr[row];
        std::vector<std::pair<int, PairPR>> tmp;
        tmp.reserve(rows[row].size());
        for (auto& kv : rows[row]) tmp.push_back(kv);
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });
        for (int i = 0; i < (int)tmp.size(); ++i) {
            indices[start + i] = tmp[i].first;
            P_data[start + i]  = tmp[i].second.p;
            R_data[start + i]  = tmp[i].second.r;
        }
    }

    // Optional: normalize P row-wise so sum_{s'} P = 1 for each (s,a)
    for (int row = 0; row < S*A; ++row) {
        int beg = indptr[row], end = indptr[row+1];
        double sum = 0.0;
        for (int j = beg; j < end; ++j) sum += P_data[j];
        if (sum > 0) {
            double inv = 1.0 / sum;
            for (int j = beg; j < end; ++j) P_data[j] *= inv;
        }
    }
    return true;
}

// ---------- Minimal JSON (CSR) loader with R alignment ----------
static bool read_file_to_string(const std::string& path, std::string& out) {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Extracts a number field like:  "S": 123
static bool json_extract_number(const std::string& js, const std::string& key, double& out) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t colon = js.find(':', k);
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < js.size() && std::isspace((unsigned char)js[i])) ++i;
    size_t j = i;
    auto isnum = [](char c){ return std::isdigit((unsigned char)c) || c=='+'||c=='-'||c=='.'||c=='e'||c=='E'; };
    while (j < js.size() && isnum(js[j])) ++j;
    try { out = std::stod(js.substr(i, j - i)); return true; } catch (...) { return false; }
}

// Extract array content between the first [...] after "key":
static bool json_extract_array_raw(const std::string& js, const std::string& key, std::string& out_inside) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return false;
    size_t colon = js.find(':', k);
    if (colon == std::string::npos) return false;
    size_t sb = js.find('[', colon);
    if (sb == std::string::npos) return false;
    int depth = 0;
    for (size_t i = sb; i < js.size(); ++i) {
        if (js[i] == '[') depth++;
        else if (js[i] == ']') {
            depth--;
            if (depth == 0) {
                out_inside = js.substr(sb + 1, i - sb - 1); // inside without brackets
                return true;
            }
        }
    }
    return false;
}

template <typename T>
static bool parse_numeric_array(const std::string& inside, std::vector<T>& out) {
    out.clear();
    std::stringstream ss(inside);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        trim_inplace(tok);
        if (tok.empty()) continue;
        try {
            if constexpr (std::is_same<T,int>::value) {
                out.push_back(static_cast<int>(std::stoll(tok)));
            } else {
                out.push_back(static_cast<double>(std::stod(tok)));
            }
        } catch (...) { /* tolerate noise */ }
    }
    return true;
}

bool read_mdp_from_json_csr(
    const std::string& filename,
    std::vector<int>& indptr,
    std::vector<int>& indices,
    std::vector<double>& P_data,
    std::vector<double>& R_data,
    int& S, int& A, double& gamma)
{
    std::string js;
    if (!read_file_to_string(filename, js)) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }

    // --- Scalars ---
    double Sd=0, Ad=0, gd=0.9;
    if (!json_extract_number(js, "S", Sd)) { std::cerr << "JSON missing S\n"; return false; }
    if (!json_extract_number(js, "A", Ad)) { std::cerr << "JSON missing A\n"; return false; }
    json_extract_number(js, "gamma", gd);
    S = static_cast<int>(Sd);
    A = static_cast<int>(Ad);
    gamma = gd;

    // --- P (CSR) ---
    size_t ppos = js.find("\"P\"");
    if (ppos == std::string::npos) { std::cerr << "JSON missing P object\n"; return false; }
    std::string jsP = js.substr(ppos);

    std::string P_indptr_raw, P_indices_raw, P_data_raw;
    if (!json_extract_array_raw(jsP, "indptr", P_indptr_raw)) { std::cerr << "JSON P.indptr missing\n"; return false; }
    if (!json_extract_array_raw(jsP, "indices", P_indices_raw)) { std::cerr << "JSON P.indices missing\n"; return false; }
    if (!json_extract_array_raw(jsP, "data", P_data_raw)) { std::cerr << "JSON P.data missing\n"; return false; }

    parse_numeric_array<int>(P_indptr_raw, indptr);
    parse_numeric_array<int>(P_indices_raw, indices);
    parse_numeric_array<double>(P_data_raw, P_data);

    if (indptr.size() != static_cast<size_t>(S*A + 1)) {
        std::cerr << "indptr length mismatch: got " << indptr.size() << " expected " << (S*A+1) << "\n";
        return false;
    }
    if (indices.size() != P_data.size()) {
        std::cerr << "P.indices and P.data length mismatch\n";
        return false;
    }
    const int nnz = static_cast<int>(P_data.size());

    // --- R: either aligned data-only or full CSR needing alignment ---
    size_t rpos = js.find("\"R\"");
    if (rpos == std::string::npos) { std::cerr << "JSON missing R object\n"; return false; }
    std::string jsR = js.substr(rpos);

    std::string R_indptr_raw, R_indices_raw, R_data_raw;
    bool have_R_indptr = json_extract_array_raw(jsR, "indptr", R_indptr_raw);
    bool have_R_indices = json_extract_array_raw(jsR, "indices", R_indices_raw);

    if (have_R_indptr && have_R_indices) {
        // Parse R CSR and align to P's CSR row-by-row
        std::vector<int> R_indptr_vec, R_indices_vec;
        std::vector<double> R_data_vec;
        if (!json_extract_array_raw(jsR, "data", R_data_raw)) { std::cerr << "JSON R.data missing\n"; return false; }

        parse_numeric_array<int>(R_indptr_raw, R_indptr_vec);
        parse_numeric_array<int>(R_indices_raw, R_indices_vec);
        parse_numeric_array<double>(R_data_raw, R_data_vec);

        if (R_indptr_vec.size() != indptr.size()) {
            std::cerr << "R.indptr length mismatch vs P.indptr (rows differ)\n";
            return false;
        }
        if (R_indices_vec.size() != R_data_vec.size()) {
            std::cerr << "R.indices and R.data length mismatch\n";
            return false;
        }

        R_data.assign(nnz, 0.0);
        for (int row = 0; row < S*A; ++row) {
            int p_beg = indptr[row];
            int p_end = indptr[row+1];
            int r_beg = R_indptr_vec[row];
            int r_end = R_indptr_vec[row+1];

            std::unordered_map<int, double> rmap;
            rmap.reserve(static_cast<size_t>(r_end - r_beg));
            for (int j = r_beg; j < r_end; ++j) rmap.emplace(R_indices_vec[j], R_data_vec[j]);

            for (int j = p_beg; j < p_end; ++j) {
                int col = indices[j];
                auto it = rmap.find(col);
                R_data[j] = (it == rmap.end()) ? 0.0 : it->second;
            }
        }
    } else {
        // R has only data: assume already aligned 1:1 with P
        if (!json_extract_array_raw(jsR, "data", R_data_raw)) { std::cerr << "JSON R.data missing\n"; return false; }
        parse_numeric_array<double>(R_data_raw, R_data);
        if (R_data.size() != static_cast<size_t>(nnz)) {
            std::cerr << "R.data length " << R_data.size() << " does not match P.nnz " << nnz << "\n";
            return false;
        }
    }

    return true;
}

// ---------- CUDA kernels (CSR) ----------

__global__ void policy_eval_kernel_csr(
    const int*    __restrict__ indptr,
    const int*    __restrict__ indices,
    const Real*   __restrict__ P_data,
    const Real*   __restrict__ R_data,
    const int*    __restrict__ policy,
    const Real*   __restrict__ V_old,
    Real*         __restrict__ V_new,
    int S, int A, Real gamma)
{
    extern __shared__ Real smem[];
    int s   = blockIdx.x;   // one block per state
    int tid = threadIdx.x;
    if (s >= S) return;

    int a   = policy[s];
    int row = s * A + a;
    int beg = indptr[row];
    int end = indptr[row + 1];

    Real acc = 0.0;
    for (int j = beg + tid; j < end; j += blockDim.x) {
        int s2 = indices[j];
        Real v = __ldg(&V_old[s2]);
        acc += P_data[j] * (R_data[j] + gamma * v);
    }
    smem[tid] = acc;
    __syncthreads();

    // reduction (blockDim must be a power of two)
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) smem[tid] += smem[tid + off];
        __syncthreads();
    }
    if (tid == 0) V_new[s] = smem[0];
}

__global__ void policy_improve_kernel_csr(
    const int*    __restrict__ indptr,
    const int*    __restrict__ indices,
    const Real*   __restrict__ P_data,
    const Real*   __restrict__ R_data,
    const Real*   __restrict__ V_old,
    const int*    __restrict__ policy_curr,
    int*          __restrict__ policy_new,
    int S, int A, Real gamma)
{
    extern __shared__ Real q_shared[];
    int s = blockIdx.x;
    int a = threadIdx.x;

    if (s >= S || a >= A) return;

    int row = s * A + a;
    int beg = indptr[row];
    int end = indptr[row + 1];

    Real q = 0.0;
    for (int j = beg; j < end; ++j) {
        int s2 = indices[j];
        q += P_data[j] * (R_data[j] + gamma * V_old[s2]);
    }
    q_shared[a] = q;
    __syncthreads();

    if (threadIdx.x == 0) {
        // Prefer current action unless an alternative improves Q by more than tie_eps
        const Real tie_eps = (Real)1e-8;
        int best_a = policy_curr[s];
        Real best_q = q_shared[best_a];
        for (int ai = 0; ai < A; ++ai) {
            if (ai == best_a) continue;
            Real q = q_shared[ai];
            if (q > best_q + tie_eps) { best_q = q; best_a = ai; }
        }
        policy_new[s] = best_a;
    }
}

__global__ void residual_kernel(
    const Real*   __restrict__ V_new,
    const Real*   __restrict__ V_old,
    Real*         __restrict__ delta,
    int S)
{
    extern __shared__ Real smem[];
    int t = threadIdx.x;
    int T = blockDim.x;

    Real max_diff = 0.0;
    for (int s = t; s < S; s += T) {
        Real d = fabs(V_new[s] - V_old[s]);
        if (d > max_diff) max_diff = d;
    }
    smem[t] = max_diff;
    __syncthreads();

    for (int off = T >> 1; off > 0; off >>= 1) {
        if (t < off) smem[t] = fmax(smem[t], smem[t + off]);
        __syncthreads();
    }
    if (t == 0) *delta = smem[0];
}

// ---------- Small util ----------
std::string extract_name_from_path(const std::string& filepath) {
    size_t last_slash = filepath.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? filepath : filepath.substr(last_slash + 1);
    size_t dot = filename.find_last_of('.');
    return (dot == std::string::npos) ? filename : filename.substr(0, dot);
}

// Return the last path component (parent directory name) of a file path.
// Example: "/a/b/gw_64x64/mdp.json" -> "gw_64x64"
static std::string parent_dir_name(const std::string& filepath) {
    if (filepath.empty()) return std::string();
    size_t last_slash = filepath.find_last_of("/\\");
    if (last_slash == std::string::npos) return std::string();
    // Find previous slash before last_slash
    size_t prev_slash = (last_slash == 0) ? std::string::npos : filepath.find_last_of("/\\", last_slash - 1);
    size_t start = (prev_slash == std::string::npos) ? 0 : prev_slash + 1;
    if (last_slash > start)
        return filepath.substr(start, last_slash - start);
    return std::string();
}

// Choose a base name for outputs given an input path.
// If the filename stem is generic (e.g., "mdp", "transitions", "rewards"),
// prefer the parent directory name (like "gw_64x64"). Otherwise use the stem.
static std::string choose_output_base(const std::string& input_path) {
    std::string stem = extract_name_from_path(input_path);
    std::string lower = stem;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "mdp" || lower == "transitions" || lower == "transition" || lower == "rewards" || lower == "reward") {
        std::string parent = parent_dir_name(input_path);
        if (!parent.empty()) return parent;
    }
    return stem;
}

// ---------- MAIN ----------
int main(int argc, char* argv[]) {
    #ifdef USE_FP32
std::cout << "Precision: FP32 (float)\n";
#else
std::cout << "Precision: FP64 (double)\n";
#endif
    // Try GPU 1 first, fall back to 0
    std::cout << "Attempting to use GPU 1...\n";
    cudaError_t device_error = cudaSetDevice(1);
    if (device_error != cudaSuccess) {
        std::cout << "GPU 1 not available, trying GPU 0...\n";
        CUDA_CHECK(cudaSetDevice(0));
    } else {
        std::cout << "Successfully selected GPU 1\n";
    }

    size_t free_mem=0, total_mem=0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        std::cout << "GPU Memory: " << (free_mem / (1024*1024)) << " MB free / "
                  << (total_mem / (1024*1024)) << " MB total\n\n";
    }

    // Params
    int S = 0, A = 0;
    double gamma = 0.9;
    const int max_eval_iters = 1000;
    const double theta = 1e-5;

    // Host CSR data
    std::vector<int>    h_indptr, h_indices;
    std::vector<double> h_P_data, h_R_data;
    std::vector<int>    h_policy, h_new_policy;
    std::vector<double> h_V_old, h_V_new;

    std::string output_name = "default";
    bool data_loaded = false;

    // Input handling
    if (argc == 2) {
        std::string filename = argv[1];
        std::string ext = (filename.find_last_of('.') != std::string::npos)
                            ? filename.substr(filename.find_last_of('.') + 1) : "";
        if (ext == "json") {
            std::cout << "Reading MDP (CSR) from JSON file: " << filename << "\n";
            if (!read_mdp_from_json_csr(filename, h_indptr, h_indices, h_P_data, h_R_data, S, A, gamma)) {
                std::cerr << "Failed to read JSON; exiting.\n";
                return 1;
            }
            data_loaded = true;
            output_name = choose_output_base(filename);
        } else {
            std::cerr << "Single argument must be a .json file. For CSV use two files.\n";
            return 1;
        }
    } else if (argc >= 3) {
        std::string trans = argv[1];
        std::string rew   = argv[2];
        std::cout << "Reading MDP (CSR) from CSV files:\n  " << trans << "\n  " << rew << "\n";
        if (!read_mdp_from_csv_csr(trans, rew, h_indptr, h_indices, h_P_data, h_R_data, S, A)) {
            std::cerr << "Failed to read CSV; exiting.\n";
            return 1;
        }
    data_loaded = true;
    output_name = choose_output_base(trans);
    } else {
        // Default tiny example (3 states, 2 actions)
        std::cout << "No files provided. Using default hardcoded CSR MDP.\n";
        S = 3; A = 2; gamma = 0.9;
        // Rows (S*A)=6; simple made-up structure with nnz=8
        h_indptr = {0, 2, 3, 5, 6, 7, 8}; // length 7
        h_indices= {0,1, 1, 1,2, 2, 2, 2};
        h_P_data = {0.8,0.2, 0.9, 0.6,0.4, 1.0, 1.0, 1.0};
        h_R_data = {5.0,0.0, 1.0, 2.0,-1.0, 2.0, 0.0, 0.0};
        data_loaded = true;
        output_name = "default";
    }

    if (!data_loaded) return 1;

    if (A > 1024) {
        std::cerr << "This simple policy_improve kernel requires A <= 1024 (got A=" << A << ").\n";
        return 1;
    }

    // Initialize policy/value
    h_policy.assign(S, 0);
    h_new_policy.resize(S);
    h_V_old.assign(S, 0.0);
    h_V_new.assign(S, 0.0);

    // Sizes
    int nnz = (int)h_P_data.size();
    std::cout << "\n--- Problem Summary ---\n";
    std::cout << "S=" << S << ", A=" << A << ", gamma=" << gamma << "\n";
    std::cout << "Rows (S*A)=" << (S*A) << ", nnz=" << nnz << "\n";

    size_t bytes_indptr = (size_t)(S*A + 1) * sizeof(int);
    size_t bytes_indices= (size_t)nnz * sizeof(int);
    size_t bytes_P      = (size_t)nnz * sizeof(Real);
    size_t bytes_R      = (size_t)nnz * sizeof(Real);
    size_t bytes_V      = (size_t)S * sizeof(Real);
    size_t bytes_pi     = (size_t)S * sizeof(int);

    // Convert host CSR data to Real for device transfers when needed
    std::vector<Real> h_P_data_real(nnz), h_R_data_real(nnz);
    for (int i = 0; i < nnz; ++i) {
        h_P_data_real[i] = static_cast<Real>(h_P_data[i]);
        h_R_data_real[i] = static_cast<Real>(h_R_data[i]);
    }
    Real gammaR = static_cast<Real>(gamma);

    // Device alloc
    int    *d_indptr=nullptr, *d_indices=nullptr;
    Real   *d_P_data=nullptr, *d_R_data=nullptr;
    Real   *d_V_old=nullptr, *d_V_new=nullptr, *d_delta=nullptr;
    int    *d_policy=nullptr, *d_new_policy=nullptr;

    CUDA_CHECK(cudaMalloc(&d_indptr, bytes_indptr));
    CUDA_CHECK(cudaMalloc(&d_indices, bytes_indices));
    CUDA_CHECK(cudaMalloc(&d_P_data, bytes_P));
    CUDA_CHECK(cudaMalloc(&d_R_data, bytes_R));
    CUDA_CHECK(cudaMalloc(&d_V_old,  bytes_V));
    CUDA_CHECK(cudaMalloc(&d_V_new,  bytes_V));
    CUDA_CHECK(cudaMalloc(&d_delta,  sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_policy, bytes_pi));
    CUDA_CHECK(cudaMalloc(&d_new_policy, bytes_pi));

    // H2D
    CUDA_CHECK(cudaMemcpy(d_indptr,  h_indptr.data(),  bytes_indptr, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), bytes_indices, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_P_data,  h_P_data_real.data(),  bytes_P,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R_data,  h_R_data_real.data(),  bytes_R,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_policy,  h_policy.data(),  bytes_pi,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_V_old, 0, bytes_V));

    // Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto cpu_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaEventRecord(start));

    // Policy Iteration
    int final_iterations = 0;
    const int T_eval = 128; // power-of-two for reduction
    for (int iter = 0; iter < 1000; ++iter) {
        // Reset V_old each policy evaluation (for consistent comparison with some CPU baselines)
        CUDA_CHECK(cudaMemset(d_V_old, 0, bytes_V));
    // Warm-start: keep V_old from previous outer iteration (no zeroing)

        // Policy evaluation
        for (int ev = 0; ev < max_eval_iters; ++ev) {
            policy_eval_kernel_csr<<<S, T_eval, T_eval * sizeof(Real)>>>(
                d_indptr, d_indices, d_P_data, d_R_data,
                d_policy, d_V_old, d_V_new, S, A, gammaR);
            CUDA_CHECK(cudaDeviceSynchronize());

            residual_kernel<<<1, T_eval, T_eval * sizeof(Real)>>>(
                d_V_new, d_V_old, d_delta, S);
            CUDA_CHECK(cudaDeviceSynchronize());

            Real delta_host = (Real)0;
            CUDA_CHECK(cudaMemcpy(&delta_host, d_delta, sizeof(Real), cudaMemcpyDeviceToHost));

            // V_old <- V_new
            CUDA_CHECK(cudaMemcpy(d_V_old, d_V_new, bytes_V, cudaMemcpyDeviceToDevice));
            if ((double)delta_host < theta) break;
        }

        // Policy improvement
        policy_improve_kernel_csr<<<S, A, A * sizeof(Real)>>>(
            d_indptr, d_indices, d_P_data, d_R_data,
            d_V_old, d_policy, d_new_policy, S, A, gammaR);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Convergence check on host
        CUDA_CHECK(cudaMemcpy(h_new_policy.data(), d_new_policy, bytes_pi, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_policy.data(),     d_policy,     bytes_pi, cudaMemcpyDeviceToHost));
        if (h_new_policy == h_policy) {
            std::cout << "Policy converged after " << (iter+1) << " iterations.\n";
            final_iterations = iter + 1;
            break;
        }
        CUDA_CHECK(cudaMemcpy(d_policy, d_new_policy, bytes_pi, cudaMemcpyDeviceToDevice));
        final_iterations = iter + 1;
    }

    // Results back
    // Copy back device values (Real) to host double buffer
    std::vector<Real> h_V_tmp(S);
    CUDA_CHECK(cudaMemcpy(h_V_tmp.data(), d_V_old, bytes_V, cudaMemcpyDeviceToHost));
    for (int i = 0; i < S; ++i) h_V_new[i] = static_cast<double>(h_V_tmp[i]);
    CUDA_CHECK(cudaMemcpy(h_policy.data(), d_policy, bytes_pi, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    auto cpu_end = std::chrono::high_resolution_clock::now();

    float gpu_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    double cpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count() / 1000.0;

    // Print results
    // std::cout << "Optimal policy: ";
    // for (int a : h_policy) std::cout << a << " ";
    // std::cout << "\nOptimal value:  ";
    // std::cout << std::fixed << std::setprecision(6);
    // for (double v : h_V_new) std::cout << v << " ";
    // std::cout << "\n";

    std::cout << "\n--- Performance Metrics ---\n";
    std::cout << "MDP dimensions: " << S << " states, " << A << " actions\n";
    std::cout << "Policy iterations completed: " << final_iterations << "\n";
    std::cout << "GPU computation time: " << gpu_ms << " ms\n";
    std::cout << "Total CPU time: " << cpu_ms << " ms\n";
    if (gpu_ms > 0) std::cout << "GPU speedup factor (CPU/GPU): " << (cpu_ms / gpu_ms) << "x\n";

    // Optional: write results
    {
        std::string out_dir = "results/";
        std::string results_path = out_dir + output_name + "-cuda.txt";
        std::string metrics_path = out_dir + output_name + "-cuda-metrics.txt";
        std::system((std::string("mkdir -p ") + out_dir).c_str());

        std::ofstream out(results_path);
        if (out) {
            out << "Optimal policy: ";
            for (int a : h_policy) out << a << " ";
            out << "\nOptimal value:  ";
            out << std::fixed << std::setprecision(6);
            for (double v : h_V_new) out << v << " ";
            out << "\n";
            std::cout << "Results written to " << results_path << "\n";
        }
        std::ofstream mout(metrics_path);
        if (mout) {
            mout << "--- Performance Metrics ---\n";
            mout << "S=" << S << ", A=" << A << ", gamma=" << gamma << "\n";
            mout << "Policy iterations: " << final_iterations << "\n";
            mout << "GPU ms: " << gpu_ms << "\n";
            mout << "CPU ms: " << cpu_ms << "\n";
            if (gpu_ms > 0) mout << "Speedup: " << (cpu_ms / gpu_ms) << "x\n";
            std::cout << "Metrics written to " << metrics_path << "\n";
        }
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_indptr);
    cudaFree(d_indices);
    cudaFree(d_P_data);
    cudaFree(d_R_data);
    cudaFree(d_V_old);
    cudaFree(d_V_new);
    cudaFree(d_delta);
    cudaFree(d_policy);
    cudaFree(d_new_policy);

    return 0;
}

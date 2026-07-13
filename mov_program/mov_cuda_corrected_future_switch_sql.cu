#include <cuda_runtime.h>
#include <sql.h>
#include <sqlext.h>

// nvcc -O2 -std=c++17 mov_cuda_corrected_future_switch_sql.cu -o mov_cuda_corrected_future_switch_sql -lodbc
// ./mov_cuda_corrected_future_switch_sql futures_data/EMiniSP500.txt

// nvcc -O2 -std=c++17 mov_cuda_corrected_future_switch_sql.cu \
//  -o mov_cuda_corrected_future_switch_sql \
//  -lodbc


#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

static constexpr int NUM_SJ = 6;   // sj = 1..6
static constexpr int NUM_LJ = 8;   // lj = 8,10,...22
static constexpr int NUM_PAIRS = NUM_SJ * NUM_LJ;

#define CHECK_CUDA(call)                                                           \
    do {                                                                           \
        cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)               \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
            std::exit(1);                                                          \
        }                                                                          \
    } while (0)

#define CHECK_ODBC_RC(rc, htype, hndl, what)                                       \
    do {                                                                           \
        if (!SQL_SUCCEEDED(rc)) {                                                  \
            print_odbc_diagnostics(htype, hndl, what);                             \
            return false;                                                          \
        }                                                                          \
    } while (0)

struct Result {
    std::vector<double> clos;
    std::vector<int> savemov;
    float gpu_ms = 0.0f;
    long long total_bisection_iters = 0;
    double estimated_total_flops = 0.0;
    double estimated_tflops = 0.0;
};

struct Bar {
    int seq;
    std::string date;
    double price;
    int idx;
};

struct TrendRow {
    int seq;
    std::string date;
    double price;
    int idx;

    std::string state;  // "unknown", "up", "down"
    std::optional<int> chg;

    std::optional<double> lph_price;
    std::optional<int> lph_seq;

    std::optional<double> ldl_price;
    std::optional<int> ldl_seq;

    std::optional<int> dsex;
    std::optional<int> wall;
    std::optional<int> d;

    std::optional<double> switch_price;
    std::optional<double> switch_diff;
};

static std::string strip_quotes(const std::string& s)
{
    std::string out = s;
    out.erase(std::remove(out.begin(), out.end(), '"'), out.end());
    return out;
}

static std::string get_env_or_fail(const char* name)
{
    const char* val = std::getenv(name);
    if (!val) {
        std::cerr << "Missing required environment variable: " << name << "\n";
        std::exit(1);
    }
    return std::string(val);
}

static std::string basename_from_path(const std::string& filename)
{
    size_t pos1 = filename.find_last_of('/');
    size_t pos2 = filename.find_last_of('\\');
    size_t pos = std::string::npos;

    if (pos1 == std::string::npos) pos = pos2;
    else if (pos2 == std::string::npos) pos = pos1;
    else pos = std::max(pos1, pos2);

    if (pos == std::string::npos) return filename;
    return filename.substr(pos + 1);
}

static std::string fmt_opt_int(const std::optional<int>& v)
{
    return v.has_value() ? std::to_string(*v) : "None";
}

static std::string fmt_opt_double(const std::optional<double>& v, int prec = 3)
{
    if (!v.has_value()) {
        return "None";
    }
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(prec) << *v;
    return oss.str();
}

__global__ void eval_pairs_kernel(
    const double* clos,
    int M,
    int* pairAvel,
    int* pairAves)
{
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= NUM_PAIRS) return;

    int sj = 1 + (pair_id / NUM_LJ);
    int lj = 8 + 2 * (pair_id % NUM_LJ);

    int* outAvel = pairAvel + static_cast<size_t>(pair_id) * M;
    int* outAves = pairAves + static_cast<size_t>(pair_id) * M;

    double shortSum = 0.0;
    for (int i = 0; i < sj; ++i) {
        shortSum += clos[i];
    }

    double longSum = 0.0;
    for (int i = 0; i < lj; ++i) {
        longSum += clos[i];
    }

    double s = 0.0;
    double l = 0.0;
    bool s_valid = false;
    bool l_valid = false;
    char sl = ' ';

    for (int i = 0; i < M; ++i) {
        if (i == sj - 1) {
            s = shortSum / static_cast<double>(sj);
            s_valid = true;
        } else if (i >= sj) {
            s = (s * static_cast<double>(sj) - clos[i - sj] + clos[i]) / static_cast<double>(sj);
            s_valid = true;
        } else {
            s_valid = false;
        }

        if (i == lj - 1) {
            l = longSum / static_cast<double>(lj);
            l_valid = true;
        } else if (i >= lj) {
            l = (l * static_cast<double>(lj) - clos[i - lj] + clos[i]) / static_cast<double>(lj);
            l_valid = true;
        } else {
            l_valid = false;
        }

        if (!s_valid || !l_valid) {
            continue;
        }

        if (s > l && sl != 'L') {
            sl = 'L';
            outAvel[i] = 1;
        }

        if (s < l && sl != 'S') {
            sl = 'S';
            outAves[i] = -1;
        }
    }
}

__global__ void reduce_pairs_kernel(
    const int* pairAvel,
    const int* pairAves,
    int M,
    int* avel,
    int* aves)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;

    int sumL = 0;
    int sumS = 0;

    for (int p = 0; p < NUM_PAIRS; ++p) {
        sumL += pairAvel[static_cast<size_t>(p) * M + i];
        sumS += pairAves[static_cast<size_t>(p) * M + i];
    }

    avel[i] = sumL;
    aves[i] = sumS;
}

__global__ void build_savemov_kernel(
    const int* avel,
    const int* aves,
    int M,
    int* savemov)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int ik = 0;
    int jk = 0;
    for (int i = 0; i < M; ++i) {
        ik += avel[i];
        jk += aves[i];
        savemov[i] = ik + jk;
    }

    int savelow = 0;
    for (int i = 0; i < M; ++i) {
        if (savelow > savemov[i]) {
            savelow = savemov[i];
        }
    }

    if (savelow < 0) {
        for (int i = 0; i < M; ++i) {
            savemov[i] -= savelow;
        }
    }
}

static bool parse_number_token(std::string token, double& value)
{
    token = strip_quotes(token);
    token.erase(std::remove(token.begin(), token.end(), ','), token.end());

    try {
        size_t used = 0;
        value = std::stod(token, &used);
        return used == token.size();
    } catch (...) {
        return false;
    }
}

static bool load_input_file(
    const std::string& filename,
    std::vector<std::string>& dates,
    std::vector<double>& clos,
    int& M)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    dates.clear();
    clos.clear();

    std::string line;
    int line_no = 0;
    int skipped = 0;

    while (std::getline(fin, line)) {
        ++line_no;

        // Remove comments, if present.  This prevents numbers in comments from
        // changing which numeric field is considered "second-to-last".
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }

        if (line.empty()) {
            ++skipped;
            continue;
        }

        // Tokenize by ordinary file separators.  This works for space-, tab-,
        // and comma-separated market data files.
        for (char& c : line) {
            if (c == ',' || c == '\t' || c == '\r') {
                c = ' ';
            }
        }

        std::istringstream iss(line);
        std::string date;
        if (!(iss >> date)) {
            ++skipped;
            continue;
        }

        std::vector<double> numbers;
        std::string token;
        while (iss >> token) {
            double x = 0.0;
            if (parse_number_token(token, x)) {
                numbers.push_back(x);
            }
        }

        // Need at least two numeric values because the requested data field is
        // the second-to-last number on the line.
        if (numbers.size() < 2) {
            std::cerr << "Skipping line " << line_no
                      << ": fewer than two numeric fields after date: "
                      << line << std::endl;
            ++skipped;
            continue;
        }

        double selected_value = numbers[numbers.size() - 2];

        dates.push_back(date);
        clos.push_back(selected_value);

#ifdef DEBUG_INPUT_PARSE
        std::cerr << "line " << line_no
                  << " date=" << date
                  << " selected=" << selected_value
                  << " numeric_fields=" << numbers.size()
                  << std::endl;
#endif
    }

    M = static_cast<int>(clos.size());

    std::cout << "Loaded " << M << " data rows from " << filename << std::endl;
    if (skipped > 0) {
        std::cout << "Skipped " << skipped << " non-data rows." << std::endl;
    }

    return M > 0;
}

static void compute_savemov_cpu(
    const std::vector<double>& clos,
    std::vector<int>& savemov)
{
    int M = static_cast<int>(clos.size());
    savemov.assign(M, 0);
    std::vector<int> avel(M, 0);
    std::vector<int> aves(M, 0);

    for (int sj = 1; sj < 7; ++sj) {
        for (int lj = 8; lj < 24; lj += 2) {
            double shortSum = 0.0;
            for (int i = 0; i < sj && i < M; ++i) shortSum += clos[i];

            double longSum = 0.0;
            for (int i = 0; i < lj && i < M; ++i) longSum += clos[i];

            double s = 0.0;
            double l = 0.0;
            bool s_valid = false;
            bool l_valid = false;
            char sl = ' ';

            for (int i = 0; i < M; ++i) {
                if (i == sj - 1) {
                    s = shortSum / static_cast<double>(sj);
                    s_valid = true;
                } else if (i >= sj) {
                    s = (s * static_cast<double>(sj) - clos[i - sj] + clos[i]) / static_cast<double>(sj);
                    s_valid = true;
                } else {
                    s_valid = false;
                }

                if (i == lj - 1) {
                    l = longSum / static_cast<double>(lj);
                    l_valid = true;
                } else if (i >= lj) {
                    l = (l * static_cast<double>(lj) - clos[i - lj] + clos[i]) / static_cast<double>(lj);
                    l_valid = true;
                } else {
                    l_valid = false;
                }

                if (!s_valid || !l_valid) continue;

                if (s > l && sl != 'L') {
                    sl = 'L';
                    avel[i] += 1;
                }
                if (s < l && sl != 'S') {
                    sl = 'S';
                    aves[i] -= 1;
                }
            }
        }
    }

    int ik = 0;
    int jk = 0;
    for (int i = 0; i < M; ++i) {
        ik += avel[i];
        jk += aves[i];
        savemov[i] = ik + jk;
    }

    int savelow = 0;
    for (int x : savemov) {
        if (savelow > x) savelow = x;
    }
    if (savelow < 0) {
        for (int& x : savemov) x -= savelow;
    }
}

static Result run_cpu_actual(const std::vector<double>& input_clos)
{
    Result r;
    r.clos = input_clos;
    compute_savemov_cpu(r.clos, r.savemov);
    return r;
}

static Result run_gpu_actual(const std::vector<double>& input_clos)
{
    Result r;
    r.clos = input_clos;
    int M = static_cast<int>(input_clos.size());
    r.savemov.assign(M, 0);

    double* d_clos = nullptr;
    int* d_pairAvel = nullptr;
    int* d_pairAves = nullptr;
    int* d_avel = nullptr;
    int* d_aves = nullptr;
    int* d_savemov = nullptr;

    size_t clos_bytes = static_cast<size_t>(M) * sizeof(double);
    size_t vec_int_bytes = static_cast<size_t>(M) * sizeof(int);
    size_t pair_int_bytes = static_cast<size_t>(NUM_PAIRS) * static_cast<size_t>(M) * sizeof(int);

    CHECK_CUDA(cudaMalloc(&d_clos, clos_bytes));
    CHECK_CUDA(cudaMalloc(&d_pairAvel, pair_int_bytes));
    CHECK_CUDA(cudaMalloc(&d_pairAves, pair_int_bytes));
    CHECK_CUDA(cudaMalloc(&d_avel, vec_int_bytes));
    CHECK_CUDA(cudaMalloc(&d_aves, vec_int_bytes));
    CHECK_CUDA(cudaMalloc(&d_savemov, vec_int_bytes));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    CHECK_CUDA(cudaEventRecord(ev_start));
    CHECK_CUDA(cudaMemcpy(d_clos, input_clos.data(), clos_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_pairAvel, 0, pair_int_bytes));
    CHECK_CUDA(cudaMemset(d_pairAves, 0, pair_int_bytes));
    CHECK_CUDA(cudaMemset(d_avel, 0, vec_int_bytes));
    CHECK_CUDA(cudaMemset(d_aves, 0, vec_int_bytes));
    CHECK_CUDA(cudaMemset(d_savemov, 0, vec_int_bytes));

    {
        const int threads = 64;
        const int blocks = (NUM_PAIRS + threads - 1) / threads;
        eval_pairs_kernel<<<blocks, threads>>>(d_clos, M, d_pairAvel, d_pairAves);
        CHECK_CUDA(cudaGetLastError());
    }
    {
        const int threads = 128;
        const int blocks = (M + threads - 1) / threads;
        reduce_pairs_kernel<<<blocks, threads>>>(d_pairAvel, d_pairAves, M, d_avel, d_aves);
        CHECK_CUDA(cudaGetLastError());
    }
    build_savemov_kernel<<<1, 1>>>(d_avel, d_aves, M, d_savemov);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(r.savemov.data(), d_savemov, vec_int_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(ev_stop));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));
    CHECK_CUDA(cudaEventElapsedTime(&r.gpu_ms, ev_start, ev_stop));

    r.estimated_total_flops = static_cast<double>(NUM_PAIRS) * static_cast<double>(M) * 12.0;
    double gpu_seconds = r.gpu_ms / 1000.0;
    r.estimated_tflops = (gpu_seconds > 0.0) ? (r.estimated_total_flops / gpu_seconds / 1.0e12) : 0.0;

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaFree(d_clos));
    CHECK_CUDA(cudaFree(d_pairAvel));
    CHECK_CUDA(cudaFree(d_pairAves));
    CHECK_CUDA(cudaFree(d_avel));
    CHECK_CUDA(cudaFree(d_aves));
    CHECK_CUDA(cudaFree(d_savemov));

    return r;
}

static int future_idx_for_candidate(const std::vector<double>& clos, double candidate_price)
{
    std::vector<double> test_clos = clos;
    test_clos.push_back(candidate_price);
    std::vector<int> test_savemov;
    compute_savemov_cpu(test_clos, test_savemov);
    return test_savemov.back();
}

struct FutureSwitch {
    bool found = false;
    std::string direction;
    double price = std::numeric_limits<double>::quiet_NaN();
    int current_idx = 0;
    int future_idx = 0;
    long long iterations = 0;
    std::string note;
};

static FutureSwitch find_next_switch_price(
    const std::vector<double>& clos,
    int current_idx,
    const std::string& current_state,
    double precision = 0.1)
{
    FutureSwitch fs;
    fs.current_idx = current_idx;

    if (clos.empty()) {
        fs.note = "No close data available.";
        return fs;
    }
    if (current_state != "up" && current_state != "down") {
        fs.note = "Current trend is unknown; no flip target was computed.";
        return fs;
    }

    const double last_price = clos.back();
    if (last_price <= 0.0) {
        fs.note = "Last price is not positive; automatic bracket was not computed.";
        return fs;
    }

    auto crosses = [&](double price) {
        int idx = future_idx_for_candidate(clos, price);
        if (current_state == "up") {
            return idx < current_idx;
        }
        return idx > current_idx;
    };

    double lo = last_price * 0.50;
    double hi = last_price * 1.50;

    if (current_state == "up") {
        // An up trend flips down by finding a sufficiently lower next price.
        fs.direction = "up_to_down";
        while (!crosses(lo) && lo > 1e-9 && fs.iterations < 40) {
            lo *= 0.5;
            ++fs.iterations;
        }
        if (!crosses(lo)) {
            fs.note = "No downward flip found inside the expanded lower bracket.";
            return fs;
        }

        double a = lo;          // crosses
        double b = last_price;  // usually does not cross
        while (std::abs(b - a) > precision && fs.iterations < 200) {
            double mid = 0.5 * (a + b);
            if (crosses(mid)) {
                a = mid;
            } else {
                b = mid;
            }
            ++fs.iterations;
        }
        fs.price = b;
        fs.future_idx = future_idx_for_candidate(clos, fs.price);
        fs.found = fs.future_idx < current_idx;
        if (!fs.found) {
            fs.price = a;
            fs.future_idx = future_idx_for_candidate(clos, fs.price);
            fs.found = fs.future_idx < current_idx;
        }
        return fs;
    }

    // A down trend flips up by finding a sufficiently higher next price.
    fs.direction = "down_to_up";
    while (!crosses(hi) && fs.iterations < 40) {
        hi *= 1.5;
        ++fs.iterations;
    }
    if (!crosses(hi)) {
        fs.note = "No upward flip found inside the expanded upper bracket.";
        return fs;
    }

    double a = last_price; // usually does not cross
    double b = hi;         // crosses
    while (std::abs(b - a) > precision && fs.iterations < 200) {
        double mid = 0.5 * (a + b);
        if (crosses(mid)) {
            b = mid;
        } else {
            a = mid;
        }
        ++fs.iterations;
    }
    fs.price = b;
    fs.future_idx = future_idx_for_candidate(clos, fs.price);
    fs.found = fs.future_idx > current_idx;
    return fs;
}

static bool compare_results(const Result& cpu, const Result& gpu, int M)
{
    bool ok = true;
    for (int i = 0; i < M; ++i) {
        if (cpu.clos[i] != gpu.clos[i]) {
            std::cout << "Mismatch in clos at i=" << i
                      << " cpu=" << std::setprecision(17) << cpu.clos[i]
                      << " gpu=" << std::setprecision(17) << gpu.clos[i]
                      << std::endl;
            ok = false;
            break;
        }
    }
    for (int i = 0; i < M; ++i) {
        if (cpu.savemov[i] != gpu.savemov[i]) {
            std::cout << "Mismatch in savemov at i=" << i
                      << " cpu=" << cpu.savemov[i]
                      << " gpu=" << gpu.savemov[i]
                      << std::endl;
            ok = false;
            break;
        }
    }
    return ok;
}

static void print_input_tail_report(
    const std::vector<std::string>& dates,
    const std::vector<double>& input_clos,
    const Result& gpu,
    const Result& cpu,
    int tail_count = 12)
{
    int M = static_cast<int>(input_clos.size());
    int start = std::max(0, M - tail_count);

    std::cout << "\n=== INPUT READBACK CHECK: LAST " << (M - start) << " ROWS ===\n";
    std::cout << "seq  date          input_close       gpu_close_after_run cpu_close_after_run\n";

    for (int i = start; i < M; ++i) {
        std::cout << std::setw(4) << i << "  "
                  << std::setw(10) << strip_quotes(dates[i]) << "  "
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << input_clos[i] << "  "
                  << std::setw(19) << gpu.clos[i] << "  "
                  << std::setw(19) << cpu.clos[i];

        if (input_clos[i] != gpu.clos[i] || input_clos[i] != cpu.clos[i]) {
            std::cout << "   <-- changed after processing";
        }
        std::cout << "\n";
    }

    std::cout << "\nOriginal final input value = " << std::fixed << std::setprecision(4)
              << input_clos[M - 1] << "\n";
    std::cout << "GPU final working value    = " << std::fixed << std::setprecision(4)
              << gpu.clos[M - 1] << "\n";
    std::cout << "CPU final working value    = " << std::fixed << std::setprecision(4)
              << cpu.clos[M - 1] << "\n";
}

static std::vector<Bar> build_bars(
    const std::vector<std::string>& dates,
    const std::vector<double>& clos,
    const std::vector<int>& savemov)
{
    std::vector<Bar> bars;
    bars.reserve(clos.size());

    for (size_t i = 0; i < clos.size(); ++i) {
        bars.push_back(Bar{
            static_cast<int>(i),
            strip_quotes(dates[i]),
            clos[i],
            savemov[i]
        });
    }

    return bars;
}

static std::vector<TrendRow> analyze_trend_cpp(const std::vector<Bar>& bars)
{
    std::vector<TrendRow> results;
    results.reserve(bars.size());

    std::string state = "unknown";
    std::optional<int> last_idx;

    std::optional<double> up_high_price;
    std::optional<int> up_high_seq;

    std::optional<double> down_low_price;
    std::optional<int> down_low_seq;

    std::optional<int> chg;
    std::optional<int> wall;
    std::optional<double> switch_price;

    for (size_t i = 0; i < bars.size(); ++i) {
        const auto& bar = bars[i];

        if (!last_idx.has_value()) {
            last_idx = bar.idx;

            results.push_back(TrendRow{
                bar.seq,
                bar.date,
                bar.price,
                bar.idx,
                state,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                std::nullopt
            });
            continue;
        }

        bool trend_changed = false;
        std::string prior_state = state;

        if (bar.idx > *last_idx) {
            if (state != "up") {
                state = "up";
                chg = bar.seq;
                trend_changed = true;
            }
        } else if (bar.idx < *last_idx) {
            if (state != "down") {
                state = "down";
                chg = bar.seq;
                trend_changed = true;
            }
        }

        if (prior_state == "up") {
            if (!up_high_price.has_value() || bar.price > *up_high_price) {
                up_high_price = bar.price;
                up_high_seq = bar.seq;
            }
        } else if (prior_state == "down") {
            if (!down_low_price.has_value() || bar.price < *down_low_price) {
                down_low_price = bar.price;
                down_low_seq = bar.seq;
            }
        }

        if (trend_changed) {
            switch_price = bar.price;

            if (state == "up") {
                up_high_price = bar.price;
                up_high_seq = bar.seq;
            } else if (state == "down") {
                down_low_price = bar.price;
                down_low_seq = bar.seq;
            }
        }

        last_idx = bar.idx;

        std::optional<int> extreme_seq;
        if (state == "down") {
            extreme_seq = up_high_seq;
        } else if (state == "up") {
            extreme_seq = down_low_seq;
        }

        std::optional<int> dsex;
        if (extreme_seq.has_value()) {
            dsex = bar.seq - *extreme_seq;
        }

        std::optional<int> d;
        if (trend_changed) {
            wall = dsex;
            int k = 0;
            if (wall.has_value()) {
                d = *wall - k;
            }
        } else {
            std::optional<int> k;
            if (chg.has_value()) {
                k = bar.seq - *chg;
            }
            if (wall.has_value() && k.has_value()) {
                d = *wall - *k;
            }
        }

        std::optional<double> switch_diff;
        if (switch_price.has_value()) {
            double raw_diff = bar.price - *switch_price;
            switch_diff = (state == "down") ? std::abs(raw_diff) : raw_diff;
        }

        results.push_back(TrendRow{
            bar.seq,
            bar.date,
            bar.price,
            bar.idx,
            state,
            chg,
            (state == "down") ? up_high_price : std::nullopt,
            (state == "down") ? up_high_seq   : std::nullopt,
            (state == "up")   ? down_low_price : std::nullopt,
            (state == "up")   ? down_low_seq   : std::nullopt,
            dsex,
            wall,
            d,
            switch_price,
            switch_diff
        });
    }

    return results;
}

static void print_trend_reports(const std::vector<TrendRow>& results)
{
    std::string last_state;

    for (const auto& r : results) {
        if (!last_state.empty() && r.state != last_state) {
            std::cout << std::endl;
        }

        std::cout
            << r.seq << "  "
            << r.date << "  "
            << r.price << "  "
            << r.idx << "  "
            << "state=" << r.state << "  "
            << "chg=" << fmt_opt_int(r.chg) << "  "
            << "lph=" << fmt_opt_double(r.lph_price) << " at seq " << fmt_opt_int(r.lph_seq) << "  "
            << "ldl=" << fmt_opt_double(r.ldl_price) << " at seq " << fmt_opt_int(r.ldl_seq) << "  "
            << "dsex=" << fmt_opt_int(r.dsex) << "  "
            << "wall=" << fmt_opt_int(r.wall) << "  "
            << "d=" << fmt_opt_int(r.d)
            << std::endl;

        std::cout
            << "    switch_price=" << fmt_opt_double(r.switch_price)
            << "  switch_diff=" << fmt_opt_double(r.switch_diff)
            << std::endl;

        last_state = r.state;
    }

    std::cout << "\n=== SUMMARY TABLE (seq, date, price, idx, state, wall, d, switch_price, switch_diff) ===\n";

    last_state.clear();

    for (const auto& r : results) {
        if (!last_state.empty() && r.state != last_state) {
            std::cout << std::endl;
        }

        std::string wall_val = fmt_opt_int(r.wall);
        std::string d_val = fmt_opt_int(r.d);
        std::string switch_price_val = fmt_opt_double(r.switch_price);
        std::string switch_diff_val = fmt_opt_double(r.switch_diff);

        std::cout
            << std::setw(5) << r.seq << "  "
            << r.date << "  "
            << std::setw(10) << std::fixed << std::setprecision(4) << r.price << "  "
            << std::setw(3) << r.idx << "  "
            << std::setw(6) << r.state << "  "
            << std::setw(8) << wall_val << "  "
            << std::setw(8) << d_val << "  "
            << switch_price_val << "  "
            << switch_diff_val
            << std::endl;

        last_state = r.state;
    }
}

static double correlation(const std::vector<double>& a, const std::vector<double>& b)
{
    const size_t n = a.size();
    if (n < 2 || b.size() != n) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double mean_a = std::accumulate(a.begin(), a.end(), 0.0) / static_cast<double>(n);
    double mean_b = std::accumulate(b.begin(), b.end(), 0.0) / static_cast<double>(n);

    double num = 0.0;
    double den_a = 0.0;
    double den_b = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double da = a[i] - mean_a;
        double db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if (den_a == 0.0 || den_b == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return num / std::sqrt(den_a * den_b);
}

static void compute_correlations_cpp(const std::vector<TrendRow>& results)
{
    std::vector<double> price;
    std::vector<double> wall;
    std::vector<double> d;

    for (const auto& r : results) {
        if (r.wall.has_value() && r.d.has_value()) {
            price.push_back(r.price);
            wall.push_back(static_cast<double>(*r.wall));
            d.push_back(static_cast<double>(*r.d));
        }
    }

    double corr_pw = correlation(price, wall);
    double corr_wd = correlation(wall, d);
    double corr_pd = correlation(price, d);

    std::cout << "\n=== CORRELATIONS ===\n";
    std::cout << "Correlation(price, wall) = " << std::fixed << std::setprecision(6) << corr_pw << "\n";
    std::cout << "Correlation(wall, d)    = " << std::fixed << std::setprecision(6) << corr_wd << "\n";
    std::cout << "Correlation(price, d)   = " << std::fixed << std::setprecision(6) << corr_pd << "\n";
}



static void print_future_switch_report(const FutureSwitch& fs, double last_price)
{
    std::cout << "\n=== NEXT-READING SWITCH PRICE ===\n";
    if (!fs.found) {
        std::cout << "No switch price found. " << fs.note << "\n";
        return;
    }

    std::cout << "Current mov index: " << fs.current_idx << "\n";
    std::cout << "Direction tested: " << fs.direction << "\n";
    std::cout << "Last real price: " << std::fixed << std::setprecision(4) << last_price << "\n";
    std::cout << "Hypothetical next price that flips trend: "
              << std::fixed << std::setprecision(4) << fs.price << "\n";
    std::cout << "Future mov index at that price: " << fs.future_idx << "\n";
    std::cout << "Difference from last real price: "
              << std::fixed << std::setprecision(4) << (fs.price - last_price) << "\n";
    std::cout << "Bisection/test iterations: " << fs.iterations << "\n";
}

static void print_odbc_diagnostics(SQLSMALLINT handleType, SQLHANDLE handle, const char* context)
{
    std::cerr << "\nODBC error in: " << context << "\n";

    SQLSMALLINT rec = 1;
    SQLINTEGER nativeError = 0;
    SQLCHAR sqlState[7] = {0};
    SQLCHAR messageText[1024] = {0};
    SQLSMALLINT textLength = 0;

    while (true) {
        SQLRETURN rc = SQLGetDiagRec(
            handleType,
            handle,
            rec,
            sqlState,
            &nativeError,
            messageText,
            sizeof(messageText),
            &textLength);

        if (rc == SQL_NO_DATA) break;
        if (!SQL_SUCCEEDED(rc)) break;

        std::cerr << "  [" << sqlState << "] "
                  << "native=" << nativeError
                  << " message=" << messageText << "\n";
        ++rec;
    }
}

static bool insert_trend_rows_to_sql_server(
    const std::string& asset,
    const std::vector<std::string>& dates,
    const std::vector<double>& clos,
    const std::vector<int>& savemov,
    const std::vector<TrendRow>& rows)
{
    if (!(dates.size() == clos.size() &&
          clos.size() == savemov.size() &&
          savemov.size() == rows.size())) {
        std::cerr << "insert_trend_rows_to_sql_server: vector size mismatch.\n";
        return false;
    }

    const std::string server   = get_env_or_fail("MSSQL_SERVER");
    const std::string database = get_env_or_fail("MSSQL_DATABASE");
    const std::string username = get_env_or_fail("MSSQL_USERNAME");
    const std::string password = get_env_or_fail("MSSQL_PASSWORD");

    std::string connStr =
        "DRIVER={ODBC Driver 18 for SQL Server};"
        "SERVER=" + server + ";"
        "DATABASE=" + database + ";"
        "UID=" + username + ";"
        "PWD=" + password + ";"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;";

    SQLHENV env = SQL_NULL_HENV;
    SQLHDBC dbc = SQL_NULL_HDBC;
    SQLHSTMT existsStmt = SQL_NULL_HSTMT;
    SQLHSTMT insertStmt = SQL_NULL_HSTMT;
    SQLHSTMT verifyStmt = SQL_NULL_HSTMT;
    SQLRETURN rc;

    rc = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env);
    if (!SQL_SUCCEEDED(rc)) {
        std::cerr << "Failed to allocate ODBC environment handle.\n";
        return false;
    }

    rc = SQLSetEnvAttr(env, SQL_ATTR_ODBC_VERSION, (void*)SQL_OV_ODBC3, 0);
    CHECK_ODBC_RC(rc, SQL_HANDLE_ENV, env, "SQLSetEnvAttr(SQL_OV_ODBC3)");

    rc = SQLAllocHandle(SQL_HANDLE_DBC, env, &dbc);
    CHECK_ODBC_RC(rc, SQL_HANDLE_ENV, env, "SQLAllocHandle(SQL_HANDLE_DBC)");

    {
        SQLCHAR outConnStr[1024] = {0};
        SQLSMALLINT outLen = 0;
        rc = SQLDriverConnect(dbc, nullptr, (SQLCHAR*)connStr.c_str(), SQL_NTS,
                              outConnStr, sizeof(outConnStr), &outLen,
                              SQL_DRIVER_NOPROMPT);
        CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLDriverConnect");
    }

    rc = SQLSetConnectAttr(dbc, SQL_ATTR_AUTOCOMMIT, (SQLPOINTER)SQL_AUTOCOMMIT_OFF, 0);
    CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLSetConnectAttr(SQL_ATTR_AUTOCOMMIT)");

    rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &existsStmt);
    CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLAllocHandle(existsStmt)");
    rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &insertStmt);
    CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLAllocHandle(insertStmt)");

    const char* existsSql =
        "SELECT COUNT(*) FROM dbo.data_asset WHERE asset = ? AND trade_date = ?";

    const char* insertSql =
        "INSERT INTO dbo.data_asset ("
        "asset, seq_no, trade_date, close_price, savemov, state, "
        "chg, lph, lph_seq, ldl, ldl_seq, dsex, wall, d, switch_price, switch_diff"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    rc = SQLPrepare(existsStmt, (SQLCHAR*)existsSql, SQL_NTS);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "SQLPrepare(exists row check)");
    rc = SQLPrepare(insertStmt, (SQLCHAR*)insertSql, SQL_NTS);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "SQLPrepare(insert row)");

    char       p_asset[256] = {0};
    SQLINTEGER p_seq_no = 0;
    char       p_trade_date[32] = {0};
    SQLDOUBLE  p_close_price = 0.0;
    SQLINTEGER p_savemov = 0;
    char       p_state[16] = {0};

    SQLINTEGER p_chg = 0;
    SQLDOUBLE  p_lph = 0.0;
    SQLINTEGER p_lph_seq = 0;
    SQLDOUBLE  p_ldl = 0.0;
    SQLINTEGER p_ldl_seq = 0;
    SQLINTEGER p_dsex = 0;
    SQLINTEGER p_wall = 0;
    SQLINTEGER p_d = 0;
    SQLDOUBLE  p_switch_price = 0.0;
    SQLDOUBLE  p_switch_diff = 0.0;

    SQLLEN ind_asset = SQL_NTS;
    SQLLEN ind_seq_no = 0;
    SQLLEN ind_trade_date = SQL_NTS;
    SQLLEN ind_close_price = 0;
    SQLLEN ind_savemov = 0;
    SQLLEN ind_state = SQL_NTS;

    SQLLEN ind_chg = SQL_NULL_DATA;
    SQLLEN ind_lph = SQL_NULL_DATA;
    SQLLEN ind_lph_seq = SQL_NULL_DATA;
    SQLLEN ind_ldl = SQL_NULL_DATA;
    SQLLEN ind_ldl_seq = SQL_NULL_DATA;
    SQLLEN ind_dsex = SQL_NULL_DATA;
    SQLLEN ind_wall = SQL_NULL_DATA;
    SQLLEN ind_d = SQL_NULL_DATA;
    SQLLEN ind_switch_price = SQL_NULL_DATA;
    SQLLEN ind_switch_diff = SQL_NULL_DATA;

    int col = 1;
    rc = SQLBindParameter(existsStmt, col++, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 255, 0,
                          p_asset, sizeof(p_asset), &ind_asset);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "Bind exists asset");
    rc = SQLBindParameter(existsStmt, col++, SQL_PARAM_INPUT, SQL_C_CHAR, SQL_VARCHAR, 31, 0,
                          p_trade_date, sizeof(p_trade_date), &ind_trade_date);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "Bind exists trade_date");

    col = 1;
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_CHAR,   SQL_VARCHAR, 255, 0, p_asset,         sizeof(p_asset), &ind_asset);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert asset");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_seq_no,        0, &ind_seq_no);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert seq_no");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_CHAR,   SQL_VARCHAR, 31, 0, p_trade_date,     sizeof(p_trade_date), &ind_trade_date);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert trade_date");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_DOUBLE, SQL_DECIMAL, 18, 4, &p_close_price,   0, &ind_close_price);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert close_price");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_savemov,       0, &ind_savemov);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert savemov");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_CHAR,   SQL_VARCHAR, 15, 0, p_state,          sizeof(p_state), &ind_state);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert state");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_chg,           0, &ind_chg);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert chg");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_DOUBLE, SQL_DECIMAL, 18, 4, &p_lph,           0, &ind_lph);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert lph");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_lph_seq,       0, &ind_lph_seq);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert lph_seq");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_DOUBLE, SQL_DECIMAL, 18, 4, &p_ldl,           0, &ind_ldl);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert ldl");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_ldl_seq,       0, &ind_ldl_seq);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert ldl_seq");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_dsex,          0, &ind_dsex);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert dsex");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_wall,          0, &ind_wall);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert wall");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_SLONG,  SQL_INTEGER, 0, 0, &p_d,             0, &ind_d);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert d");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_DOUBLE, SQL_DECIMAL, 18, 4, &p_switch_price,  0, &ind_switch_price);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert switch_price");
    rc = SQLBindParameter(insertStmt, col++, SQL_PARAM_INPUT, SQL_C_DOUBLE, SQL_DECIMAL, 18, 4, &p_switch_diff,   0, &ind_switch_diff);
    CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "Bind insert switch_diff");

    std::cout << "\nInserting trend rows into mov.dbo.data_asset...\n";
    std::cout << "Duplicate policy: check (asset, trade_date); skip duplicates; continue through all rows.\n";

    long long rows_tested = 0;
    long long rows_inserted = 0;
    long long rows_not_inserted = 0;

    std::snprintf(p_asset, sizeof(p_asset), "%s", asset.c_str());

    for (size_t i = 0; i < rows.size(); ++i) {
        const TrendRow& r = rows[i];
        ++rows_tested;

        p_seq_no = static_cast<SQLINTEGER>(r.seq);
        std::snprintf(p_trade_date, sizeof(p_trade_date), "%s", strip_quotes(dates[i]).c_str());
        p_close_price = clos[i];
        p_savemov = savemov[i];
        std::snprintf(p_state, sizeof(p_state), "%s", r.state.c_str());

        if (r.chg)          { p_chg = *r.chg; ind_chg = 0; } else ind_chg = SQL_NULL_DATA;
        if (r.lph_price)    { p_lph = *r.lph_price; ind_lph = 0; } else ind_lph = SQL_NULL_DATA;
        if (r.lph_seq)      { p_lph_seq = *r.lph_seq; ind_lph_seq = 0; } else ind_lph_seq = SQL_NULL_DATA;
        if (r.ldl_price)    { p_ldl = *r.ldl_price; ind_ldl = 0; } else ind_ldl = SQL_NULL_DATA;
        if (r.ldl_seq)      { p_ldl_seq = *r.ldl_seq; ind_ldl_seq = 0; } else ind_ldl_seq = SQL_NULL_DATA;
        if (r.dsex)         { p_dsex = *r.dsex; ind_dsex = 0; } else ind_dsex = SQL_NULL_DATA;
        if (r.wall)         { p_wall = *r.wall; ind_wall = 0; } else ind_wall = SQL_NULL_DATA;
        if (r.d)            { p_d = *r.d; ind_d = 0; } else ind_d = SQL_NULL_DATA;
        if (r.switch_price) { p_switch_price = *r.switch_price; ind_switch_price = 0; } else ind_switch_price = SQL_NULL_DATA;
        if (r.switch_diff)  { p_switch_diff = *r.switch_diff; ind_switch_diff = 0; } else ind_switch_diff = SQL_NULL_DATA;

        rc = SQLExecute(existsStmt);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "SQLExecute(exists row check)");

        SQLLEN existing_count = 0;
        rc = SQLFetch(existsStmt);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "SQLFetch(exists row check)");
        rc = SQLGetData(existsStmt, 1, SQL_C_SLONG, &existing_count, 0, nullptr);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "SQLGetData(exists count)");
        rc = SQLFreeStmt(existsStmt, SQL_CLOSE);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, existsStmt, "SQLFreeStmt(existsStmt SQL_CLOSE)");

        if (existing_count > 0) {
            ++rows_not_inserted;
            continue;
        }

        rc = SQLExecute(insertStmt);
        if (!SQL_SUCCEEDED(rc)) {
            SQLINTEGER nativeError = 0;
            SQLCHAR sqlState[7] = {0};
            SQLCHAR messageText[1024] = {0};
            SQLSMALLINT textLength = 0;
            SQLRETURN diagRc = SQLGetDiagRec(SQL_HANDLE_STMT, insertStmt, 1,
                                             sqlState, &nativeError,
                                             messageText, sizeof(messageText), &textLength);
            if (SQL_SUCCEEDED(diagRc) && (nativeError == 2601 || nativeError == 2627)) {
                ++rows_not_inserted;
                rc = SQLFreeStmt(insertStmt, SQL_CLOSE);
                CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "SQLFreeStmt(insert duplicate SQL_CLOSE)");
                continue;
            }

            print_odbc_diagnostics(SQL_HANDLE_STMT, insertStmt, "SQLExecute(insert row)");
            SQLEndTran(SQL_HANDLE_DBC, dbc, SQL_ROLLBACK);
            SQLFreeHandle(SQL_HANDLE_STMT, existsStmt);
            SQLFreeHandle(SQL_HANDLE_STMT, insertStmt);
            SQLFreeHandle(SQL_HANDLE_DBC, dbc);
            SQLFreeHandle(SQL_HANDLE_ENV, env);
            return false;
        }

        SQLLEN affected = 0;
        rc = SQLRowCount(insertStmt, &affected);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "SQLRowCount(insert row)");
        if (affected > 0) {
            ++rows_inserted;
        } else {
            ++rows_not_inserted;
        }

        rc = SQLFreeStmt(insertStmt, SQL_CLOSE);
        CHECK_ODBC_RC(rc, SQL_HANDLE_STMT, insertStmt, "SQLFreeStmt(insertStmt SQL_CLOSE)");
    }

    rc = SQLEndTran(SQL_HANDLE_DBC, dbc, SQL_COMMIT);
    CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLEndTran(COMMIT)");

    std::cout << "Trend-row insert complete.\n"
              << "Rows tested:       " << rows_tested << "\n"
              << "Rows inserted:     " << rows_inserted << "\n"
              << "Rows not inserted: " << rows_not_inserted << "\n";

    rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &verifyStmt);
    CHECK_ODBC_RC(rc, SQL_HANDLE_DBC, dbc, "SQLAllocHandle(verifyStmt)");
    rc = SQLExecDirect(
        verifyStmt,
        (SQLCHAR*)"SELECT TOP 10 ID, asset, seq_no, trade_date, close_price, savemov, state "
                  "FROM dbo.data_asset ORDER BY ID DESC",
        SQL_NTS);

    if (SQL_SUCCEEDED(rc)) {
        std::cout << "\nMost recent rows in dbo.data_asset:\n";
        std::cout << "ID  asset  seq_no  trade_date  close_price  savemov  state\n";

        SQLINTEGER id = 0;
        char v_asset[256] = {0};
        SQLINTEGER seqNo = 0;
        char tradeDate[64] = {0};
        double closePrice = 0.0;
        SQLINTEGER mov = 0;
        char state[32] = {0};

        while (SQLFetch(verifyStmt) == SQL_SUCCESS) {
            SQLGetData(verifyStmt, 1, SQL_C_SLONG, &id, 0, nullptr);
            SQLGetData(verifyStmt, 2, SQL_C_CHAR, v_asset, sizeof(v_asset), nullptr);
            SQLGetData(verifyStmt, 3, SQL_C_SLONG, &seqNo, 0, nullptr);
            SQLGetData(verifyStmt, 4, SQL_C_CHAR, tradeDate, sizeof(tradeDate), nullptr);
            SQLGetData(verifyStmt, 5, SQL_C_DOUBLE, &closePrice, 0, nullptr);
            SQLGetData(verifyStmt, 6, SQL_C_SLONG, &mov, 0, nullptr);
            SQLGetData(verifyStmt, 7, SQL_C_CHAR, state, sizeof(state), nullptr);

            std::cout << id << "  " << v_asset << "  " << seqNo << "  "
                      << tradeDate << "  " << std::fixed << std::setprecision(4)
                      << closePrice << "  " << mov << "  " << state << "\n";
        }
    } else {
        print_odbc_diagnostics(SQL_HANDLE_STMT, verifyStmt, "SQLExecDirect(verify select)");
    }

    if (verifyStmt != SQL_NULL_HSTMT) SQLFreeHandle(SQL_HANDLE_STMT, verifyStmt);
    if (existsStmt != SQL_NULL_HSTMT) SQLFreeHandle(SQL_HANDLE_STMT, existsStmt);
    if (insertStmt != SQL_NULL_HSTMT) SQLFreeHandle(SQL_HANDLE_STMT, insertStmt);
    SQLDisconnect(dbc);
    SQLFreeHandle(SQL_HANDLE_DBC, dbc);
    SQLFreeHandle(SQL_HANDLE_ENV, env);
    return true;
}


int main(int argc, char** argv)
{
    std::string filename = (argc > 1) ? argv[1] : "sp500.txt";
    std::string asset = basename_from_path(filename);

    std::vector<std::string> dates;
    std::vector<double> clos;
    int M = 0;

    if (!load_input_file(filename, dates, clos, M)) {
        return 1;
    }

    if (M <= 0) {
        std::cerr << "No rows loaded." << std::endl;
        return 1;
    }

    std::cout << "Asset/file: " << asset << std::endl;
    std::cout << "Input rows: " << M << std::endl;
    std::cout << "Data span: "
              << strip_quotes(dates[0])
              << " to "
              << strip_quotes(dates[M - 1])
              << std::endl;

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    // Correct order:
    // 1. First compute the real historical moving-average series.
    // 2. Do not alter clos[M - 1].
    // 3. Only later append a hypothetical next value for the flip-price test.
    Result gpu = run_gpu_actual(clos);
    Result cpu = run_cpu_actual(clos);

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

    bool exact = compare_results(cpu, gpu, M);
    std::cout << (exact ? "Exact CPU/GPU match." : "CPU/GPU mismatch detected.") << std::endl;

    std::cout << "GPU compute time: " << std::fixed << std::setprecision(3)
              << gpu.gpu_ms << " ms" << std::endl;
    std::cout << "GPU compute time: " << std::fixed << std::setprecision(6)
              << (gpu.gpu_ms / 1000.0) << " s" << std::endl;
    std::cout << "Estimated actual-pass FLOPs: " << std::fixed << std::setprecision(3)
              << gpu.estimated_total_flops << std::endl;
    std::cout << "Estimated actual-pass throughput: " << std::fixed << std::setprecision(9)
              << gpu.estimated_tflops << " TFLOP/s" << std::endl;
    std::cout << "Total wall time (GPU actual + CPU actual + compare): " << std::fixed
              << std::setprecision(3) << wall_ms << " ms" << std::endl;

    print_input_tail_report(dates, clos, gpu, cpu, 12);

    auto bars = build_bars(dates, clos, gpu.savemov);
    auto trend_results = analyze_trend_cpp(bars);

    print_trend_reports(trend_results);
    compute_correlations_cpp(trend_results);

    if (!trend_results.empty()) {
        const TrendRow& last = trend_results.back();
        FutureSwitch fs = find_next_switch_price(clos, last.idx, last.state, 0.1);
        print_future_switch_report(fs, clos.back());
    }


    if (!insert_trend_rows_to_sql_server(asset, dates, clos, gpu.savemov, trend_results)) {
        std::cerr << "\nFailed to write trend rows to SQL Server.\n";
        return 3;
    }

    return exact ? 0 : 2;
}

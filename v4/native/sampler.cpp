// Exact basket sampler of the Version-4 law given a taste z, the exact log Z_z, and the 2-D size dynamic
// program used when the size potential is on the number of categories present (M3).
// Plain C ABI, loaded with ctypes by v4/kernels.py (built with clang++ on first use).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int64_t MAX_THREADS = 4;

// ---- Row-scaled (n, m) tables ---------------------------------------------------------
// The 2-D size table holds the total weight of baskets with n products in m categories.
// Weights of different sizes n can differ by far more than a double can hold (about
// 2^+-1022): one product with utility 19 next to many ordinary ones makes size 120 about
// e^1000 heavier than size 1.  So every row n keeps its own power-of-two exponent E[n]
// (true value = stored * 2^E[n]); entries within one row share it.  A single tilt per trip
// (the old scheme) cannot do this and silently rounded whole sizes to zero.
constexpr int64_t ROW_EMPTY = std::numeric_limits<int64_t>::min() / 4;   // row is all zero

inline double pow2i(int64_t e) {             // 2^e for e <= 1023; 0 below the normal range
    if (e < -1022) return 0.0;
    if (e > 1023) return std::numeric_limits<double>::infinity();
    uint64_t bits = static_cast<uint64_t>(e + 1023) << 52;
    double v;
    std::memcpy(&v, &bits, sizeof v);
    return v;
}

// exp(lp) = qm * 2^qe with qm in [0.5, 1); qm = 0 for lp = -inf.
inline void split_log(double lp, double& qm, int64_t& qe) {
    if (!(lp > -std::numeric_limits<double>::infinity())) {
        qm = 0.0;
        qe = ROW_EMPTY;
        return;
    }
    if (std::fabs(lp) < 700.0) {
        int ex;
        qm = std::frexp(std::exp(lp), &ex);
        qe = ex;
        return;
    }
    double t = std::floor(lp / M_LN2) + 1.0;
    qe = static_cast<int64_t>(t);
    qm = std::exp(lp - t * M_LN2);
}

// Rescale rows 0..deg (entries m = 0..min(n, cat)) so each row's largest entry is in [0.5, 1).
inline void renorm_rows(double* F, int64_t* E, int64_t deg, int64_t cat, int64_t M1) {
    for (int64_t n = 0; n <= deg; ++n) {
        double* row = &F[n * M1];
        int64_t m_hi = std::min(n, cat);
        double mx = 0.0;
        for (int64_t m = 0; m <= m_hi; ++m) mx = std::max(mx, row[m]);
        if (!(mx > 0.0)) {
            E[n] = ROW_EMPTY;
            for (int64_t m = 0; m <= m_hi; ++m) row[m] = 0.0;
            continue;
        }
        int ex;
        std::frexp(mx, &ex);
        if (ex != 0) {
            double s = std::ldexp(1.0, -ex);
            for (int64_t m = 0; m <= m_hi; ++m) row[m] *= s;
            E[n] += ex;
        }
    }
}

// The exponent each row n' = 0..next_deg needs after multiplying by one category polynomial
// sum_k qm[k] 2^qe[k] t^k (k = 0..d): the largest exponent among the terms that reach it,
// so every scaled term is <= 1.
inline void target_exponents(const int64_t* E, int64_t* NB, int64_t deg_cur, int64_t next_deg,
                             const double* qm, const int64_t* qe, int64_t d, int64_t cap) {
    for (int64_t n = 0; n <= next_deg; ++n) NB[n] = ROW_EMPTY;
    for (int64_t n = 0; n <= deg_cur; ++n) {
        if (E[n] == ROW_EMPTY) continue;
        if (qm[0] > 0.0) NB[n] = std::max(NB[n], E[n] + qe[0]);
        int64_t k_max = std::min(d, cap - n);
        for (int64_t k = 1; k <= k_max; ++k) {
            if (qm[k] > 0.0) NB[n + k] = std::max(NB[n + k], E[n] + qe[k]);
        }
    }
}

// Multiply the table F (rows 0..deg_cur, categories 0..cat_cur, row stride M1, row exponents
// E) in place by one category polynomial sum_k qm[k] 2^qe[k] t^k y^[k>0], k = 0..d, keeping
// sizes n <= cap and categories <= cat_cap.  NB is scratch of length cap + 1.
inline void scaled_category_step(double* F, int64_t* E, int64_t* NB, int64_t& deg_cur, int64_t& cat_cur,
                                 const double* qm, const int64_t* qe, int64_t d, int64_t cap,
                                 int64_t cat_cap, int64_t M1) {
    const int64_t next_deg = std::min(cap, deg_cur + d);
    const int64_t next_cat = std::min(cat_cap, cat_cur + 1);
    target_exponents(E, NB, deg_cur, next_deg, qm, qe, d, cap);
    for (int64_t n = 0; n <= deg_cur; ++n) {
        int64_t m_prev = std::min(n, cat_cur);
        int64_t m_next = std::min(n, next_cat);
        if (m_next > m_prev) F[n * M1 + m_next] = 0.0;
    }
    for (int64_t n = deg_cur + 1; n <= next_deg; ++n) {
        std::memset(&F[n * M1], 0, (std::min(n, next_cat) + 1) * sizeof(double));
    }
    // Downward in n: row n is read (old values, old exponent) before anything is added to it.
    for (int64_t n = deg_cur; n >= 0; --n) {
        int64_t m_hi = std::min(n, cat_cur);
        double* row = &F[n * M1];
        if (E[n] == ROW_EMPTY) {
            for (int64_t m = 0; m <= m_hi; ++m) row[m] = 0.0;
            continue;
        }
        int64_t k_max = std::min(d, cap - n);
        for (int64_t k = 1; k <= k_max; ++k) {
            if (qm[k] == 0.0) continue;
            double f = qm[k] * pow2i(E[n] + qe[k] - NB[n + k]);
            if (f == 0.0) continue;
            double* dst = &F[(n + k) * M1 + 1];
            for (int64_t m = 0; m <= m_hi; ++m) dst[m] += row[m] * f;
        }
        double f0 = (qm[0] > 0.0) ? qm[0] * pow2i(E[n] + qe[0] - NB[n]) : 0.0;
        if (f0 != 1.0) {
            for (int64_t m = 0; m <= m_hi; ++m) row[m] *= f0;
        }
    }
    for (int64_t n = 0; n <= next_deg; ++n) E[n] = NB[n];
    renorm_rows(F, E, next_deg, next_cat, M1);
    deg_cur = next_deg;
    cat_cur = next_cat;
}

// Persistent thread pool (at most 4 threads total: calling thread + up to 3 workers)
// to avoid pthread_create / pthread_join overhead across sweeps and kernel calls.
class ThreadPool {
public:
    ThreadPool() : stop_(false), active_workers_(0), gen_(0) {
        for (int64_t t = 1; t < MAX_THREADS; ++t) {
            workers_.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            ++gen_;
        }
        cv_.notify_all();
        for (auto& w : workers_) {
            if (w.joinable()) w.join();
        }
    }

    inline void run(int64_t n_threads, const std::function<void(int64_t)>& fn) {
        if (n_threads <= 1) {
            fn(0);
            return;
        }
        if (n_threads > MAX_THREADS) n_threads = MAX_THREADS;
        {
            std::lock_guard<std::mutex> lk(mu_);
            fn_ = &fn;
            n_threads_ = n_threads;
            active_workers_.store(n_threads - 1, std::memory_order_release);
            ++gen_;
        }
        cv_.notify_all();
        fn(0);
        while (active_workers_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
    }

private:
    void worker_loop(int64_t tid) {
        uint64_t last_gen = 0;
        while (true) {
            const std::function<void(int64_t)>* fn_ptr = nullptr;
            bool should_run = false;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]() { return gen_ != last_gen || stop_; });
                if (stop_) return;
                last_gen = gen_;
                if (tid < n_threads_) {
                    fn_ptr = fn_;
                    should_run = true;
                }
            }
            if (should_run && fn_ptr) {
                (*fn_ptr)(tid);
                active_workers_.fetch_sub(1, std::memory_order_acq_rel);
            }
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::thread> workers_;
    const std::function<void(int64_t)>* fn_ = nullptr;
    int64_t n_threads_ = 1;
    bool stop_;
    std::atomic<int64_t> active_workers_;
    uint64_t gen_;
};

ThreadPool& pool() {
    static ThreadPool p;
    return p;
}

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    inline uint64_t next_u64() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    inline double uniform() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
    inline void normalpair(double& z0, double& z1) {
        double u1 = uniform();
        if (u1 < 1e-300) u1 = 1e-300;
        double u2 = uniform();
        double rad = std::sqrt(-2.0 * std::log(u1));
        double th = 6.28318530717958647692 * u2;
        z0 = rad * std::cos(th);
        z1 = rad * std::sin(th);
    }
};

inline double extra_count_pen(const double* f_extra, int64_t c, int64_t k) {
    if (!f_extra || k < 2) return 0.0;
    const double* fe = &f_extra[c * 3];
    if (k == 2) return fe[0];
    if (k == 3) return fe[1];
    return fe[2];
}

// 1-D category presence log Z_inf(theta) = log sum_{m=1..m_max} exp(-rho_0[m]) [y^m] prod_c (1 + y sum_{k=1..Kr} exp(G_c[k] + theta*k)).
// Normalizes each category factor by mc = max(0, lc) with conditional rescaling so B_buf never overflows or underflows even when one category has a huge outlier lc.
inline double compute_log_z_inf_theta(
    int64_t n_active,
    int64_t m_max,
    int64_t K1,
    const int64_t* act_fslot,
    const int64_t* act_Kr,
    const double* G,
    const double* rho_0,
    double theta,
    double* l_buf,
    double* B_buf
) {
    const double neg_inf = -1.0 / 0.0;
    if (n_active <= 0 || m_max <= 0) return neg_inf;
    bool any_finite = false;
    for (int64_t ia = 0; ia < n_active; ++ia) {
        const double* g_ptr = &G[act_fslot[ia] * K1];
        int64_t Kr = act_Kr[ia];
        double mx = neg_inf;
        for (int64_t k = 1; k <= Kr; ++k) {
            double v = g_ptr[k] + theta * static_cast<double>(k);
            if (v > mx) mx = v;
        }
        if (!(mx > neg_inf)) {
            l_buf[ia] = neg_inf;
            continue;
        }
        double s = 0.0;
        for (int64_t k = 1; k <= Kr; ++k) {
            double v = g_ptr[k] + theta * static_cast<double>(k);
            if (v > neg_inf) s += std::exp(v - mx);
        }
        l_buf[ia] = mx + std::log(s);
        any_finite = true;
    }
    if (!any_finite) return neg_inf;

    B_buf[0] = 1.0;
    std::fill(B_buf + 1, B_buf + m_max + 1, 0.0);
    int64_t m_cur = 0;
    double log_scale = 0.0;
    double accum_mc = 0.0;
    for (int64_t ia = 0; ia < n_active; ++ia) {
        double lc = l_buf[ia];
        if (!(lc > neg_inf)) continue;
        double mc = (lc > 0.0) ? lc : 0.0;
        double q0 = (mc > 0.0) ? std::exp(-mc) : 1.0;
        double q1 = (mc > 0.0) ? 1.0 : std::exp(lc);
        if (q1 == 0.0 && q0 == 1.0) continue;
        m_cur = std::min<int64_t>(m_cur + 1, m_max);
        if (q0 == 1.0) {
            for (int64_t m = m_cur; m >= 1; --m) {
                B_buf[m] += B_buf[m - 1] * q1;
            }
        } else {
            for (int64_t m = m_cur; m >= 1; --m) {
                B_buf[m] = B_buf[m] * q0 + B_buf[m - 1] * q1;
            }
            B_buf[0] *= q0;
        }
        log_scale += mc;
        accum_mc += mc;
        if (accum_mc >= 60.0 || q0 == 0.0) {
            double mx = 0.0;
            for (int64_t m = 0; m <= m_cur; ++m) {
                if (B_buf[m] > mx) mx = B_buf[m];
            }
            if (mx > 0.0) {
                double inv = 1.0 / mx;
                for (int64_t m = 0; m <= m_cur; ++m) B_buf[m] *= inv;
                log_scale += std::log(mx);
            }
            accum_mc = 0.0;
        }
    }
    double max_lp = neg_inf;
    for (int64_t m = 1; m <= m_cur; ++m) {
        double bval = B_buf[m];
        if (bval > 0.0) {
            double lp = std::log(bval) + log_scale - rho_0[m];
            l_buf[m - 1] = lp;
            if (lp > max_lp) max_lp = lp;
        } else {
            l_buf[m - 1] = neg_inf;
        }
    }
    if (!(max_lp > neg_inf)) return neg_inf;
    double sum_exp = 0.0;
    for (int64_t m = 1; m <= m_cur; ++m) {
        if (l_buf[m - 1] > neg_inf) sum_exp += std::exp(l_buf[m - 1] - max_lp);
    }
    return max_lp + std::log(sum_exp);
}

struct TripCapBound {
    int64_t K_cap;
    int64_t D_tot;
    double lz_inf0;
    double lz_inf_t1;
    double lz_inf_t2;
    bool used_trunc;

    // Proven upper bound on log Z(nmax) - log Z(K_cap):
    //   0 <= log Z(nmax) - log Z(K_cap) <= min(lz_inf0 - lz_K, exp(lz_inf_t1 - t1*(K+1) - lz_K), exp(lz_inf_t2 - t2*(K+1) - lz_K))
    inline bool certified(double lz_K, double tol = 1e-8) const {
        if (!used_trunc || K_cap >= D_tot) return true;
        if (!std::isfinite(lz_K) || !std::isfinite(lz_inf0)) return false;
        if (lz_K > lz_inf0 + 1e-9) return false;
        double err_sand = (lz_inf0 >= lz_K) ? (lz_inf0 - lz_K) : 0.0;
        if (err_sand <= tol) return true;
        double kp1 = static_cast<double>(K_cap + 1);
        if (std::isfinite(lz_inf_t1) && lz_inf_t1 >= lz_inf0 - 1e-9) {
            if (std::exp(lz_inf_t1 - 0.4 * kp1 - lz_K) <= tol) return true;
        }
        if (std::isfinite(lz_inf_t2) && lz_inf_t2 >= lz_inf0 - 1e-9) {
            if (std::exp(lz_inf_t2 - 0.8 * kp1 - lz_K) <= tol) return true;
        }
        return false;
    }
};

inline TripCapBound choose_trip_cap_m3(
    int64_t n_active,
    int64_t nmax,
    int64_t min_cap,
    int64_t K1,
    const int64_t* act_fslot,
    const int64_t* act_Kr,
    const double* G,
    const double* rho_0,
    double* l_buf,
    double* B_buf
) {
    int64_t sum_Kr = 0;
    for (int64_t ia = 0; ia < n_active; ++ia) sum_Kr += act_Kr[ia];
    int64_t D_tot = std::min(nmax, sum_Kr);
    TripCapBound res{D_tot, D_tot, -1.0 / 0.0, -1.0 / 0.0, -1.0 / 0.0, false};
    if (D_tot <= min_cap || n_active <= 0) {
        return res;
    }
    int64_t m_max = std::min(n_active, nmax);
    res.lz_inf0 = compute_log_z_inf_theta(n_active, m_max, K1, act_fslot, act_Kr, G, rho_0, 0.0, l_buf, B_buf);
    res.lz_inf_t1 = compute_log_z_inf_theta(n_active, m_max, K1, act_fslot, act_Kr, G, rho_0, 0.4, l_buf, B_buf);
    res.lz_inf_t2 = compute_log_z_inf_theta(n_active, m_max, K1, act_fslot, act_Kr, G, rho_0, 0.8, l_buf, B_buf);
    if (!std::isfinite(res.lz_inf0)) return res;

    // Target 5e-9 (< 1e-8) in Chernoff prediction: -log(5e-9) = 19.11382792451231
    constexpr double neg_log_eps = 19.11382792451231;
    int64_t k_best = D_tot;
    if (std::isfinite(res.lz_inf_t1) && res.lz_inf_t1 >= res.lz_inf0) {
        double kf1 = std::ceil((res.lz_inf_t1 - res.lz_inf0 + neg_log_eps) / 0.4);
        if (kf1 >= 1.0 && kf1 < static_cast<double>(k_best)) k_best = static_cast<int64_t>(kf1);
    }
    if (std::isfinite(res.lz_inf_t2) && res.lz_inf_t2 >= res.lz_inf0) {
        double kf2 = std::ceil((res.lz_inf_t2 - res.lz_inf0 + neg_log_eps) / 0.8);
        if (kf2 >= 1.0 && kf2 < static_cast<double>(k_best)) k_best = static_cast<int64_t>(kf2);
    }
    k_best = std::clamp<int64_t>(k_best, std::max<int64_t>(1, min_cap), D_tot);
    if (k_best < D_tot) {
        res.K_cap = k_best;
        res.used_trunc = true;
    }
    return res;
}

struct TripSamplerScratch {
    std::vector<double> Fs2d;
    std::vector<double> gl;
    std::vector<double> lp2d;
    std::vector<int64_t> deg_hist;
    std::vector<int64_t> cat_hist;
    std::vector<int64_t> act_r;
    std::vector<int64_t> act_fslot;
    std::vector<int64_t> act_Kr;
    std::vector<int64_t> act_Kr_eff;
    std::vector<double> l_buf;
    std::vector<double> B_buf;
    std::vector<int64_t> row_exp;   // row-scaled log Z table: exponent per size n
    std::vector<int64_t> row_exp_next;
    std::vector<int64_t> q_exp;
};

int sample_trip_categories_2d_impl(
    int64_t b,
    int64_t Cpad,
    int64_t K,
    int64_t K1,
    int64_t nmax,
    int64_t min_cap,
    const int64_t* slot_row,
    const int64_t* row_size,
    const double* G,
    const double* rho_0,
    SplitMix64& rng,
    TripSamplerScratch& sc,
    std::vector<int64_t>& need_row
) {
    const double neg_inf = -1.0 / 0.0;
    if ((int64_t)sc.act_r.size() < Cpad) {
        sc.act_r.resize(Cpad);
        sc.act_fslot.resize(Cpad);
        sc.act_Kr.resize(Cpad);
        sc.act_Kr_eff.resize(Cpad);
        sc.deg_hist.resize(Cpad + 1);
        sc.cat_hist.resize(Cpad + 1);
        sc.l_buf.resize(Cpad + 1);
        sc.B_buf.resize(Cpad + 2);
    }
    int64_t n_active = 0;
    for (int64_t c = 0; c < Cpad; ++c) {
        int64_t fslot = b * Cpad + c;
        int64_t r = slot_row[fslot];
        if (r < 0) continue;
        int64_t Kr = std::min({K, row_size[r], nmax});
        if (Kr <= 0) continue;
        sc.act_r[n_active] = r;
        sc.act_fslot[n_active] = fslot;
        sc.act_Kr[n_active] = Kr;
        ++n_active;
    }
    if (n_active == 0) return -1;

    TripCapBound cap_info = choose_trip_cap_m3(
        n_active, nmax, min_cap, K1, sc.act_fslot.data(), sc.act_Kr.data(),
        G, rho_0, sc.l_buf.data(), sc.B_buf.data());

    if ((int64_t)sc.gl.size() < (n_active + 1) * K1) {
        sc.gl.resize((n_active + 1) * K1, 0.0);
    }

    int64_t cur_cap = cap_info.K_cap;
    int64_t M1 = 0;
    int64_t stride2d = 0;
    int64_t deg_so_far = 0;
    int64_t cat_so_far = 0;
    double log_scale = 0.0;
    double max_lp = neg_inf;
    double sum_p = 0.0;

    for (int attempt = 0; attempt < 2; ++attempt) {
        int64_t K_use = cur_cap;
        M1 = std::min(n_active, K_use) + 1;
        stride2d = (K_use + 1) * M1;
        if ((int64_t)sc.Fs2d.size() < (n_active + 1) * stride2d) {
            sc.Fs2d.resize((n_active + 1) * stride2d, 0.0);
        }
        if ((int64_t)sc.lp2d.size() < stride2d) {
            sc.lp2d.resize(stride2d, neg_inf);
        }
        sc.Fs2d[0] = 1.0;
        deg_so_far = 0;
        cat_so_far = 0;
        log_scale = 0.0;
        double accum_mg = 0.0;
        double bound_prod = 1.0;
        bool underflowed = false;

        for (int64_t ia = 0; ia < n_active; ++ia) {
            sc.deg_hist[ia] = deg_so_far;
            sc.cat_hist[ia] = cat_so_far;
            const double* f_prev = &sc.Fs2d[ia * stride2d];
            double* f_cur = &sc.Fs2d[(ia + 1) * stride2d];
            double* q_c = &sc.gl[ia * K1];
            int64_t fslot = sc.act_fslot[ia];
            int64_t Kr_max = std::min(sc.act_Kr[ia], K_use);
            const double* g_ptr = &G[fslot * K1];

            double mg = 0.0;
            for (int64_t k = 1; k <= Kr_max; ++k) {
                if (g_ptr[k] > mg) mg = g_ptr[k];
            }
            q_c[0] = (mg == 0.0) ? 1.0 : std::exp(-mg);
            int64_t Kr = 0;
            double q_sum = q_c[0];
            for (int64_t k = 1; k <= Kr_max; ++k) {
                double v = (g_ptr[k] > neg_inf) ? std::exp(g_ptr[k] - mg) : 0.0;
                q_c[k] = v;
                q_sum += v;
                if (v > 0.0) Kr = k;
            }
            sc.act_Kr_eff[ia] = Kr;
            log_scale += mg;

            if (Kr <= 0) {
                // Since mg >= 0 and Kr == 0, mg must be 0.0 and q_c[0] == 1.0.
                for (int64_t n = 0; n <= deg_so_far; ++n) {
                    int64_t m_hi = std::min(n, cat_so_far);
                    std::memcpy(&f_cur[n * M1], &f_prev[n * M1], (m_hi + 1) * sizeof(double));
                }
                continue;
            }
            int64_t next_deg = std::min(K_use, deg_so_far + Kr);
            int64_t next_cat = std::min(K_use, cat_so_far + 1);
            double g0 = q_c[0];

            if (g0 == 1.0) {
                for (int64_t n = 0; n <= deg_so_far; ++n) {
                    int64_t m_prev = std::min(n, cat_so_far);
                    int64_t m_next = std::min(n, next_cat);
                    std::memcpy(&f_cur[n * M1], &f_prev[n * M1], (m_prev + 1) * sizeof(double));
                    if (m_next > m_prev) {
                        f_cur[n * M1 + m_next] = 0.0;
                    }
                }
                for (int64_t n = deg_so_far + 1; n <= next_deg; ++n) {
                    int64_t m_next = std::min(n, next_cat);
                    std::memset(&f_cur[n * M1], 0, (m_next + 1) * sizeof(double));
                }
            } else if (g0 > 0.0) {
                for (int64_t n = 0; n <= deg_so_far; ++n) {
                    int64_t m_prev = std::min(n, cat_so_far);
                    int64_t m_next = std::min(n, next_cat);
                    for (int64_t m = 0; m <= m_prev; ++m) {
                        f_cur[n * M1 + m] = f_prev[n * M1 + m] * g0;
                    }
                    if (m_next > m_prev) {
                        f_cur[n * M1 + m_next] = 0.0;
                    }
                }
                for (int64_t n = deg_so_far + 1; n <= next_deg; ++n) {
                    int64_t m_next = std::min(n, next_cat);
                    std::memset(&f_cur[n * M1], 0, (m_next + 1) * sizeof(double));
                }
            } else {
                for (int64_t n = 0; n <= next_deg; ++n) {
                    int64_t m_next = std::min(n, next_cat);
                    std::memset(&f_cur[n * M1], 0, (m_next + 1) * sizeof(double));
                }
            }

            for (int64_t k = 1; k <= Kr; ++k) {
                double gk = q_c[k];
                if (gk == 0.0) continue;
                int64_t max_n = std::min(K_use - k, deg_so_far);
                for (int64_t n = 0; n <= max_n; ++n) {
                    int64_t m_hi = std::min(n, cat_so_far);
                    const double* fp_row = &f_prev[n * M1];
                    double* fc_row = &f_cur[(n + k) * M1 + 1];
                    for (int64_t m = 0; m <= m_hi; ++m) {
                        fc_row[m] += fp_row[m] * gk;
                    }
                }
            }
            accum_mg += mg;
            bound_prod *= q_sum;
            if (accum_mg >= 60.0 || bound_prod >= 1e60 || g0 == 0.0) {
                double max_cv = 0.0;
                for (int64_t n = 0; n <= next_deg; ++n) {
                    int64_t m_hi = std::min(n, next_cat);
                    for (int64_t m = 0; m <= m_hi; ++m) {
                        double cv = f_cur[n * M1 + m];
                        if (cv > max_cv) max_cv = cv;
                    }
                }
                if (max_cv <= 0.0) {
                    underflowed = true;
                    break;
                }
                double inv_max = 1.0 / max_cv;
                for (int64_t n = 0; n <= next_deg; ++n) {
                    int64_t m_hi = std::min(n, next_cat);
                    for (int64_t m = 0; m <= m_hi; ++m) {
                        f_cur[n * M1 + m] *= inv_max;
                    }
                }
                log_scale += std::log(max_cv);
                accum_mg = 0.0;
                bound_prod = 1.0;
            }
            deg_so_far = next_deg;
            cat_so_far = next_cat;
        }

        if (underflowed) {
            if (cur_cap < cap_info.D_tot) {
                cur_cap = cap_info.D_tot;
                cap_info.used_trunc = false;
                continue;
            }
            return -1;
        }

        const double* f_last = &sc.Fs2d[n_active * stride2d];
        max_lp = neg_inf;
        for (int64_t n = 1; n <= deg_so_far; ++n) {
            int64_t m_hi = std::min(n, cat_so_far);
            for (int64_t m = 1; m <= m_hi; ++m) {
                double v = f_last[n * M1 + m];
                double val = (v > 0.0) ? (std::log(v) + log_scale - rho_0[m]) : neg_inf;
                sc.lp2d[n * M1 + m] = val;
                if (val > max_lp) max_lp = val;
            }
        }
        if (!(max_lp > neg_inf)) {
            if (cur_cap < cap_info.D_tot) {
                cur_cap = cap_info.D_tot;
                cap_info.used_trunc = false;
                continue;
            }
            return -1;
        }
        sum_p = 0.0;
        for (int64_t n = 1; n <= deg_so_far; ++n) {
            int64_t m_hi = std::min(n, cat_so_far);
            for (int64_t m = 1; m <= m_hi; ++m) {
                double val = sc.lp2d[n * M1 + m];
                double p = (val > neg_inf) ? std::exp(val - max_lp) : 0.0;
                sc.lp2d[n * M1 + m] = p;
                sum_p += p;
            }
        }
        double lz_K = max_lp + std::log(sum_p);
        if (cap_info.certified(lz_K)) {
            break;
        }
        cur_cap = cap_info.D_tot;
        cap_info.used_trunc = false;
    }

    double target = rng.uniform() * sum_p;
    double acc = 0.0;
    int64_t rem_n = 1, rem_m = 1;
    bool found = false;
    for (int64_t n = 1; n <= deg_so_far && !found; ++n) {
        int64_t m_hi = std::min(n, cat_so_far);
        for (int64_t m = 1; m <= m_hi; ++m) {
            acc += sc.lp2d[n * M1 + m];
            if (target <= acc) {
                rem_n = n;
                rem_m = m;
                found = true;
                break;
            }
        }
    }

    double* w_k = &sc.gl[n_active * K1];
    for (int64_t ia = n_active - 1; ia >= 0; --ia) {
        if (rem_n == 0) break;
        int64_t r = sc.act_r[ia];
        int64_t Kr = std::min(sc.act_Kr[ia], rem_n);
        const double* f_prev = &sc.Fs2d[ia * stride2d];
        const double* q_c = &sc.gl[ia * K1];
        int64_t d_prev = sc.deg_hist[ia];
        int64_t c_prev = sc.cat_hist[ia];
        double sum_c = 0.0;
        for (int64_t k = 0; k <= Kr; ++k) {
            double w = 0.0;
            if (k == 0) {
                if (rem_n <= d_prev && rem_m <= std::min(rem_n, c_prev)) {
                    w = f_prev[rem_n * M1 + rem_m] * q_c[0];
                }
            } else if (rem_m >= 1) {
                int64_t rn = rem_n - k;
                int64_t rm = rem_m - 1;
                if (rn <= d_prev && rm <= std::min(rn, c_prev)) {
                    w = f_prev[rn * M1 + rm] * q_c[k];
                }
            }
            w_k[k] = w;
            sum_c += w;
        }
        if (sum_c <= 0.0) {
            // Log-space fallback in case f_prev * q_c underflowed double
            const double* g_ptr = &G[sc.act_fslot[ia] * K1];
            double max_lw = neg_inf;
            for (int64_t k = 0; k <= Kr; ++k) {
                double lw = neg_inf;
                if (k == 0) {
                    if (rem_n <= d_prev && rem_m <= std::min(rem_n, c_prev)) {
                        double fp = f_prev[rem_n * M1 + rem_m];
                        if (fp > 0.0) lw = std::log(fp);
                    }
                } else if (rem_m >= 1) {
                    int64_t rn = rem_n - k;
                    int64_t rm = rem_m - 1;
                    if (rn <= d_prev && rm <= std::min(rn, c_prev)) {
                        double fp = f_prev[rn * M1 + rm];
                        if (fp > 0.0 && g_ptr[k] > neg_inf) lw = std::log(fp) + g_ptr[k];
                    }
                }
                w_k[k] = lw;
                if (lw > max_lw) max_lw = lw;
            }
            sum_c = 0.0;
            for (int64_t k = 0; k <= Kr; ++k) {
                w_k[k] = (w_k[k] > neg_inf && max_lw > neg_inf) ? std::exp(w_k[k] - max_lw) : 0.0;
                sum_c += w_k[k];
            }
        }
        double targ_c = rng.uniform() * sum_c;
        double acc_c = 0.0;
        int64_t chosen_k = Kr;
        for (int64_t k = 0; k <= Kr; ++k) {
            acc_c += w_k[k];
            if (targ_c <= acc_c) {
                chosen_k = k;
                break;
            }
        }
        need_row[r] = chosen_k;
        rem_n -= chosen_k;
        if (chosen_k > 0) rem_m -= 1;
    }
    return (rem_n == 0 && rem_m == 0) ? 0 : -1;
}

// Compute exact log Z for a single trip b given its row polynomials G (with G[fslot*K1 + 0] = 0)
// and per-trip max slot utility trip_scale; both branches keep an exponent per basket size.
double trip_log_partition_from_G(
    int64_t b,
    int64_t Cpad,
    int64_t K,
    int64_t K1,
    int64_t nmax,
    int64_t cat_rho0,
    int64_t min_cap,
    double trip_scale,
    const int64_t* slot_row,
    const int64_t* row_size,
    const double* G,
    const double* rho_0,
    TripSamplerScratch& sc
) {
    const double neg_inf = -1.0 / 0.0;
    if ((int64_t)sc.act_r.size() < Cpad) {
        sc.act_r.resize(Cpad);
        sc.act_fslot.resize(Cpad);
        sc.act_Kr.resize(Cpad);
        sc.act_Kr_eff.resize(Cpad);
        sc.l_buf.resize(Cpad + 1);
        sc.B_buf.resize(Cpad + 2);
    }
    int64_t n_active = 0;
    int64_t sum_Kr = 0;
    for (int64_t c = 0; c < Cpad; ++c) {
        int64_t fslot = b * Cpad + c;
        int64_t r = slot_row[fslot];
        if (r < 0) continue;
        int64_t Kr = std::min({K, row_size[r], nmax});
        if (Kr <= 0) continue;
        sc.act_r[n_active] = r;
        sc.act_fslot[n_active] = fslot;
        sc.act_Kr[n_active] = Kr;
        sum_Kr += Kr;
        ++n_active;
    }
    if (n_active == 0) return neg_inf;

    if ((int64_t)sc.gl.size() < K1) {
        sc.gl.resize(K1, 0.0);
    }

    if (!cat_rho0) {
        // 1-D product over sizes n, each coefficient with its own power-of-two exponent.
        int64_t D_tot = std::min(nmax, sum_Kr);
        if ((int64_t)sc.Fs2d.size() < D_tot + 1) sc.Fs2d.resize(D_tot + 1);
        if ((int64_t)sc.row_exp.size() < D_tot + 1) {
            sc.row_exp.resize(D_tot + 1);
            sc.row_exp_next.resize(D_tot + 1);
        }
        if ((int64_t)sc.q_exp.size() < K1) sc.q_exp.resize(K1);
        double* F1 = sc.Fs2d.data();
        int64_t* E = sc.row_exp.data();
        int64_t* NB = sc.row_exp_next.data();
        F1[0] = 1.0;
        E[0] = 0;
        std::fill(F1 + 1, F1 + D_tot + 1, 0.0);
        std::fill(E + 1, E + D_tot + 1, ROW_EMPTY);
        int64_t deg_cur = 0;
        double* q_c = sc.gl.data();
        int64_t* q_e = sc.q_exp.data();
        for (int64_t ia = 0; ia < n_active; ++ia) {
            int64_t fslot = sc.act_fslot[ia];
            int64_t Kr_max = std::min(sc.act_Kr[ia], D_tot);
            const double* g_ptr = &G[fslot * K1];
            int64_t d = 0;
            for (int64_t k = 1; k <= Kr_max; ++k) {
                split_log(g_ptr[k] - static_cast<double>(k) * trip_scale, q_c[k], q_e[k]);
                if (q_c[k] > 0.0) d = k;
            }
            if (d <= 0) continue;
            int64_t next_deg = std::min(D_tot, deg_cur + d);
            // target exponent per n (the absent term has weight exactly 1 = 1 * 2^0)
            for (int64_t n = 0; n <= next_deg; ++n) NB[n] = (n <= deg_cur) ? E[n] : ROW_EMPTY;
            for (int64_t n = 0; n <= deg_cur; ++n) {
                if (E[n] == ROW_EMPTY) continue;
                int64_t k_max = std::min(d, D_tot - n);
                for (int64_t k = 1; k <= k_max; ++k) {
                    if (q_c[k] > 0.0) NB[n + k] = std::max(NB[n + k], E[n] + q_e[k]);
                }
            }
            for (int64_t n = deg_cur + 1; n <= next_deg; ++n) F1[n] = 0.0;
            // Downward in n: F1[n] still holds its old value (old exponent) when it is read; it
            // is moved to its new exponent after it has been added to the larger sizes.
            for (int64_t n = deg_cur; n >= 0; --n) {
                if (E[n] == ROW_EMPTY) continue;
                double fn = F1[n];
                int64_t k_max = std::min(d, D_tot - n);
                for (int64_t k = 1; k <= k_max; ++k) {
                    if (q_c[k] == 0.0) continue;
                    F1[n + k] += fn * q_c[k] * pow2i(E[n] + q_e[k] - NB[n + k]);
                }
                F1[n] = fn * pow2i(E[n] - NB[n]);
            }
            for (int64_t n = 0; n <= next_deg; ++n) {
                if (NB[n] == ROW_EMPTY || !(F1[n] > 0.0)) {
                    F1[n] = 0.0;
                    E[n] = ROW_EMPTY;
                    continue;
                }
                int ex;
                F1[n] = std::frexp(F1[n], &ex);
                E[n] = NB[n] + ex;
            }
            deg_cur = next_deg;
        }
        double max_lp = neg_inf;
        if ((int64_t)sc.lp2d.size() < deg_cur + 1) sc.lp2d.resize(deg_cur + 1);
        for (int64_t n = 1; n <= deg_cur; ++n) {
            double v = F1[n];
            double lp = (v > 0.0 && E[n] != ROW_EMPTY)
                            ? (std::log(v) + static_cast<double>(E[n]) * M_LN2 + static_cast<double>(n) * trip_scale - rho_0[n])
                            : neg_inf;
            sc.lp2d[n] = lp;
            if (lp > max_lp) max_lp = lp;
        }
        if (!(max_lp > neg_inf)) return neg_inf;
        double sum_exp = 0.0;
        for (int64_t n = 1; n <= deg_cur; ++n) {
            if (sc.lp2d[n] > neg_inf) sum_exp += std::exp(sc.lp2d[n] - max_lp);
        }
        return max_lp + std::log(sum_exp);
    }

    TripCapBound cap_info = choose_trip_cap_m3(
        n_active, nmax, min_cap, K1, sc.act_fslot.data(), sc.act_Kr.data(),
        G, rho_0, sc.l_buf.data(), sc.B_buf.data());

    int64_t cur_cap = cap_info.K_cap;
    double* q_c = sc.gl.data();
    for (int attempt = 0; attempt < 2; ++attempt) {
        int64_t K_use = cur_cap;
        int64_t M1 = std::min(n_active, K_use) + 1;
        int64_t stride2d = (K_use + 1) * M1;
        if ((int64_t)sc.Fs2d.size() < stride2d) sc.Fs2d.resize(stride2d, 0.0);
        if ((int64_t)sc.lp2d.size() < stride2d) sc.lp2d.resize(stride2d, neg_inf);
        double* F = sc.Fs2d.data();
        F[0] = 1.0;
        if ((int64_t)sc.row_exp.size() < K_use + 1) {
            sc.row_exp.resize(K_use + 1);
            sc.row_exp_next.resize(K_use + 1);
        }
        if ((int64_t)sc.q_exp.size() < K1) sc.q_exp.resize(K1);
        int64_t* E = sc.row_exp.data();
        E[0] = 0;
        int64_t deg_cur = 0;
        int64_t cat_cur = 0;

        for (int64_t ia = 0; ia < n_active; ++ia) {
            int64_t fslot = sc.act_fslot[ia];
            int64_t Kr_max = std::min(sc.act_Kr[ia], K_use);
            const double* g_ptr = &G[fslot * K1];
            int64_t* q_e = sc.q_exp.data();
            q_c[0] = 1.0;                      // G[.., 0] = 0: the absent term is exactly 1 = 1 * 2^0
            q_e[0] = 0;
            int64_t d = 0;
            for (int64_t k = 1; k <= Kr_max; ++k) {
                split_log(g_ptr[k] - static_cast<double>(k) * trip_scale, q_c[k], q_e[k]);
                if (q_c[k] > 0.0) d = k;
            }
            if (d <= 0) continue;
            scaled_category_step(F, E, sc.row_exp_next.data(), deg_cur, cat_cur, q_c, q_e, d,
                                 K_use, K_use, M1);
        }

        double max_lp = neg_inf;
        for (int64_t n = 1; n <= deg_cur; ++n) {
            int64_t m_hi = (E[n] == ROW_EMPTY) ? 0 : std::min(n, cat_cur);
            double n_shift = static_cast<double>(E[n]) * M_LN2 + static_cast<double>(n) * trip_scale;
            for (int64_t m = 1; m <= m_hi; ++m) {
                double v = F[n * M1 + m];
                double val = (v > 0.0) ? (std::log(v) + n_shift - rho_0[m]) : neg_inf;
                sc.lp2d[n * M1 + m] = val;
                if (val > max_lp) max_lp = val;
            }
        }
        if (!(max_lp > neg_inf)) {
            if (cur_cap < cap_info.D_tot) {
                cur_cap = cap_info.D_tot;
                cap_info.used_trunc = false;
                continue;
            }
            return neg_inf;
        }
        double sum_exp = 0.0;
        for (int64_t n = 1; n <= deg_cur; ++n) {
            int64_t m_hi = std::min(n, cat_cur);
            for (int64_t m = 1; m <= m_hi; ++m) {
                double val = sc.lp2d[n * M1 + m];
                if (val > neg_inf) sum_exp += std::exp(val - max_lp);
            }
        }
        double lz_K = max_lp + std::log(sum_exp);
        if (cap_info.certified(lz_K)) {
            return lz_K;
        }
        cur_cap = cap_info.D_tot;
        cap_info.used_trunc = false;
    }
    return neg_inf;
}

}  // namespace

// log coefficients over (n, m) of the product of category polynomials, with a binary
// exponent per row n (see "Row-scaled (n, m) tables").  With states != null (the backward
// pass), every intermediate table is kept in states [B, Cpad+1, N1, N1] with its row exponents
// in state_exp [B, Cpad+1, N1], and each category's mantissas / exponents in q_all / q_exp.
extern "C" int log_degree_2d_forward_cpp(
    int64_t B,
    int64_t Cpad,
    int64_t stride,
    int64_t nmax,
    const double* log_coeffs,
    const int64_t* degrees,
    double* log_out,
    double* q_all,
    double* states,
    double* q_exp,
    double* state_exp
) {
    const double neg_inf = -1.0 / 0.0;
    const int64_t N1 = nmax + 1;
    const int64_t stride2d = N1 * N1;
    std::atomic<int> err{0};
    int64_t n_threads = std::min<int64_t>(B, MAX_THREADS);
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](int64_t tid) {
        int64_t b_start = (B * tid) / n_threads;
        int64_t b_end = (B * (tid + 1)) / n_threads;
        std::vector<double> buf(states ? 0 : stride2d, 0.0);
        std::vector<int64_t> E(N1), NB(N1), qe(stride);
        std::vector<double> qm(stride);

        for (int64_t b = b_start; b < b_end; ++b) {
            double* first = states ? (states + b * (Cpad + 1) * stride2d) : nullptr;
            double* first_exp = states ? (state_exp + b * (Cpad + 1) * N1) : nullptr;
            std::fill(E.begin(), E.end(), ROW_EMPTY);
            E[0] = 0;
            if (first) {
                for (int64_t n = 0; n <= nmax; ++n) {
                    first[n * N1 + 0] = 0.0;
                    if (n >= 1) first[n * N1 + 1] = 0.0;
                    first_exp[n] = static_cast<double>(ROW_EMPTY);
                }
                first[0] = 1.0;
                first_exp[0] = 0.0;
            } else {
                buf[0] = 1.0;
            }
            int64_t deg_cur = 0;
            int64_t cat_cur = 0;

            for (int64_t c = 0; c < Cpad; ++c) {
                const int64_t d = std::clamp<int64_t>(degrees[b * Cpad + c], 0, nmax);
                const double* lp = log_coeffs + (b * Cpad + c) * stride;
                bool any = false;
                int64_t d_eff = 0;
                for (int64_t k = 0; k <= d; ++k) {
                    split_log(lp[k], qm[k], qe[k]);
                    if (qm[k] > 0.0) {
                        any = true;
                        if (k >= 1) d_eff = k;
                    }
                }
                if (!any) {
                    err.store(-1, std::memory_order_relaxed);
                    return;
                }
                if (q_all) {
                    double* q = q_all + (b * Cpad + c) * stride;
                    double* qx = q_exp + (b * Cpad + c) * stride;
                    for (int64_t k = 0; k <= d; ++k) {
                        q[k] = qm[k];
                        qx[k] = static_cast<double>(qe[k]);
                    }
                }

                if (!first) {
                    if (d_eff == 0) {                  // only the absent term: rescale rows
                        for (int64_t n = 0; n <= deg_cur; ++n) {
                            if (E[n] == ROW_EMPTY) continue;
                            if (qm[0] == 0.0) {
                                E[n] = ROW_EMPTY;
                                continue;
                            }
                            E[n] += qe[0];
                            int64_t m_hi = std::min(n, cat_cur);
                            for (int64_t m = 0; m <= m_hi; ++m) buf[n * N1 + m] *= qm[0];
                        }
                        renorm_rows(buf.data(), E.data(), deg_cur, cat_cur, N1);
                        continue;
                    }
                    scaled_category_step(buf.data(), E.data(), NB.data(), deg_cur, cat_cur,
                                         qm.data(), qe.data(), d_eff, nmax, nmax, N1);
                } else {
                    const double* previous = first + c * stride2d;
                    double* next = first + (c + 1) * stride2d;
                    double* next_exp = first_exp + (c + 1) * N1;
                    int64_t next_deg = std::min(nmax, deg_cur + d);
                    int64_t next_cat = std::min(nmax, cat_cur + (d > 0 ? 1 : 0));
                    int64_t m_clear = std::min(nmax, c + 2);
                    for (int64_t n = 0; n <= nmax; ++n) {
                        int64_t m_hi = std::min(n, m_clear);
                        std::fill(&next[n * N1], &next[n * N1 + m_hi + 1], 0.0);
                    }
                    target_exponents(E.data(), NB.data(), deg_cur, next_deg, qm.data(), qe.data(), d_eff, nmax);
                    if (qm[0] > 0.0) {
                        for (int64_t n = 0; n <= deg_cur; ++n) {
                            if (E[n] == ROW_EMPTY) continue;
                            double f0 = qm[0] * pow2i(E[n] + qe[0] - NB[n]);
                            int64_t m_hi = std::min(n, cat_cur);
                            for (int64_t m = 0; m <= m_hi; ++m) {
                                next[n * N1 + m] = previous[n * N1 + m] * f0;
                            }
                        }
                    }
                    for (int64_t k = 1; k <= d_eff; ++k) {
                        if (qm[k] == 0.0) continue;
                        int64_t max_n = std::min(nmax - k, deg_cur);
                        for (int64_t n = 0; n <= max_n; ++n) {
                            if (E[n] == ROW_EMPTY) continue;
                            double f = qm[k] * pow2i(E[n] + qe[k] - NB[n + k]);
                            if (f == 0.0) continue;
                            int64_t m_hi = std::min(n, cat_cur);
                            const double* prev_row = &previous[n * N1];
                            double* next_row = &next[(n + k) * N1 + 1];
                            for (int64_t m = 0; m <= m_hi; ++m) {
                                next_row[m] += prev_row[m] * f;
                            }
                        }
                    }
                    for (int64_t n = 0; n <= next_deg; ++n) E[n] = NB[n];
                    for (int64_t n = next_deg + 1; n <= nmax; ++n) E[n] = ROW_EMPTY;
                    renorm_rows(next, E.data(), next_deg, next_cat, N1);
                    for (int64_t n = 0; n <= nmax; ++n) next_exp[n] = static_cast<double>(E[n]);
                    deg_cur = next_deg;
                    cat_cur = next_cat;
                }
            }
            const double* final_state = first ? (first + Cpad * stride2d) : buf.data();
            double* out = log_out + b * stride2d;
            for (int64_t n = 0; n <= nmax; ++n) {
                int64_t m_hi = (n <= deg_cur && E[n] != ROW_EMPTY) ? std::min(n, cat_cur) : -1;
                double row_log = (m_hi >= 0) ? static_cast<double>(E[n]) * M_LN2 : 0.0;
                for (int64_t m = 0; m <= m_hi; ++m) {
                    double v = final_state[n * N1 + m];
                    out[n * N1 + m] = (v > 0.0) ? (std::log(v) + row_log) : neg_inf;
                }
                for (int64_t m = m_hi + 1; m <= nmax; ++m) {
                    out[n * N1 + m] = neg_inf;
                }
            }
        }
    };

    pool().run(n_threads, worker);
    return err.load(std::memory_order_relaxed);
}

// One step of the 2-D product with the table kept: next = previous x (category c's polynomial
// sum_k qm[k] 2^qe[k] t^k y^[k>0], k = 0..d_eff), both row-scaled.  E holds previous's row
// exponents on entry and next's on exit; NB is scratch.  The same arithmetic as the stored-table
// branch of log_degree_2d_forward_cpp.
static inline void kept_category_step(const double* previous, double* next, int64_t* E, int64_t* NB,
                                      int64_t& deg_cur, int64_t& cat_cur, const double* qm,
                                      const int64_t* qe, int64_t d, int64_t d_eff, int64_t c,
                                      int64_t nmax, int64_t N1) {
    const int64_t next_deg = std::min(nmax, deg_cur + d);
    const int64_t next_cat = std::min(nmax, cat_cur + (d > 0 ? 1 : 0));
    const int64_t m_clear = std::min(nmax, c + 2);
    for (int64_t n = 0; n <= nmax; ++n) {
        int64_t m_hi = std::min(n, m_clear);
        std::fill(&next[n * N1], &next[n * N1 + m_hi + 1], 0.0);
    }
    target_exponents(E, NB, deg_cur, next_deg, qm, qe, d_eff, nmax);
    if (qm[0] > 0.0) {
        for (int64_t n = 0; n <= deg_cur; ++n) {
            if (E[n] == ROW_EMPTY) continue;
            double f0 = qm[0] * pow2i(E[n] + qe[0] - NB[n]);
            int64_t m_hi = std::min(n, cat_cur);
            for (int64_t m = 0; m <= m_hi; ++m) next[n * N1 + m] = previous[n * N1 + m] * f0;
        }
    }
    for (int64_t k = 1; k <= d_eff; ++k) {
        if (qm[k] == 0.0) continue;
        int64_t max_n = std::min(nmax - k, deg_cur);
        for (int64_t n = 0; n <= max_n; ++n) {
            if (E[n] == ROW_EMPTY) continue;
            double f = qm[k] * pow2i(E[n] + qe[k] - NB[n + k]);
            if (f == 0.0) continue;
            int64_t m_hi = std::min(n, cat_cur);
            const double* prev_row = &previous[n * N1];
            double* next_row = &next[(n + k) * N1 + 1];
            for (int64_t m = 0; m <= m_hi; ++m) next_row[m] += prev_row[m] * f;
        }
    }
    for (int64_t n = 0; n <= next_deg; ++n) E[n] = NB[n];
    for (int64_t n = next_deg + 1; n <= nmax; ++n) E[n] = ROW_EMPTY;
    renorm_rows(next, E, next_deg, next_cat, N1);
    deg_cur = next_deg;
    cat_cur = next_cat;
}

// Backward through one such step: given the gradient hnext on next, add each term's share
// hnext * term / next to its source entry (hprevious) and to its coefficient (hq[k]).  Returns
// false when the step only rescales rows (d = 0): then hprevious = hnext and the caller keeps
// hnext.  The loops run in the forward's order (source row, degree, contiguous m).
static inline bool kept_category_backward(const double* previous, const double* next, const int64_t* Ep,
                                          const int64_t* En, const double* q, const int64_t* qe,
                                          int64_t d, int64_t n_prev_hi, int64_t m_prev_cap,
                                          int64_t n_cur_hi, int64_t m_cur_cap, const double* hnext,
                                          double* hprevious, double* hq, int64_t N1) {
    if (d == 0) {
        double hq0 = 0.0;
        for (int64_t n = 0; n <= n_cur_hi; ++n) {
            if (Ep[n] == ROW_EMPTY || En[n] == ROW_EMPTY) continue;
            double f0 = q[0] * pow2i(Ep[n] + qe[0] - En[n]);
            int64_t m_hi = std::min(n, m_cur_cap);
            for (int64_t m = 0; m <= m_hi; ++m) {
                if (next[n * N1 + m] > 0.0 && hnext[n * N1 + m] != 0.0 && previous[n * N1 + m] * f0 > 0.0) {
                    hq0 += hnext[n * N1 + m];
                }
            }
        }
        hq[0] = hq0;
        return false;
    }
    for (int64_t n = 0; n <= n_prev_hi; ++n) {
        int64_t m_hi = std::min(n, m_prev_cap);
        std::fill(&hprevious[n * N1], &hprevious[n * N1 + m_hi + 1], 0.0);
    }
    double hq0 = 0.0;
    for (int64_t n = 0; n <= std::min(n_prev_hi, n_cur_hi); ++n) {             // c absent
        if (Ep[n] == ROW_EMPTY || En[n] == ROW_EMPTY || q[0] == 0.0) continue;
        const double f0 = q[0] * pow2i(Ep[n] + qe[0] - En[n]);
        const int64_t m_hi = std::min(std::min(n, m_prev_cap), m_cur_cap);
        const double* pr = &previous[n * N1];
        const double* nr = &next[n * N1];
        const double* hr = &hnext[n * N1];
        double* hp = &hprevious[n * N1];
        double acc = 0.0;
        for (int64_t m = 0; m <= m_hi; ++m) {
            const double term = pr[m] * f0;
            const double c0 = (term > 0.0 && nr[m] > 0.0 && hr[m] != 0.0) ? hr[m] * (term / nr[m]) : 0.0;
            hp[m] += c0;
            acc += c0;
        }
        hq0 += acc;
    }
    hq[0] = hq0;
    for (int64_t k = 1; k <= d; ++k) {                                        // k products of c
        if (q[k] == 0.0) continue;
        double hqk = 0.0;
        const int64_t n_hi = std::min(n_prev_hi, n_cur_hi - k);
        for (int64_t n = 0; n <= n_hi; ++n) {
            if (Ep[n] == ROW_EMPTY || En[n + k] == ROW_EMPTY) continue;
            const double f = q[k] * pow2i(Ep[n] + qe[k] - En[n + k]);
            const int64_t m_hi = std::min(std::min(n, m_prev_cap), std::min(n + k, m_cur_cap) - 1);
            const double* pr = &previous[n * N1];
            const double* nr = &next[(n + k) * N1 + 1];
            const double* hr = &hnext[(n + k) * N1 + 1];
            double* hp = &hprevious[n * N1];
            double acc = 0.0;
            for (int64_t m = 0; m <= m_hi; ++m) {
                const double term = pr[m] * f;
                const double ck = (term > 0.0 && nr[m] > 0.0 && hr[m] != 0.0) ? hr[m] * (term / nr[m]) : 0.0;
                hp[m] += ck;
                acc += ck;
            }
            hqk += acc;
        }
        hq[k] = hqk;
    }
    return true;
}

// Gradient of the 2-D product (log_degree_2d_forward_cpp) with respect to the log coefficients.
// Keeping every intermediate table costs B x (Cpad+1) x (nmax+1)^2 doubles (about 9 GB for 256
// Dunnhumby trips), so each trip keeps a table every S = ceil(sqrt(Cpad)) categories and
// recomputes the S tables of a segment just before stepping back through it: about one extra
// forward pass, and a working set of 2 sqrt(Cpad) tables per thread.  caps[b] <= nmax: trip b's
// tables are built for sizes n <= caps[b] only (rows n depend on rows <= n alone, so they are
// exact); the caller sets it where grad_log_out has no weight left above.
extern "C" int log_degree_2d_grad_cpp(
    int64_t B,
    int64_t Cpad,
    int64_t stride,
    int64_t nmax,
    const double* log_coeffs,
    const int64_t* degrees,
    const double* grad_log_out,
    const int64_t* caps,
    double* grad_log_coeffs
) {
    const int64_t N1 = nmax + 1;
    const int64_t stride2d = N1 * N1;
    const int64_t S = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::sqrt(static_cast<double>(Cpad)))));
    const int64_t n_ck = Cpad / S + 1;                       // kept tables at c = 0, S, 2S, ...
    std::atomic<int> err{0};
    int64_t n_threads = std::min<int64_t>(B, MAX_THREADS);
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](int64_t tid) {
        int64_t b_start = (B * tid) / n_threads;
        int64_t b_end = (B * (tid + 1)) / n_threads;
        std::vector<double> ck(n_ck * stride2d), seg((S + 1) * stride2d);
        std::vector<int64_t> ckE(n_ck * N1), segE((S + 1) * N1), ck_deg(n_ck), ck_cat(n_ck);
        std::vector<double> hnext(stride2d), hprevious(stride2d);
        std::vector<int64_t> deg_hist(Cpad + 1), cat_hist(Cpad + 1), dd(Cpad), E(N1), NB(N1);
        std::vector<double> qm(stride * Cpad);
        std::vector<int64_t> qe(stride * Cpad), d_eff(Cpad);

        for (int64_t b = b_start; b < b_end && err.load(std::memory_order_relaxed) == 0; ++b) {
            const int64_t cap = std::clamp<int64_t>(caps[b], 0, nmax);
            int64_t dc = 0, cc = 0;
            for (int64_t c = 0; c < Cpad; ++c) {
                deg_hist[c] = dc;
                cat_hist[c] = cc;
                dd[c] = std::clamp<int64_t>(degrees[b * Cpad + c], 0, cap);
                dc = std::min(cap, dc + dd[c]);
                cc = std::min(cap, cc + (dd[c] > 0 ? 1 : 0));
                const double* lp = log_coeffs + (b * Cpad + c) * stride;
                bool any = false;
                d_eff[c] = 0;
                for (int64_t k = 0; k <= dd[c]; ++k) {
                    split_log(lp[k], qm[c * stride + k], qe[c * stride + k]);
                    if (qm[c * stride + k] > 0.0) {
                        any = true;
                        if (k >= 1) d_eff[c] = k;
                    }
                }
                if (!any) {
                    err.store(-1, std::memory_order_relaxed);
                    return;
                }
            }
            deg_hist[Cpad] = dc;
            cat_hist[Cpad] = cc;

            // Forward, keeping the table every S categories (rolling pair of tables).
            double* cur = seg.data();
            double* nxt = seg.data() + stride2d;
            std::fill(E.begin(), E.end(), ROW_EMPTY);
            E[0] = 0;
            for (int64_t n = 0; n <= cap; ++n) {
                cur[n * N1 + 0] = 0.0;
                if (n >= 1) cur[n * N1 + 1] = 0.0;
            }
            cur[0] = 1.0;
            std::copy(cur, cur + stride2d, ck.begin());
            std::copy(E.begin(), E.end(), ckE.begin());
            ck_deg[0] = 0;
            ck_cat[0] = 0;
            int64_t deg_cur = 0, cat_cur = 0;
            for (int64_t c = 0; c < Cpad; ++c) {
                kept_category_step(cur, nxt, E.data(), NB.data(), deg_cur, cat_cur, &qm[c * stride],
                                   &qe[c * stride], dd[c], d_eff[c], c, cap, N1);
                std::swap(cur, nxt);
                const int64_t j = (c + 1) / S;
                if ((c + 1) % S == 0 && j < n_ck) {
                    std::copy(cur, cur + stride2d, &ck[j * stride2d]);
                    std::copy(E.begin(), E.end(), &ckE[j * N1]);
                    ck_deg[j] = deg_cur;
                    ck_cat[j] = cat_cur;
                }
            }

            // Backward, one segment at a time from the last.
            std::copy(grad_log_out + b * stride2d, grad_log_out + (b + 1) * stride2d, hnext.begin());
            for (int64_t j = (Cpad - 1) / S; j >= 0; --j) {
                int64_t c0 = j * S, c1 = std::min(Cpad, c0 + S);
                std::copy(&ck[j * stride2d], &ck[(j + 1) * stride2d], seg.begin());
                std::copy(&ckE[j * N1], &ckE[(j + 1) * N1], E.begin());
                std::copy(E.begin(), E.end(), segE.begin());
                int64_t dg = ck_deg[j], ct = ck_cat[j];
                for (int64_t c = c0; c < c1; ++c) {
                    kept_category_step(&seg[(c - c0) * stride2d], &seg[(c - c0 + 1) * stride2d], E.data(),
                                       NB.data(), dg, ct, &qm[c * stride], &qe[c * stride], dd[c], d_eff[c],
                                       c, cap, N1);
                    std::copy(E.begin(), E.end(), &segE[(c - c0 + 1) * N1]);
                }
                for (int64_t c = c1 - 1; c >= c0; --c) {
                    double* hq = grad_log_coeffs + (b * Cpad + c) * stride;
                    std::fill(hq, hq + stride, 0.0);
                    bool moved = kept_category_backward(
                        &seg[(c - c0) * stride2d], &seg[(c - c0 + 1) * stride2d], &segE[(c - c0) * N1],
                        &segE[(c - c0 + 1) * N1], &qm[c * stride], &qe[c * stride], dd[c], deg_hist[c],
                        cat_hist[c], deg_hist[c + 1], cat_hist[c + 1], hnext.data(), hprevious.data(), hq, N1);
                    if (moved) hnext.swap(hprevious);
                }
            }
        }
    };

    pool().run(n_threads, worker);
    return err.load(std::memory_order_relaxed);
}

extern "C" int log_partition_cpp(
    int64_t B,
    int64_t n_rows,
    int64_t T,
    int64_t Cpad,
    int64_t K,
    int64_t nmax,
    int64_t cat_rho0,
    int64_t min_cap,
    const double* u,
    const int64_t* row_of,
    const int64_t* item_pos,
    const int64_t* row_size,
    const int64_t* row_cat,
    const int64_t* row_trip,
    const int64_t* flat_slot,
    const double* rho_c,
    const double* rho_0,
    const double* f_extra,
    const double* row_cat_shift,
    const double* row_iv_shift,
    double* out_logz
) {
    const double neg_inf = -1.0 / 0.0;
    std::vector<int64_t> row_start(n_rows + 1, 0);
    std::vector<int64_t> trip_row_start(B + 1, 0);
    for (int64_t r = 0; r < n_rows; ++r) {
        row_start[r + 1] = row_start[r] + row_size[r];
        trip_row_start[row_trip[r] + 1] = r + 1;
    }
    for (int64_t b = 0; b < B; ++b) {
        if (trip_row_start[b + 1] < trip_row_start[b]) trip_row_start[b + 1] = trip_row_start[b];
    }

    bool ordered = true;
    for (int64_t t = 0; t < T; ++t) {
        if (row_start[row_of[t]] + item_pos[t] != t) {
            ordered = false;
            break;
        }
    }
    std::vector<double> W_vec;
    const double* W = u;
    if (!ordered) {
        W_vec.assign(T, neg_inf);
        for (int64_t t = 0; t < T; ++t) {
            W_vec[row_start[row_of[t]] + item_pos[t]] = u[t];
        }
        W = W_vec.data();
    }

    int64_t K1 = K + 1;
    int64_t n_threads = std::min<int64_t>(B, MAX_THREADS);
    if (n_threads < 1) n_threads = 1;

    pool().run(n_threads, [&](int64_t tid) {
        int64_t b_start = (B * tid) / n_threads;
        int64_t b_end = (B * (tid + 1)) / n_threads;
        std::vector<double> G_loc(Cpad * K1, neg_inf);
        std::vector<int64_t> slot_row_loc(Cpad, -1);
        std::vector<double> e_cur(K1);
        TripSamplerScratch sc;

        for (int64_t b = b_start; b < b_end; ++b) {
            std::fill(slot_row_loc.begin(), slot_row_loc.end(), -1);
            for (int64_t c = 0; c < Cpad; ++c) G_loc[c * K1 + 0] = 0.0;
            double trip_scale = neg_inf;
            for (int64_t r = trip_row_start[b]; r < trip_row_start[b + 1]; ++r) {
                int64_t len = row_size[r];
                if (len <= 0) continue;
                int64_t Kr = std::min(K, len);
                int64_t base = row_start[r];
                double rmax = neg_inf;
                for (int64_t p = 0; p < len; ++p) {
                    double w = W[base + p];
                    if (w > rmax) rmax = w;
                }
                if (rmax > trip_scale) trip_scale = rmax;
                std::fill(e_cur.begin(), e_cur.begin() + Kr + 1, 0.0);
                e_cur[0] = 1.0;
                for (int64_t p = 0; p < len; ++p) {
                    double we = std::exp(W[base + p] - rmax);
                    int64_t hi = std::min(p + 1, Kr);
                    for (int64_t k = hi; k >= 1; --k) {
                        e_cur[k] += e_cur[k - 1] * we;
                    }
                }
                int64_t c = flat_slot[r] - b * Cpad;
                slot_row_loc[c] = r;
                int64_t cat = row_cat[r];
                double rc = rho_c[cat];
                double cshift = row_cat_shift ? row_cat_shift[r] : 0.0;
                double iv_sh = row_iv_shift ? row_iv_shift[r] : 0.0;
                double* g_ptr = &G_loc[c * K1];
                g_ptr[0] = 0.0;
                for (int64_t k = 1; k <= Kr; ++k) {
                    double pen = rc * (0.5 * k * (k - 1)) + extra_count_pen(f_extra, cat, k) - iv_sh;
                    if (k >= 2) pen += (k - 1) * cshift;
                    g_ptr[k] = (e_cur[k] > 0.0)
                                   ? (std::log(e_cur[k]) + k * rmax - pen)
                                   : neg_inf;
                }
            }
            out_logz[b] = trip_log_partition_from_G(
                0, Cpad, K, K1, nmax, cat_rho0, min_cap, trip_scale,
                slot_row_loc.data(), row_size, G_loc.data(), rho_0, sc);
        }
    });
    return 0;
}

extern "C" int sample_slots_cpp(
    int64_t B,
    int64_t n_rows,
    int64_t T,
    int64_t Cpad,
    int64_t K,
    int64_t nmax,
    int64_t cat_rho0,
    int64_t min_cap,
    uint64_t seed,
    const double* u,
    const int64_t* row_of,
    const int64_t* item_pos,
    const int64_t* row_size,
    const int64_t* row_cat,
    const int64_t* flat_slot,
    const double* rho_c,
    const double* rho_0,
    const double* f_extra,
    const double* row_cat_shift,
    const double* row_iv_shift,
    uint8_t* out_mask
) {
    const double neg_inf = -1.0 / 0.0;
    SplitMix64 rng(seed);

    std::vector<int64_t> row_start(n_rows + 1, 0);
    for (int64_t r = 0; r < n_rows; ++r) {
        row_start[r + 1] = row_start[r] + row_size[r];
    }
    std::vector<double> W(T, neg_inf);
    std::vector<double> We(T, 0.0);
    std::vector<double> row_max(n_rows, neg_inf);
    std::vector<int64_t> slot_at(T, 0);
    for (int64_t t = 0; t < T; ++t) {
        int64_t r = row_of[t];
        int64_t p = item_pos[t];
        int64_t idx = row_start[r] + p;
        double w = u[t];
        W[idx] = w;
        slot_at[idx] = t;
        if (w > row_max[r]) row_max[r] = w;
    }

    int64_t K1 = K + 1;
    std::vector<double> G(B * Cpad * K1, neg_inf);
    std::vector<int64_t> slot_row(B * Cpad, -1);
    for (int64_t s = 0; s < B * Cpad; ++s) G[s * K1 + 0] = 0.0;

    int64_t n_threads_rows = std::min<int64_t>(n_rows, MAX_THREADS);
    if (n_threads_rows < 1) n_threads_rows = 1;
    pool().run(n_threads_rows, [&](int64_t tid) {
        int64_t r_start = (n_rows * tid) / n_threads_rows;
        int64_t r_end = (n_rows * (tid + 1)) / n_threads_rows;
        std::vector<double> e_cur(K1);
        for (int64_t r = r_start; r < r_end; ++r) {
            int64_t len = row_size[r];
            if (len <= 0) continue;
            int64_t Kr = std::min(K, len);
            double rmax = row_max[r];
            std::fill(e_cur.begin(), e_cur.begin() + Kr + 1, 0.0);
            e_cur[0] = 1.0;
            int64_t base = row_start[r];
            for (int64_t p = 0; p < len; ++p) {
                double we = std::exp(W[base + p] - rmax);
                We[base + p] = we;
                int64_t hi = std::min(p + 1, Kr);
                for (int64_t k = hi; k >= 1; --k) {
                    e_cur[k] += e_cur[k - 1] * we;
                }
            }
            int64_t fslot = flat_slot[r];
            slot_row[fslot] = r;
            int64_t cat = row_cat[r];
            double rc = rho_c[cat];
            double cshift = row_cat_shift ? row_cat_shift[r] : 0.0;
            double iv_sh = row_iv_shift ? row_iv_shift[r] : 0.0;
            double* g_ptr = &G[fslot * K1];
            g_ptr[0] = 0.0;
            for (int64_t k = 1; k <= Kr; ++k) {
                double pen = rc * (0.5 * k * (k - 1)) + extra_count_pen(f_extra, cat, k) - iv_sh;
                if (k >= 2) pen += (k - 1) * cshift;
                g_ptr[k] = (e_cur[k] > 0.0)
                               ? (std::log(e_cur[k]) + k * rmax - pen)
                               : neg_inf;
            }
        }
    });

    int64_t N1 = nmax + 1;
    std::vector<int64_t> need_row(n_rows, 0);

    if (cat_rho0) {
        std::vector<uint64_t> trip_seeds(B);
        for (int64_t b = 0; b < B; ++b) trip_seeds[b] = rng.next_u64();
        std::atomic<int> err{0};
        int64_t n_threads = std::min<int64_t>(B, MAX_THREADS);
        if (n_threads < 1) n_threads = 1;
        pool().run(n_threads, [&](int64_t tid) {
            int64_t b_start = (B * tid) / n_threads;
            int64_t b_end = (B * (tid + 1)) / n_threads;
            TripSamplerScratch sc;
            for (int64_t b = b_start; b < b_end; ++b) {
                SplitMix64 rng_b(trip_seeds[b]);
                int rc = sample_trip_categories_2d_impl(
                    b, Cpad, K, K1, nmax, min_cap, slot_row.data(), row_size, G.data(), rho_0,
                    rng_b, sc, need_row);
                if (rc != 0) {
                    err.store(rc, std::memory_order_relaxed);
                    return;
                }
            }
        });
        int rc = err.load(std::memory_order_relaxed);
        if (rc != 0) return rc;
    } else {
        std::vector<double> Fs((Cpad + 1) * N1, neg_inf);
        std::vector<double> fl(N1), gl(K1), conv(N1), lp(N1);
        for (int64_t b = 0; b < B; ++b) {
            Fs[0] = 0.0;
            for (int64_t n = 1; n <= nmax; ++n) Fs[n] = neg_inf;
            int64_t deg_so_far = 0;
            for (int64_t c = 0; c < Cpad; ++c) {
                const double* f_prev = &Fs[c * N1];
                double* f_cur = &Fs[(c + 1) * N1];
                int64_t fslot = b * Cpad + c;
                int64_t r = slot_row[fslot];
                if (r < 0) {
                    for (int64_t n = 0; n <= deg_so_far; ++n) f_cur[n] = f_prev[n];
                    for (int64_t n = deg_so_far + 1; n <= nmax; ++n) f_cur[n] = neg_inf;
                    continue;
                }
                int64_t Kr = std::min(K, row_size[r]);
                const double* g_ptr = &G[fslot * K1];
                double mf = neg_inf, mg = neg_inf;
                for (int64_t n = 0; n <= deg_so_far; ++n) if (f_prev[n] > mf) mf = f_prev[n];
                for (int64_t k = 0; k <= Kr; ++k) if (g_ptr[k] > mg) mg = g_ptr[k];
                for (int64_t n = 0; n <= deg_so_far; ++n)
                    fl[n] = (f_prev[n] > neg_inf) ? std::exp(f_prev[n] - mf) : 0.0;
                for (int64_t k = 0; k <= Kr; ++k)
                    gl[k] = (g_ptr[k] > neg_inf) ? std::exp(g_ptr[k] - mg) : 0.0;
                int64_t next_deg = std::min(nmax, deg_so_far + Kr);
                std::fill(conv.begin(), conv.begin() + next_deg + 1, 0.0);
                for (int64_t k = 0; k <= Kr; ++k) {
                    double gk = gl[k];
                    if (gk == 0.0) continue;
                    int64_t max_n = std::min(nmax - k, deg_so_far);
                    for (int64_t n = 0; n <= max_n; ++n) {
                        conv[n + k] += fl[n] * gk;
                    }
                }
                double shift = mf + mg;
                for (int64_t n = 0; n <= next_deg; ++n) {
                    f_cur[n] = (conv[n] > 0.0) ? (shift + std::log(conv[n])) : neg_inf;
                }
                for (int64_t n = next_deg + 1; n <= nmax; ++n) f_cur[n] = neg_inf;
                deg_so_far = next_deg;
            }

            const double* f_last = &Fs[Cpad * N1];
            double max_lp = neg_inf;
            for (int64_t n = 1; n <= nmax; ++n) {
                lp[n] = (f_last[n] > neg_inf) ? (f_last[n] - rho_0[n]) : neg_inf;
                if (lp[n] > max_lp) max_lp = lp[n];
            }
            double sum_p = 0.0;
            for (int64_t n = 1; n <= nmax; ++n) {
                lp[n] = (lp[n] > neg_inf) ? std::exp(lp[n] - max_lp) : 0.0;
                sum_p += lp[n];
            }
            double target = rng.uniform() * sum_p;
            double acc = 0.0;
            int64_t rem = nmax;
            for (int64_t n = 1; n <= nmax; ++n) {
                acc += lp[n];
                if (target <= acc) {
                    rem = n;
                    break;
                }
            }

            for (int64_t c = Cpad - 1; c >= 0; --c) {
                if (rem == 0) break;
                int64_t fslot = b * Cpad + c;
                int64_t r = slot_row[fslot];
                if (r < 0) continue;
                int64_t Kr = std::min({K, row_size[r], rem});
                const double* f_prev = &Fs[c * N1];
                const double* g_ptr = &G[fslot * K1];
                double max_c = neg_inf;
                for (int64_t k = 0; k <= Kr; ++k) {
                    double v = (f_prev[rem - k] > neg_inf && g_ptr[k] > neg_inf)
                                   ? (f_prev[rem - k] + g_ptr[k])
                                   : neg_inf;
                    lp[k] = v;
                    if (v > max_c) max_c = v;
                }
                double sum_c = 0.0;
                for (int64_t k = 0; k <= Kr; ++k) {
                    lp[k] = (lp[k] > neg_inf) ? std::exp(lp[k] - max_c) : 0.0;
                    sum_c += lp[k];
                }
                double targ_c = rng.uniform() * sum_c;
                double acc_c = 0.0;
                int64_t chosen_k = Kr;
                for (int64_t k = 0; k <= Kr; ++k) {
                    acc_c += lp[k];
                    if (targ_c <= acc_c) {
                        chosen_k = k;
                        break;
                    }
                }
                need_row[r] = chosen_k;
                rem -= chosen_k;
            }
            if (rem != 0) return -1;
        }
    }

    std::fill(out_mask, out_mask + T, 0);
    std::vector<double> E_row;
    for (int64_t r = 0; r < n_rows; ++r) {
        int64_t need = need_row[r];
        if (need <= 0) continue;
        int64_t len = row_size[r];
        int64_t base = row_start[r];
        if (need == len) {
            for (int64_t p = 0; p < len; ++p) out_mask[slot_at[base + p]] = 1;
            continue;
        }
        int64_t Kr = need;
        int64_t stride = Kr + 1;
        E_row.assign((len + 1) * stride, 0.0);
        E_row[0] = 1.0;
        for (int64_t p = 0; p < len; ++p) {
            double we = We[base + p];
            const double* ep = &E_row[p * stride];
            double* en = &E_row[(p + 1) * stride];
            en[0] = 1.0;
            int64_t hi = std::min(p + 1, Kr);
            for (int64_t k = 1; k <= hi; ++k) {
                en[k] = ep[k] + ep[k - 1] * we;
            }
        }
        for (int64_t p = len - 1; p >= 0 && need > 0; --p) {
            if (p + 1 == need) {
                for (int64_t q = 0; q <= p; ++q) out_mask[slot_at[base + q]] = 1;
                need = 0;
                break;
            }
            double we = We[base + p];
            double num = we * E_row[p * stride + (need - 1)];
            double den = E_row[(p + 1) * stride + need];
            double prob = (den > 0.0) ? std::min(1.0, num / den) : 0.0;
            if (rng.uniform() < prob) {
                out_mask[slot_at[base + p]] = 1;
                need -= 1;
            }
        }
        if (need != 0) return -2;
    }
    return 0;
}

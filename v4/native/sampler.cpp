// Exact basket sampler and Gibbs sweeps of the Version-4 law, plus the 2-D size dynamic
// program used when the size potential is on the number of categories present (M3).
// Plain C ABI, loaded with ctypes by v4/kernels.py (built with clang++ on first use).
//
//   sample_slots_cpp            one exact draw of S per trip given slot utilities u:
//                               per-row ESPs, a forward DP over categories and sizes, then
//                               backward sampling of each category's count and its products
//   gibbs_sweeps_cpp            block Gibbs on (z, S): z ~ N(s(S), I), then S | z exactly
//   log_degree_2d_forward_cpp   log coefficients A[n, m] = total weight of baskets with
//   log_degree_2d_backward_cpp  n products in m categories, and its adjoint
//
// Optional terms arrive as null pointers when absent: f_extra (M1 count potentials),
// row_cat_shift / psi_cat (M2 presence coupling), row_iv_shift (M4c inclusive value);
// cat_rho0 = 1 puts the size potential rho_0 on the number of categories (M3).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace {

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

int sample_trip_categories_2d(
    int64_t b,
    int64_t Cpad,
    int64_t K,
    int64_t K1,
    int64_t nmax,
    int64_t N1,
    const int64_t* slot_row,
    const int64_t* row_size,
    const double* G,
    const double* rho_0,
    SplitMix64& rng,
    std::vector<double>& Fs2d,
    std::vector<double>& fl2d,
    std::vector<double>& conv2d,
    std::vector<double>& gl,
    std::vector<double>& lp2d,
    std::vector<int64_t>& need_row
) {
    const double neg_inf = -1.0 / 0.0;
    const int64_t stride2d = N1 * N1;
    if ((int64_t)gl.size() < (Cpad + 1) * K1) gl.resize((Cpad + 1) * K1, 0.0);
    Fs2d[0] = 1.0;
    int64_t deg_so_far = 0;
    int64_t cat_so_far = 0;
    double log_scale = 0.0;
    std::vector<int64_t> deg_hist(Cpad + 1);
    std::vector<int64_t> cat_hist(Cpad + 1);

    for (int64_t c = 0; c < Cpad; ++c) {
        deg_hist[c] = deg_so_far;
        cat_hist[c] = cat_so_far;
        const double* f_prev = &Fs2d[c * stride2d];
        double* f_cur = &Fs2d[(c + 1) * stride2d];
        double* q_c = &gl[c * K1];
        int64_t fslot = b * Cpad + c;
        int64_t r = slot_row[fslot];
        if (r < 0) {
            q_c[0] = 1.0;
            for (int64_t n = 0; n <= deg_so_far; ++n) {
                int64_t m_hi = std::min(n, cat_so_far);
                std::copy(&f_prev[n * N1], &f_prev[n * N1 + m_hi + 1], &f_cur[n * N1]);
            }
            continue;
        }
        int64_t Kr = std::min(K, row_size[r]);
        const double* g_ptr = &G[fslot * K1];
        double mg = neg_inf;
        for (int64_t k = 0; k <= Kr; ++k) {
            if (g_ptr[k] > mg) mg = g_ptr[k];
        }
        for (int64_t k = 0; k <= Kr; ++k) {
            q_c[k] = (g_ptr[k] > neg_inf) ? std::exp(g_ptr[k] - mg) : 0.0;
        }
        int64_t next_deg = std::min(nmax, deg_so_far + Kr);
        int64_t next_cat = std::min(nmax, cat_so_far + 1);
        for (int64_t n = 0; n <= next_deg; ++n) {
            int64_t m_hi = std::min(n, next_cat);
            std::fill(&f_cur[n * N1], &f_cur[n * N1 + m_hi + 1], 0.0);
        }
        double g0 = q_c[0];
        if (g0 > 0.0) {
            for (int64_t n = 0; n <= deg_so_far; ++n) {
                int64_t m_hi = std::min(n, cat_so_far);
                for (int64_t m = 0; m <= m_hi; ++m) {
                    f_cur[n * N1 + m] = f_prev[n * N1 + m] * g0;
                }
            }
        }
        for (int64_t k = 1; k <= Kr; ++k) {
            double gk = q_c[k];
            if (gk == 0.0) continue;
            int64_t max_n = std::min(nmax - k, deg_so_far);
            for (int64_t n = 0; n <= max_n; ++n) {
                int64_t m_hi = std::min(n, cat_so_far);
                const double* fp_row = &f_prev[n * N1];
                double* fc_row = &f_cur[(n + k) * N1 + 1];
                for (int64_t m = 0; m <= m_hi; ++m) {
                    fc_row[m] += fp_row[m] * gk;
                }
            }
        }
        double max_cv = 0.0;
        for (int64_t n = 0; n <= next_deg; ++n) {
            int64_t m_hi = std::min(n, next_cat);
            for (int64_t m = 0; m <= m_hi; ++m) {
                double cv = f_cur[n * N1 + m];
                if (cv > max_cv) max_cv = cv;
            }
        }
        if (max_cv <= 0.0) return -1;
        double inv_max = 1.0 / max_cv;
        for (int64_t n = 0; n <= next_deg; ++n) {
            int64_t m_hi = std::min(n, next_cat);
            for (int64_t m = 0; m <= m_hi; ++m) {
                f_cur[n * N1 + m] *= inv_max;
            }
        }
        log_scale += mg + std::log(max_cv);
        deg_so_far = next_deg;
        cat_so_far = next_cat;
    }

    const double* f_last = &Fs2d[Cpad * stride2d];
    double max_lp = neg_inf;
    for (int64_t n = 1; n <= nmax; ++n) {
        int64_t m_hi = (n <= deg_so_far) ? std::min(n, cat_so_far) : 0;
        for (int64_t m = 1; m <= m_hi; ++m) {
            double v = f_last[n * N1 + m];
            double val = (v > 0.0) ? (std::log(v) + log_scale - rho_0[m]) : neg_inf;
            lp2d[n * N1 + m] = val;
            if (val > max_lp) max_lp = val;
        }
        for (int64_t m = m_hi + 1; m <= n; ++m) {
            lp2d[n * N1 + m] = neg_inf;
        }
    }
    double sum_p = 0.0;
    for (int64_t n = 1; n <= nmax; ++n) {
        for (int64_t m = 1; m <= n; ++m) {
            double val = lp2d[n * N1 + m];
            double p = (val > neg_inf) ? std::exp(val - max_lp) : 0.0;
            lp2d[n * N1 + m] = p;
            sum_p += p;
        }
    }
    double target = rng.uniform() * sum_p;
    double acc = 0.0;
    int64_t rem_n = 1, rem_m = 1;
    bool found = false;
    for (int64_t n = 1; n <= nmax && !found; ++n) {
        for (int64_t m = 1; m <= n; ++m) {
            acc += lp2d[n * N1 + m];
            if (target <= acc) {
                rem_n = n;
                rem_m = m;
                found = true;
                break;
            }
        }
    }

    double* w_k = &gl[Cpad * K1];
    for (int64_t c = Cpad - 1; c >= 0; --c) {
        if (rem_n == 0) break;
        int64_t fslot = b * Cpad + c;
        int64_t r = slot_row[fslot];
        if (r < 0) continue;
        int64_t Kr = std::min({K, row_size[r], rem_n});
        const double* f_prev = &Fs2d[c * stride2d];
        const double* q_c = &gl[c * K1];
        int64_t d_prev = deg_hist[c];
        int64_t c_prev = cat_hist[c];
        double sum_c = 0.0;
        for (int64_t k = 0; k <= Kr; ++k) {
            double w = 0.0;
            if (k == 0) {
                if (rem_n <= d_prev && rem_m <= std::min(rem_n, c_prev)) {
                    w = f_prev[rem_n * N1 + rem_m] * q_c[0];
                }
            } else if (rem_m >= 1) {
                int64_t rn = rem_n - k;
                int64_t rm = rem_m - 1;
                if (rn <= d_prev && rm <= std::min(rn, c_prev)) {
                    w = f_prev[rn * N1 + rm] * q_c[k];
                }
            }
            w_k[k] = w;
            sum_c += w;
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

}  // namespace

extern "C" int log_degree_2d_forward_cpp(
    int64_t B,
    int64_t Cpad,
    int64_t stride,
    int64_t nmax,
    const double* log_coeffs,
    const int64_t* degrees,
    double* log_out,
    double* q_all,
    double* states
) {
    const double neg_inf = -1.0 / 0.0;
    const int64_t N1 = nmax + 1;
    const int64_t stride2d = N1 * N1;
    std::atomic<int> err{0};
    int64_t n_threads = std::min<int64_t>(B, 4);
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](int64_t tid) {
        int64_t b_start = (B * tid) / n_threads;
        int64_t b_end = (B * (tid + 1)) / n_threads;
        std::vector<double> buf0(states ? 0 : stride2d, 0.0);
        std::vector<double> buf1(states ? 0 : stride2d, 0.0);
        std::vector<double> q_loc(q_all ? 0 : stride, 0.0);

        for (int64_t b = b_start; b < b_end; ++b) {
            double* first = states ? (states + b * (Cpad + 1) * stride2d) : nullptr;
            if (first) {
                for (int64_t n = 0; n <= nmax; ++n) {
                    first[n * N1 + 0] = 0.0;
                    if (n >= 1) first[n * N1 + 1] = 0.0;
                }
                first[0] = 1.0;
            } else {
                buf0[0] = 1.0;
            }
            double log_scale = 0.0;
            int64_t deg_cur = 0;
            int64_t cat_cur = 0;

            for (int64_t c = 0; c < Cpad; ++c) {
                const int64_t d = std::clamp<int64_t>(degrees[b * Cpad + c], 0, nmax);
                const double* lp = log_coeffs + (b * Cpad + c) * stride;
                double* q = q_all ? (q_all + (b * Cpad + c) * stride) : q_loc.data();
                double maximum = neg_inf;
                for (int64_t k = 0; k <= d; ++k) {
                    if (lp[k] > maximum) maximum = lp[k];
                }
                if (!std::isfinite(maximum)) {
                    err.store(-1, std::memory_order_relaxed);
                    return;
                }
                for (int64_t k = 0; k <= d; ++k) {
                    q[k] = std::exp(lp[k] - maximum);
                }
                log_scale += maximum;
                const double* previous = first ? (first + c * stride2d)
                                               : ((c & 1) ? buf1.data() : buf0.data());
                double* next = first ? (first + (c + 1) * stride2d)
                                     : ((c & 1) ? buf0.data() : buf1.data());
                int64_t next_deg = std::min(nmax, deg_cur + d);
                int64_t next_cat = std::min(nmax, cat_cur + (d > 0 ? 1 : 0));

                if (first) {
                    int64_t m_clear = std::min(nmax, c + 2);
                    for (int64_t n = 0; n <= nmax; ++n) {
                        int64_t m_hi = std::min(n, m_clear);
                        std::fill(&next[n * N1], &next[n * N1 + m_hi + 1], 0.0);
                    }
                } else {
                    for (int64_t n = 0; n <= next_deg; ++n) {
                        int64_t m_hi = std::min(n, next_cat);
                        std::fill(&next[n * N1], &next[n * N1 + m_hi + 1], 0.0);
                    }
                }

                double q0 = q[0];
                if (q0 > 0.0) {
                    for (int64_t n = 0; n <= deg_cur; ++n) {
                        int64_t m_hi = std::min(n, cat_cur);
                        for (int64_t m = 0; m <= m_hi; ++m) {
                            next[n * N1 + m] = previous[n * N1 + m] * q0;
                        }
                    }
                }
                for (int64_t k = 1; k <= d; ++k) {
                    double qk = q[k];
                    if (qk == 0.0) continue;
                    int64_t max_n = std::min(nmax - k, deg_cur);
                    for (int64_t n = 0; n <= max_n; ++n) {
                        int64_t m_hi = std::min(n, cat_cur);
                        const double* prev_row = &previous[n * N1];
                        double* next_row = &next[(n + k) * N1 + 1];
                        for (int64_t m = 0; m <= m_hi; ++m) {
                            next_row[m] += prev_row[m] * qk;
                        }
                    }
                }
                deg_cur = next_deg;
                cat_cur = next_cat;
            }
            const double* final_state = first ? (first + Cpad * stride2d)
                                              : ((Cpad & 1) ? buf1.data() : buf0.data());
            double* out = log_out + b * stride2d;
            for (int64_t n = 0; n <= nmax; ++n) {
                int64_t m_hi = (n <= deg_cur) ? std::min(n, cat_cur) : -1;
                for (int64_t m = 0; m <= m_hi; ++m) {
                    double v = final_state[n * N1 + m];
                    out[n * N1 + m] = (v > 0.0) ? (std::log(v) + log_scale) : neg_inf;
                }
                for (int64_t m = m_hi + 1; m <= nmax; ++m) {
                    out[n * N1 + m] = neg_inf;
                }
            }
        }
    };

    if (n_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int64_t t = 0; t < n_threads; ++t) threads.emplace_back(worker, t);
        for (auto& th : threads) th.join();
    }
    return err.load(std::memory_order_relaxed);
}

extern "C" int log_degree_2d_backward_cpp(
    int64_t B,
    int64_t Cpad,
    int64_t stride,
    int64_t nmax,
    const double* grad_log_out,
    const double* q_all,
    const int64_t* degrees,
    const double* states,
    double* grad_log_coeffs
) {
    const int64_t N1 = nmax + 1;
    const int64_t stride2d = N1 * N1;
    int64_t n_threads = std::min<int64_t>(B, 4);
    if (n_threads < 1) n_threads = 1;

    auto worker = [&](int64_t tid) {
        int64_t b_start = (B * tid) / n_threads;
        int64_t b_end = (B * (tid + 1)) / n_threads;
        std::vector<double> hnext(stride2d, 0.0), hprevious(stride2d, 0.0);

        for (int64_t b = b_start; b < b_end; ++b) {
            std::copy(grad_log_out + b * stride2d, grad_log_out + (b + 1) * stride2d, hnext.begin());
            const double* first = states + b * (Cpad + 1) * stride2d;
            for (int64_t c = Cpad - 1; c >= 0; --c) {
                int64_t m_cap_prev = std::min(nmax, c + 1);
                for (int64_t n = 0; n <= nmax; ++n) {
                    int64_t m_hi = std::min(n, m_cap_prev);
                    std::fill(&hprevious[n * N1], &hprevious[n * N1 + m_hi + 1], 0.0);
                }
                const int64_t d = std::clamp<int64_t>(degrees[b * Cpad + c], 0, nmax);
                const double* previous = first + c * stride2d;
                const double* next = previous + stride2d;
                const double* q = q_all + (b * Cpad + c) * stride;
                double* hq = grad_log_coeffs + (b * Cpad + c) * stride;
                std::fill(hq, hq + stride, 0.0);
                int64_t m_cap = std::min(nmax, c + 1);
                for (int64_t n = 0; n <= nmax; ++n) {
                    int64_t m_hi = std::min(n, m_cap);
                    for (int64_t m = 0; m <= m_hi; ++m) {
                        double nxt = next[n * N1 + m];
                        double hn = hnext[n * N1 + m];
                        if (nxt <= 0.0 || hn == 0.0) continue;
                        double inv = 1.0 / nxt;
                        bool ord = std::isfinite(inv);
                        double term0 = previous[n * N1 + m] * q[0];
                        if (term0 > 0.0) {
                            double c0 = hn * (ord ? term0 * inv : term0 / nxt);
                            hprevious[n * N1 + m] += c0;
                            hq[0] += c0;
                        }
                        if (m >= 1) {
                            int64_t upper = std::min(d, n - (m - 1));
                            for (int64_t k = 1; k <= upper; ++k) {
                                double term = previous[(n - k) * N1 + (m - 1)] * q[k];
                                if (term > 0.0) {
                                    double ck = hn * (ord ? term * inv : term / nxt);
                                    hprevious[(n - k) * N1 + (m - 1)] += ck;
                                    hq[k] += ck;
                                }
                            }
                        }
                    }
                }
                hnext.swap(hprevious);
            }
        }
    };

    if (n_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(n_threads);
        for (int64_t t = 0; t < n_threads; ++t) threads.emplace_back(worker, t);
        for (auto& th : threads) th.join();
    }
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

    std::vector<double> e_cur(K1);
    for (int64_t r = 0; r < n_rows; ++r) {
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

    int64_t N1 = nmax + 1;
    std::vector<int64_t> need_row(n_rows, 0);
    std::vector<double> gl(K1);

    if (cat_rho0) {
        std::vector<uint64_t> trip_seeds(B);
        for (int64_t b = 0; b < B; ++b) trip_seeds[b] = rng.next_u64();
        std::atomic<int> err{0};
        int64_t n_threads = std::min<int64_t>(B, 4);
        if (n_threads < 1) n_threads = 1;
        auto worker = [&](int64_t tid) {
            int64_t b_start = (B * tid) / n_threads;
            int64_t b_end = (B * (tid + 1)) / n_threads;
            std::vector<double> Fs2d((Cpad + 1) * N1 * N1, neg_inf);
            std::vector<double> fl2d(N1 * N1, 0.0), conv2d(N1 * N1, 0.0), lp2d(N1 * N1, neg_inf);
            std::vector<double> gl_loc((Cpad + 1) * K1, 0.0);
            for (int64_t b = b_start; b < b_end; ++b) {
                SplitMix64 rng_b(trip_seeds[b]);
                int rc = sample_trip_categories_2d(
                    b, Cpad, K, K1, nmax, N1, slot_row.data(), row_size, G.data(), rho_0,
                    rng_b, Fs2d, fl2d, conv2d, gl_loc, lp2d, need_row);
                if (rc != 0) {
                    err.store(rc, std::memory_order_relaxed);
                    return;
                }
            }
        };
        if (n_threads == 1) {
            worker(0);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(n_threads);
            for (int64_t t = 0; t < n_threads; ++t) threads.emplace_back(worker, t);
            for (auto& th : threads) th.join();
        }
        int rc = err.load(std::memory_order_relaxed);
        if (rc != 0) return rc;
    } else {
        std::vector<double> Fs((Cpad + 1) * N1, neg_inf);
        std::vector<double> fl(N1), conv(N1), lp(N1);
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

extern "C" int gibbs_sweeps_cpp(
    int64_t B,
    int64_t n_rows,
    int64_t T,
    int64_t Cpad,
    int64_t K,
    int64_t nmax,
    int64_t rank,
    int64_t sweeps,
    int64_t cat_rho0,
    double sqrt_beta,
    uint64_t seed,
    const double* b_base,
    const double* phi_slot,
    const int64_t* item_trip,
    const int64_t* row_of,
    const int64_t* item_pos,
    const int64_t* row_size,
    const int64_t* row_cat,
    const int64_t* row_trip,
    const int64_t* flat_slot,
    const double* rho_c,
    const double* rho_0,
    const double* f_extra,
    const double* psi_cat,
    const double* row_iv_shift,
    uint8_t* mask
) {
    const double neg_inf = -1.0 / 0.0;
    const double beta = sqrt_beta * sqrt_beta;
    SplitMix64 rng(seed);

    std::vector<int64_t> row_start(n_rows + 1, 0);
    for (int64_t r = 0; r < n_rows; ++r) {
        row_start[r + 1] = row_start[r] + row_size[r];
    }
    std::vector<double> b_ord(T, 0.0);
    std::vector<double> phi_ord(T * rank, 0.0);
    std::vector<int64_t> slot_at(T, 0);
    std::vector<double> s(B * rank, 0.0);
    std::vector<int64_t> init_row_cnt(psi_cat ? n_rows : 0, 0);

    for (int64_t t = 0; t < T; ++t) {
        int64_t r = row_of[t];
        int64_t p = item_pos[t];
        int64_t idx = row_start[r] + p;
        b_ord[idx] = b_base[t];
        slot_at[idx] = t;
        const double* ps = &phi_slot[t * rank];
        double* po = &phi_ord[idx * rank];
        for (int64_t d = 0; d < rank; ++d) po[d] = ps[d];
        if (mask[t]) {
            double* sb = &s[item_trip[t] * rank];
            for (int64_t d = 0; d < rank; ++d) sb[d] += ps[d];
            if (psi_cat) init_row_cnt[r] += 1;
        }
    }
    if (psi_cat) {
        for (int64_t r = 0; r < n_rows; ++r) {
            if (init_row_cnt[r] >= 2) {
                double mult = static_cast<double>(init_row_cnt[r] - 1);
                double* sb = &s[row_trip[r] * rank];
                const double* pc = &psi_cat[row_cat[r] * rank];
                for (int64_t d = 0; d < rank; ++d) sb[d] -= mult * pc[d];
            }
        }
    }

    int64_t K1 = K + 1;
    int64_t N1 = nmax + 1;
    std::vector<double> G(B * Cpad * K1, neg_inf);
    std::vector<int64_t> slot_row(B * Cpad, -1);
    for (int64_t i = 0; i < B * Cpad; ++i) G[i * K1 + 0] = 0.0;
    for (int64_t r = 0; r < n_rows; ++r) {
        if (row_size[r] > 0) slot_row[flat_slot[r]] = r;
    }

    std::vector<double> z(B * rank, 0.0);
    std::vector<double> We(T, 0.0);
    std::vector<double> w_row(K1 + 256);
    std::vector<double> e_cur(K1);
    std::vector<double> Fs(cat_rho0 ? 0 : (Cpad + 1) * N1, neg_inf);
    std::vector<double> fl(cat_rho0 ? 0 : N1), gl(K1), conv(cat_rho0 ? 0 : N1), lp(cat_rho0 ? 0 : N1);
    std::vector<double> Fs2d(cat_rho0 ? (Cpad + 1) * N1 * N1 : 0, neg_inf);
    std::vector<double> fl2d(cat_rho0 ? N1 * N1 : 0, 0.0), conv2d(cat_rho0 ? N1 * N1 : 0, 0.0), lp2d(cat_rho0 ? N1 * N1 : 0, neg_inf);
    std::vector<int64_t> need_row(n_rows, 0);
    std::vector<double> E_row;

    for (int64_t sw = 0; sw < sweeps; ++sw) {
        for (int64_t b = 0; b < B; ++b) {
            const double* sb = &s[b * rank];
            double* zb = &z[b * rank];
            int64_t d = 0;
            for (; d + 1 < rank; d += 2) {
                double n0, n1;
                rng.normalpair(n0, n1);
                zb[d] = sqrt_beta * (sb[d] + n0);
                zb[d + 1] = sqrt_beta * (sb[d + 1] + n1);
            }
            if (d < rank) {
                double n0, n1;
                rng.normalpair(n0, n1);
                zb[d] = sqrt_beta * (sb[d] + n0);
            }
        }

        for (int64_t r = 0; r < n_rows; ++r) {
            int64_t len = row_size[r];
            if (len <= 0) continue;
            int64_t Kr = std::min(K, len);
            int64_t base = row_start[r];
            const double* zb = &z[row_trip[r] * rank];
            if ((int64_t)w_row.size() < len) w_row.resize(len);
            double rmax = neg_inf;
            for (int64_t p = 0; p < len; ++p) {
                const double* po = &phi_ord[(base + p) * rank];
                double dot = 0.0;
                for (int64_t d = 0; d < rank; ++d) dot += zb[d] * po[d];
                double u_val = b_ord[base + p] + dot;
                w_row[p] = u_val;
                if (u_val > rmax) rmax = u_val;
            }
            std::fill(e_cur.begin(), e_cur.begin() + Kr + 1, 0.0);
            e_cur[0] = 1.0;
            for (int64_t p = 0; p < len; ++p) {
                double we = std::exp(w_row[p] - rmax);
                We[base + p] = we;
                int64_t hi = std::min(p + 1, Kr);
                for (int64_t k = hi; k >= 1; --k) {
                    e_cur[k] += e_cur[k - 1] * we;
                }
            }
            int64_t fslot = flat_slot[r];
            int64_t cat = row_cat[r];
            double rc = rho_c[cat];
            double cshift = 0.0;
            if (psi_cat && Kr >= 2) {
                const double* pc = &psi_cat[cat * rank];
                double dot_c = 0.0, norm_c = 0.0;
                for (int64_t d = 0; d < rank; ++d) {
                    dot_c += zb[d] * pc[d];
                    norm_c += pc[d] * pc[d];
                }
                cshift = dot_c - 0.5 * beta * norm_c;
            }
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

        std::fill(need_row.begin(), need_row.end(), 0);
        if (cat_rho0) {
            std::vector<uint64_t> trip_seeds(B);
            for (int64_t b = 0; b < B; ++b) trip_seeds[b] = rng.next_u64();
            std::atomic<int> err{0};
            int64_t n_threads = std::min<int64_t>(B, 4);
            if (n_threads < 1) n_threads = 1;
            auto worker = [&](int64_t tid) {
                int64_t b_start = (B * tid) / n_threads;
                int64_t b_end = (B * (tid + 1)) / n_threads;
                std::vector<double> Fs2d_loc((Cpad + 1) * N1 * N1, neg_inf);
                std::vector<double> fl2d_loc(N1 * N1, 0.0), conv2d_loc(N1 * N1, 0.0), lp2d_loc(N1 * N1, neg_inf);
                std::vector<double> gl_loc((Cpad + 1) * K1, 0.0);
                for (int64_t b = b_start; b < b_end; ++b) {
                    SplitMix64 rng_b(trip_seeds[b]);
                    int rc = sample_trip_categories_2d(
                        b, Cpad, K, K1, nmax, N1, slot_row.data(), row_size, G.data(), rho_0,
                        rng_b, Fs2d_loc, fl2d_loc, conv2d_loc, gl_loc, lp2d_loc, need_row);
                    if (rc != 0) {
                        err.store(rc, std::memory_order_relaxed);
                        return;
                    }
                }
            };
            if (n_threads == 1) {
                worker(0);
            } else {
                std::vector<std::thread> threads;
                threads.reserve(n_threads);
                for (int64_t t = 0; t < n_threads; ++t) threads.emplace_back(worker, t);
                for (auto& th : threads) th.join();
            }
            int rc = err.load(std::memory_order_relaxed);
            if (rc != 0) return rc;
        } else {
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

        std::fill(s.begin(), s.end(), 0.0);
        bool last_sw = (sw == sweeps - 1);
        if (last_sw) std::fill(mask, mask + T, 0);

        for (int64_t r = 0; r < n_rows; ++r) {
            int64_t need_orig = need_row[r];
            if (need_orig <= 0) continue;
            int64_t need = need_orig;
            int64_t len = row_size[r];
            int64_t base = row_start[r];
            double* sb = &s[row_trip[r] * rank];
            if (need == len) {
                for (int64_t p = 0; p < len; ++p) {
                    int64_t idx = base + p;
                    const double* po = &phi_ord[idx * rank];
                    for (int64_t d = 0; d < rank; ++d) sb[d] += po[d];
                    if (last_sw) mask[slot_at[idx]] = 1;
                }
            } else {
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
                    int64_t idx = base + p;
                    if (p + 1 == need) {
                        for (int64_t q = 0; q <= p; ++q) {
                            int64_t iq = base + q;
                            const double* po = &phi_ord[iq * rank];
                            for (int64_t d = 0; d < rank; ++d) sb[d] += po[d];
                            if (last_sw) mask[slot_at[iq]] = 1;
                        }
                        need = 0;
                        break;
                    }
                    double we = We[idx];
                    double num = we * E_row[p * stride + (need - 1)];
                    double den = E_row[(p + 1) * stride + need];
                    double prob = (den > 0.0) ? std::min(1.0, num / den) : 0.0;
                    if (rng.uniform() < prob) {
                        const double* po = &phi_ord[idx * rank];
                        for (int64_t d = 0; d < rank; ++d) sb[d] += po[d];
                        if (last_sw) mask[slot_at[idx]] = 1;
                        need -= 1;
                    }
                }
                if (need != 0) return -2;
            }
            if (psi_cat && need_orig >= 2) {
                double mult = static_cast<double>(need_orig - 1);
                const double* pc = &psi_cat[row_cat[r] * rank];
                for (int64_t d = 0; d < rank; ++d) sb[d] -= mult * pc[d];
            }
        }
    }
    return 0;
}

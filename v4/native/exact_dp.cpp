// Exact dynamic programs of the basket normaliser (PyTorch C++ extension, CPU, float64).
//
// Two kernels, each with a hand-written backward pass:
//
//   esp_blocked_log_*   the elementary symmetric polynomials e_0..e_R of the product weights
//   esp_ragged_log_*    of one (trip, category) row: e_k = total weight of all k-product
//                       subsets of the row.  Forward is the subtraction-free recursion
//                       e_k <- e_k + w e_{k-1}.
//   log_degree_*        the product over a trip's categories of their degree polynomials,
//                       truncated at nmax: coefficient n is the total weight of all baskets
//                       of size n.
//
// Both take and return log coefficients.  The backward passes propagate
// h_k = e_k * dL/de_k, so every multiplier is a share in [0, 1] and cannot overflow.
#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

void check_inputs(const torch::Tensor& p, const torch::Tensor& degree,
                  int64_t nmax) {
  TORCH_CHECK(p.device().is_cpu() && degree.device().is_cpu(),
              "degree-aware polynomial kernel is CPU-only");
  TORCH_CHECK(p.scalar_type() == torch::kFloat64,
              "degree-aware polynomial kernel requires float64 coefficients");
  TORCH_CHECK(degree.scalar_type() == torch::kInt64,
              "degree-aware polynomial kernel requires int64 degrees");
  TORCH_CHECK(p.is_contiguous() && degree.is_contiguous(),
              "degree-aware polynomial inputs must be contiguous");
  TORCH_CHECK(p.dim() == 4 && degree.dim() == 2,
              "expected coefficients[D,B,C,N+1] and degree[B,C]");
  TORCH_CHECK(degree.size(0) == p.size(1) && degree.size(1) == p.size(2),
              "degree shape does not match coefficient batch/category axes");
  TORCH_CHECK(nmax >= 0 && p.size(3) >= nmax + 1,
              "coefficient axis is shorter than nmax+1");
}

}  // namespace

std::vector<torch::Tensor> esp_ragged_log_forward(
    torch::Tensor log_weights, torch::Tensor row_start,
    torch::Tensor row_size, int64_t nmax) {
  TORCH_CHECK(log_weights.device().is_cpu() &&
              log_weights.scalar_type() == torch::kFloat64,
              "native ragged log-ESP requires CPU float64 log weights");
  log_weights = log_weights.contiguous();
  row_start = row_start.contiguous();
  row_size = row_size.contiguous();
  const int64_t n_rows = row_size.size(0);
  const int64_t width = nmax + 1;
  auto weights = torch::exp(log_weights);
  auto log_output = torch::full(
      {n_rows, width}, -std::numeric_limits<double>::infinity(),
      log_weights.options());
  const double* w_ptr = weights.data_ptr<double>();
  const int64_t* starts = row_start.data_ptr<int64_t>();
  const int64_t* sizes = row_size.data_ptr<int64_t>();
  double* out_ptr = log_output.data_ptr<double>();

  at::parallel_for(0, n_rows, 32, [&](int64_t begin, int64_t end) {
    std::vector<double> state(width, 0.0);
    for (int64_t r = begin; r < end; ++r) {
      const int64_t len = sizes[r];
      const int64_t base = starts[r];
      const int64_t deg_max = std::min<int64_t>(nmax, len);
      std::fill(state.begin(), state.begin() + deg_max + 1, 0.0);
      state[0] = 1.0;
      const double* row = w_ptr + base;
      for (int64_t item = 0; item < len; ++item) {
        const double value = row[item];
        const int64_t degree = std::min<int64_t>(nmax, item + 1);
        for (int64_t k = degree; k >= 1; --k) {
          state[k] += value * state[k - 1];
        }
      }
      double* out_row = out_ptr + r * width;
      out_row[0] = 0.0;
      for (int64_t k = 1; k <= deg_max; ++k) {
        if (state[k] > 0.0) {
          out_row[k] = std::log(state[k]);
        }
      }
    }
  });
  return {log_output, weights};
}

torch::Tensor esp_ragged_log_backward(
    torch::Tensor grad_log_output, torch::Tensor weights,
    torch::Tensor row_start, torch::Tensor row_size, int64_t nmax) {
  grad_log_output = grad_log_output.contiguous();
  weights = weights.contiguous();
  row_start = row_start.contiguous();
  row_size = row_size.contiguous();
  const int64_t n_rows = row_size.size(0);
  const int64_t width = nmax + 1;
  auto grad_log_weights = torch::zeros_like(weights);
  const double* w_ptr = weights.data_ptr<double>();
  const int64_t* starts = row_start.data_ptr<int64_t>();
  const int64_t* sizes = row_size.data_ptr<int64_t>();
  const double* output_grad = grad_log_output.data_ptr<double>();
  double* input_grad = grad_log_weights.data_ptr<double>();

  at::parallel_for(0, n_rows, 16, [&](int64_t begin, int64_t end) {
    std::vector<double> hnext(width), hprevious(width);
    std::vector<double> local;
    for (int64_t r = begin; r < end; ++r) {
      const int64_t len = sizes[r];
      if (len <= 0) continue;
      const int64_t base = starts[r];
      const int64_t deg_max = std::min<int64_t>(nmax, len);
      const int64_t w_loc = deg_max + 1;
      if (static_cast<int64_t>(local.size()) < (len + 1) * w_loc) {
        local.resize((len + 1) * w_loc);
      }
      std::fill(local.begin(), local.begin() + w_loc, 0.0);
      local[0] = 1.0;
      const double* row = w_ptr + base;
      for (int64_t item = 0; item < len; ++item) {
        const double* prev = local.data() + item * w_loc;
        double* next = local.data() + (item + 1) * w_loc;
        std::copy(prev, prev + w_loc, next);
        const double value = row[item];
        const int64_t degree = std::min<int64_t>(deg_max, item + 1);
        for (int64_t k = degree; k >= 1; --k) {
          next[k] += value * prev[k - 1];
        }
      }
      std::copy(output_grad + r * width,
                output_grad + r * width + w_loc, hnext.begin());
      double* row_grad = input_grad + base;
      for (int64_t item = len - 1; item >= 0; --item) {
        const double* previous = local.data() + item * w_loc;
        const double* next = local.data() + (item + 1) * w_loc;
        const double value = row[item];
        const int64_t degree = std::min<int64_t>(deg_max, item + 1);
        std::fill(hprevious.begin(), hprevious.begin() + degree + 1, 0.0);
        double value_gradient = 0.0;
        for (int64_t k = 0; k <= degree; ++k) {
          if (next[k] <= 0.0 || hnext[k] == 0.0) continue;
          const double inverse = 1.0 / next[k];
          const bool ordinary = std::isfinite(inverse);
          const double stay_fraction = ordinary
              ? previous[k] * inverse : previous[k] / next[k];
          hprevious[k] += hnext[k] * stay_fraction;
          if (k >= 1) {
            const double take = value * previous[k - 1];
            const double take_fraction = ordinary
                ? take * inverse : take / next[k];
            const double contribution = hnext[k] * take_fraction;
            hprevious[k - 1] += contribution;
            value_gradient += contribution;
          }
        }
        row_grad[item] = value_gradient;
        hnext.swap(hprevious);
      }
    }
  });
  return grad_log_weights;
}

std::vector<torch::Tensor> esp_forward(
    torch::Tensor weights, torch::Tensor lengths,
    int64_t nmax, int64_t block_size) {
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == torch::kFloat64,
              "native ESP requires CPU float64 weights");
  TORCH_CHECK(weights.dim() >= 2 && nmax >= 0 && block_size > 0,
              "native ESP expects [..., rows, items], nmax>=0, block_size>0");
  weights = weights.contiguous();
  lengths = lengths.contiguous();
  TORCH_CHECK(lengths.device().is_cpu() && lengths.scalar_type() == torch::kInt64 &&
              lengths.dim() == 1 && lengths.size(0) == weights.size(-2),
              "native ESP lengths must be int64 with one entry per row");
  const int64_t items = weights.size(-1);
  const int64_t rows = weights.size(-2);
  const int64_t sequences = weights.numel() / std::max<int64_t>(items, 1);
  const int64_t width = nmax + 1;
  const int64_t blocks = (items + block_size - 1) / block_size;
  std::vector<int64_t> out_shape(weights.sizes().begin(), weights.sizes().end() - 1);
  out_shape.push_back(width);
  auto output = torch::zeros(out_shape, weights.options());
  auto boundary = torch::zeros({sequences, blocks + 1, width}, weights.options());
  const double* input = weights.data_ptr<double>();
  const int64_t* row_lengths = lengths.data_ptr<int64_t>();
  double* result = output.data_ptr<double>();
  double* saved = boundary.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> state(width, 0.0);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::fill(state.begin(), state.end(), 0.0);
      state[0] = 1.0;
      double* first_boundary = saved + sequence * (blocks + 1) * width;
      std::copy(state.begin(), state.end(), first_boundary);
      const double* row = input + sequence * items;
      const int64_t item_count = std::clamp<int64_t>(
          row_lengths[sequence % rows], 0, items);
      int64_t next_boundary = 1;
      for (int64_t item = 0; item < item_count; ++item) {
        const double value = row[item];
        const int64_t degree = std::min<int64_t>(nmax, item + 1);
        for (int64_t k = degree; k >= 1; --k) {
          state[k] += value * state[k - 1];
        }
        if ((item + 1) % block_size == 0 || item + 1 == item_count) {
          std::copy(state.begin(), state.end(),
                    first_boundary + next_boundary * width);
          ++next_boundary;
        }
      }
      std::copy(state.begin(), state.end(), result + sequence * width);
    }
  });
  return {output, boundary};
}

std::vector<torch::Tensor> esp_blocked_log_forward(
    torch::Tensor log_weights, torch::Tensor lengths,
    int64_t nmax, int64_t block_size) {
  TORCH_CHECK(log_weights.device().is_cpu() &&
              log_weights.scalar_type() == torch::kFloat64,
              "native blocked log-ESP requires CPU float64 log weights");
  auto weights = torch::exp(log_weights.contiguous());
  auto packed = esp_forward(weights, lengths, nmax, block_size);
  auto output = packed[0];
  auto log_output = torch::where(
      output > 0, torch::log(output),
      torch::full_like(output, -std::numeric_limits<double>::infinity()));
  return {log_output, weights, packed[1]};
}

torch::Tensor esp_blocked_log_backward(
    torch::Tensor grad_log_output, torch::Tensor weights,
    torch::Tensor lengths, torch::Tensor boundary,
    int64_t nmax, int64_t block_size) {
  grad_log_output = grad_log_output.contiguous();
  weights = weights.contiguous();
  lengths = lengths.contiguous();
  boundary = boundary.contiguous();
  TORCH_CHECK(weights.device().is_cpu() && weights.scalar_type() == torch::kFloat64,
              "native blocked log-ESP requires CPU float64 weights");
  const int64_t items = weights.size(-1);
  const int64_t rows = weights.size(-2);
  const int64_t sequences = weights.numel() / std::max<int64_t>(items, 1);
  const int64_t width = nmax + 1;
  const int64_t blocks = (items + block_size - 1) / block_size;
  TORCH_CHECK(grad_log_output.numel() == sequences * width,
              "native blocked log-ESP output-gradient shape mismatch");
  TORCH_CHECK(boundary.sizes() ==
                  torch::IntArrayRef({sequences, blocks + 1, width}),
              "native blocked log-ESP boundary shape mismatch");
  auto grad_log_weights = torch::zeros_like(weights);
  const double* input = weights.data_ptr<double>();
  const int64_t* row_lengths = lengths.data_ptr<int64_t>();
  const double* output_grad = grad_log_output.data_ptr<double>();
  const double* saved = boundary.data_ptr<double>();
  double* input_grad = grad_log_weights.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> hnext(width), hprevious(width);
    std::vector<double> local((block_size + 1) * width);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::copy(output_grad + sequence * width,
                output_grad + (sequence + 1) * width, hnext.begin());
      const double* row = input + sequence * items;
      const double* first_boundary = saved + sequence * (blocks + 1) * width;
      double* row_grad = input_grad + sequence * items;
      const int64_t item_count = std::clamp<int64_t>(
          row_lengths[sequence % rows], 0, items);
      const int64_t used_blocks = (item_count + block_size - 1) / block_size;
      for (int64_t block = used_blocks - 1; block >= 0; --block) {
        const int64_t start = block * block_size;
        const int64_t finish = std::min<int64_t>(item_count, start + block_size);
        const int64_t length = finish - start;
        std::copy(first_boundary + block * width,
                  first_boundary + (block + 1) * width, local.begin());
        for (int64_t offset = 0; offset < length; ++offset) {
          double* previous = local.data() + offset * width;
          double* next = local.data() + (offset + 1) * width;
          std::copy(previous, previous + width, next);
          const double value = row[start + offset];
          const int64_t degree = std::min<int64_t>(nmax, start + offset + 1);
          for (int64_t k = degree; k >= 1; --k) {
            next[k] += value * previous[k - 1];
          }
        }
        for (int64_t offset = length - 1; offset >= 0; --offset) {
          const int64_t item = start + offset;
          const double* previous = local.data() + offset * width;
          const double* next = local.data() + (offset + 1) * width;
          const double value = row[item];
          const int64_t degree = std::min<int64_t>(nmax, item + 1);
          std::fill(hprevious.begin(), hprevious.end(), 0.0);
          double value_gradient = 0.0;
          for (int64_t k = 0; k <= degree; ++k) {
            if (next[k] <= 0.0 || hnext[k] == 0.0) continue;
            const double inverse = 1.0 / next[k];
            const bool ordinary = std::isfinite(inverse);
            const double stay_fraction = ordinary
                ? previous[k] * inverse : previous[k] / next[k];
            hprevious[k] += hnext[k] * stay_fraction;
            if (k >= 1) {
              const double take = value * previous[k - 1];
              const double take_fraction = ordinary
                  ? take * inverse : take / next[k];
              const double contribution = hnext[k] * take_fraction;
              hprevious[k - 1] += contribution;
              value_gradient += contribution;
            }
          }
          row_grad[item] = value_gradient;
          hnext.swap(hprevious);
        }
      }
    }
  });
  return grad_log_weights;
}

std::vector<torch::Tensor> log_degree_product_forward(
    torch::Tensor log_coefficients, torch::Tensor degree, int64_t nmax) {
  log_coefficients = log_coefficients.contiguous();
  degree = degree.contiguous();
  check_inputs(log_coefficients, degree, nmax);
  const int64_t draws = log_coefficients.size(0);
  const int64_t batch = log_coefficients.size(1);
  const int64_t categories = log_coefficients.size(2);
  const int64_t stride = log_coefficients.size(3);
  const int64_t width = nmax + 1;
  const int64_t sequences = draws * batch;
  auto normalized = torch::zeros_like(log_coefficients);
  auto prefix = torch::zeros({draws, batch, categories + 1, width},
                             log_coefficients.options());
  auto log_output = torch::full({draws, batch, width},
      -std::numeric_limits<double>::infinity(), log_coefficients.options());
  const double* input = log_coefficients.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  double* q_all = normalized.data_ptr<double>();
  double* states = prefix.data_ptr<double>();
  double* result = log_output.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      const int64_t b = sequence % batch;
      double* first = states + sequence * (categories + 1) * width;
      first[0] = 1.0;
      double log_scale = 0.0;
      int64_t deg_cur = 0;
      for (int64_t c = 0; c < categories; ++c) {
        const int64_t d = std::clamp<int64_t>(degrees[b * categories + c], 0, nmax);
        const double* lp = input + (sequence * categories + c) * stride;
        double* q = q_all + (sequence * categories + c) * stride;
        double maximum = -std::numeric_limits<double>::infinity();
        for (int64_t k = 0; k <= d; ++k) maximum = std::max(maximum, lp[k]);
        TORCH_CHECK(std::isfinite(maximum), "category polynomial has no finite coefficient");
        for (int64_t k = 0; k <= d; ++k) q[k] = std::exp(lp[k] - maximum);
        log_scale += maximum;
        double* previous = first + c * width;
        double* next = previous + width;
        int64_t next_deg = std::min<int64_t>(nmax, deg_cur + d);
        for (int64_t n = 0; n <= next_deg; ++n) {
          double value = 0.0;
          const int64_t k_lo = std::max<int64_t>(0, n - deg_cur);
          const int64_t upper = std::min<int64_t>(d, n);
          for (int64_t k = k_lo; k <= upper; ++k) value += previous[n - k] * q[k];
          next[n] = value;
        }
        deg_cur = next_deg;
      }
      const double* final_state = first + categories * width;
      double* out = result + sequence * width;
      for (int64_t n = 0; n <= deg_cur; ++n) {
        if (final_state[n] > 0.0) out[n] = std::log(final_state[n]) + log_scale;
      }
    }
  });
  return {log_output, normalized, prefix};
}

torch::Tensor log_degree_product_backward(
    torch::Tensor grad_log_output, torch::Tensor normalized,
    torch::Tensor degree, torch::Tensor prefix, int64_t nmax) {
  grad_log_output = grad_log_output.contiguous();
  normalized = normalized.contiguous();
  degree = degree.contiguous();
  prefix = prefix.contiguous();
  check_inputs(normalized, degree, nmax);
  const int64_t draws = normalized.size(0);
  const int64_t batch = normalized.size(1);
  const int64_t categories = normalized.size(2);
  const int64_t stride = normalized.size(3);
  const int64_t width = nmax + 1;
  const int64_t sequences = draws * batch;
  auto grad_log_coefficients = torch::zeros_like(normalized);
  const double* q_all = normalized.data_ptr<double>();
  const int64_t* degrees = degree.data_ptr<int64_t>();
  const double* states = prefix.data_ptr<double>();
  const double* output_grad = grad_log_output.data_ptr<double>();
  double* input_grad = grad_log_coefficients.data_ptr<double>();

  at::parallel_for(0, sequences, 1, [&](int64_t begin, int64_t end) {
    std::vector<double> hnext(width), hprevious(width);
    for (int64_t sequence = begin; sequence < end; ++sequence) {
      std::copy(output_grad + sequence * width,
                output_grad + (sequence + 1) * width, hnext.begin());
      const int64_t b = sequence % batch;
      const double* first = states + sequence * (categories + 1) * width;
      for (int64_t c = categories - 1; c >= 0; --c) {
        std::fill(hprevious.begin(), hprevious.end(), 0.0);
        const int64_t d = std::clamp<int64_t>(degrees[b * categories + c], 0, nmax);
        const double* previous = first + c * width;
        const double* next = previous + width;
        const double* q = q_all + (sequence * categories + c) * stride;
        double* hq = input_grad + (sequence * categories + c) * stride;
        for (int64_t n = 0; n <= nmax; ++n) {
          if (next[n] <= 0.0 || hnext[n] == 0.0) continue;
          const double inverse = 1.0 / next[n];
          const bool ordinary = std::isfinite(inverse);
          const int64_t upper = std::min<int64_t>(d, n);
          for (int64_t k = 0; k <= upper; ++k) {
            const int64_t m = n - k;
            const double term = previous[m] * q[k];
            const double fraction = ordinary ? term * inverse : term / next[n];
            const double contribution = hnext[n] * fraction;
            hprevious[m] += contribution;
            hq[k] += contribution;
          }
        }
        hnext.swap(hprevious);
      }
    }
  });
  return grad_log_coefficients;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("esp_ragged_log_forward", &esp_ragged_log_forward,
             "Ragged elementary-symmetric-polynomial log forward");
  module.def("esp_ragged_log_backward", &esp_ragged_log_backward,
             "Ragged bounded log-adjoint for elementary symmetric polynomials");
  module.def("esp_blocked_log_forward", &esp_blocked_log_forward,
             "Blocked elementary-symmetric-polynomial log forward");
  module.def("esp_blocked_log_backward", &esp_blocked_log_backward,
             "Blocked bounded log-adjoint for elementary symmetric polynomials");
  module.def("log_degree_forward", &log_degree_product_forward,
             "Degree-aware category product in log-coefficient coordinates");
  module.def("log_degree_backward", &log_degree_product_backward,
             "Bounded logarithmic adjoint for the category product");
}

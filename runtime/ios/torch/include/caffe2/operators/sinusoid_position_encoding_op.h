#ifndef CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_
#define CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_

#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#endif // _MSC_VER
#include <cmath>

#include "caffe2/core/operator.h"

#include "Eigen/Core"
#include "caffe2/utils/eigen_utils.h"

namespace caffe2 {

template <class Context>
class SinusoidPositionEncodingOp : public Operator<Context> {
 public:
  template <class... Args>
  explicit SinusoidPositionEncodingOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        embedding_size_(
            this->template GetSingleArgument<int>("embedding_size", 100)),
        alpha_(this->template GetSingleArgument<float>("alpha", 10000)),
        amplitude_(this->template GetSingleArgument<float>("amplitude", 1)) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, this->template Input<Tensor>(0, CPU));
  }

  template <typename Index>
  bool DoRunWithType() {
    auto& positions = Input(0);

    CAFFE_ENFORCE_EQ(positions.dim(), 2, "POSITIONS should be a 2-D tensor");

    auto shape = positions.sizes().vec();
    shape.push_back(embedding_size_);
    auto* output = Output(0, shape, at::dtype<float>());

    int M = shape[0];
    int K = shape[1];
    const Index* idxs = positions.template data<Index>();
    float* out = output->template mutable_data<float>();

    float log_alpha = std::log(alpha_);
    float max_alpha_pow =
        ((float)embedding_size_ - 1.0f) / (float)embedding_size_;

    for (int i = 0; i < M; ++i) {
      float pos = (float)idxs[i * K];

      // Compute the embedding for position i, example 0 first
      float* row = &out[i * K * embedding_size_];
      Eigen::Map<Eigen::VectorXf> row_map(row, embedding_size_, 1);
      auto row_array = row_map.array();

      float log_pos = std::log(pos);
      row_array.setLinSpaced(
          embedding_size_, log_pos, log_pos - log_alpha * max_alpha_pow);
      row_array = row_array.exp().eval();
      // row_array[k] == pos / alpha^(k / embedding_size)

      // Phase shift so that alternating elements are cosines
      for (int k = 1; k < embedding_size_; k += 2) {
        row[k] += (float)M_PI_2;
      }
      row_array = amplitude_ * row_array.sin().eval();

      // Copy the embedding to position i in the other examples
      for (int j = 1; j < K; ++j) {
        int base = i * K * embedding_size_;
        std::copy(
            &out[base],
            &out[base + embedding_size_],
            &out[base + j * embedding_size_]);
      }
    }
    return true;
  }

 protected:
  int embedding_size_;
  float alpha_;
  float amplitude_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SINUSOID_POSITION_ENCODING_OP_H_

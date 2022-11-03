
#ifndef CAFFE2_OPERATORS_LENGTHS_REDUCER_ROWWISE_8bits_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_REDUCER_ROWWISE_8bits_OP_H_
// SparseLengthsSum8bits

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/perfkernels/embedding_lookup.h"
#include "caffe2/utils/eigen_utils.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
const float kEqualityThreshold = 1e-10f;
}

template <
    class Context,
    bool USE_WEIGHTS = 0,
    bool USE_MEAN = 0,
    class OutDataT = float>
class SparseLengths8BitsRowwiseOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SparseLengths8BitsRowwiseOp);

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    auto& dataInput = Input(DATA);
    auto& lengthsInput = Input(LENGTHS);

    auto* scale_bias = Input(SCALE_BIAS).template data<float>();
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t outputSize = lengthsInput.size(0);

    auto& indicesInput = Input(INDICES);
    CAFFE_ENFORCE_EQ(2, Input(SCALE_BIAS).dim(), "scale_bias has to be matrix");
    CAFFE_ENFORCE_EQ(
        dataInput.size(0),
        Input(SCALE_BIAS).size(0),
        "scale_bias must have the same first dim as data");
    CAFFE_ENFORCE_EQ(
        2,
        Input(SCALE_BIAS).size(1),
        "the second dim of scale_bias has to be equal to 2");
    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    const IndexType* indices = indicesInput.template data<IndexType>();
    int64_t dataToReduceSize = indicesInput.size(0);

    const int* lengths = lengthsInput.template data<int>();
    vector<int64_t> shape = dataInput.sizes().vec();
    shape[0] = outputSize;
    auto* output = Output(0, shape, at::dtype<OutDataT>());
    const float* w = nullptr;
    if (USE_WEIGHTS) {
      w = Input(WEIGHTS).template data<float>();
    }
    int64_t in_block_size = dataInput.size_from_dim(1);
    OutDataT* out = output->template mutable_data<OutDataT>();
    const uint8_t* input_data = dataInput.template data<uint8_t>();

    // delegate work to perfkernel that branches based on architecture
    const int64_t indices_size = indicesInput.numel();
    const int64_t N = dataInput.size(0);
    EmbeddingLookup(
        in_block_size,
        outputSize,
        indices_size,
        N, // embedding table length
        input_data,
        indices,
        lengths,
        w,
        scale_bias,
        USE_MEAN,
        out);

    return true;
  }

  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + USE_WEIGHTS,
    LENGTHS = 2 + USE_WEIGHTS,
    SCALE_BIAS = 3 + USE_WEIGHTS
  };
};

template <class Context>
class FloatToRowwiseQuantized8BitsOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FloatToRowwiseQuantized8BitsOp);
  bool RunOnDevice() override {
    auto& input = Input(DATA_FLOAT);

    auto* input_data = input.template data<float>();
    auto* output = Output(DATA_UINT8, input.sizes(), at::dtype<uint8_t>());
    vector<int64_t> scale_bias_dims = {input.size(0), 2};
    auto* scale_bias = Output(SCALE_BIAS, scale_bias_dims, at::dtype<float>());
    auto* output_data = output->template mutable_data<uint8_t>();
    float* scale_bias_data = scale_bias->template mutable_data<float>();
    size_t n_blocks = input.size(0);
    size_t block_size = input.size_from_dim(1);
    for (size_t i = 0; i < n_blocks; ++i) {
      ConstEigenVectorArrayMap<float> input_row(
          input_data + i * block_size, block_size);
      EigenVectorArrayMap<uint8_t> output_row(
          output_data + i * block_size, block_size);
      auto min_element = input_row.minCoeff();
      auto max_element = input_row.maxCoeff();
      if (max_element - min_element < kEqualityThreshold) {
        scale_bias_data[2 * i] = 1.0f;
        scale_bias_data[2 * i + 1] = min_element;
        memset(output_data + i * block_size, 0, block_size);
      } else {
        scale_bias_data[2 * i] = (max_element - min_element) / 255.0f;
        scale_bias_data[2 * i + 1] = min_element;
        const float inv_scale = 1.0f / scale_bias_data[2 * i];
        output_row = ((input_row - scale_bias_data[2 * i + 1]) * inv_scale)
                         .round()
                         .template cast<uint8_t>();
      }
    }
    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_UINT8, SCALE_BIAS);
};

template <class Context>
class Rowwise8BitQuantizedToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(Rowwise8BitQuantizedToFloatOp);
  bool RunOnDevice() override {
    auto& input = Input(DATA_UINT8);
    auto& scale_bias = Input(SCALE_BIAS);

    CAFFE_ENFORCE_EQ(2, scale_bias.dim(), "scale_bias has to be matrix");
    CAFFE_ENFORCE_EQ(
        input.size(0),
        scale_bias.size(0),
        "scale_bias must have the same first dim as data");
    CAFFE_ENFORCE_EQ(
        2,
        scale_bias.size(1),
        "the second dim of scale_bias has to be equal to 2");
    auto* output = Output(DATA_FLOAT, input.sizes(), at::dtype<float>());
    auto* input_data = input.template data<uint8_t>();
    auto* scale_bias_data = scale_bias.template data<float>();

    auto* output_data = output->template mutable_data<float>();
    size_t block_size = input.size_from_dim(1);
    size_t n_blocks = input.size(0);

    for (size_t i = 0; i < n_blocks; ++i) {
      ConstEigenVectorArrayMap<uint8_t> input_row(
          input_data + i * block_size, block_size);
      EigenVectorArrayMap<float> output_row(
          output_data + i * block_size, block_size);
      output_row = input_row.template cast<float>() * scale_bias_data[2 * i] +
          scale_bias_data[2 * i + 1];
    }
    return true;
  }

 private:
  INPUT_TAGS(DATA_UINT8, SCALE_BIAS);
  OUTPUT_TAGS(DATA_FLOAT);
};
} // namespace caffe2
#endif // CAFFE2_OPERATORS_LENGTHS_REDUCER_ROWWISE_8bits_H_

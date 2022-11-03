#ifndef CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_
#define CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_

#include <qnnpack.h>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor_int8.h"
#include "caffe2/operators/quantized/int8_utils.h"

namespace caffe2 {

namespace int8 {

class Int8SoftmaxOp final : public Operator<CPUContext> {
 public:
  explicit Int8SoftmaxOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws), ws_(ws) {}

  ~Int8SoftmaxOp() {
    if (this->qnnpackOperator_ != nullptr) {
      qnnp_delete_operator(this->qnnpackOperator_);
      this->qnnpackOperator_ = nullptr;
    }
  }

  bool RunOnDevice() override {
    const auto& X = Inputs()[0]->template Get<Int8TensorCPU>();
    auto* Y = Outputs()[0]->template GetMutable<Int8TensorCPU>();
    const int32_t Y_zero_point =
        this->template GetSingleArgument<int>("Y_zero_point", 0);
    const float Y_scale = this->template GetSingleArgument<float>("Y_scale", 1);
    CHECK_EQ(Y_zero_point, 0);
    CHECK_EQ(Y_scale, 1.0f / 256.0f);

    /*
     * Record quantization parameters for the input, because if the op is
     * in-place, we may overwrite these parameters later, when we set
     * quantization parameters for output tensor.
     */
    const uint8_t X_zero_point = X.zero_point;
    const float X_scale = X.scale;

    Y->scale = Y_scale;
    Y->zero_point = Y_zero_point;
    Y->t.ResizeLike(X.t);

    initQNNPACK();

    if (this->qnnpackOperator_ == nullptr) {
      const qnnp_status createStatus = qnnp_create_softargmax_nc_q8(
          X.t.numel() / X.t.size(0) /* channels */,
          X_scale,
          static_cast<uint8_t>(Y_zero_point),
          Y_scale,
          0 /* flags */,
          &qnnpackOperator_);
      CAFFE_ENFORCE(
          createStatus == qnnp_status_success,
          "failed to create QNNPACK SoftArgMax operator");
      CAFFE_ENFORCE(this->qnnpackOperator_ != nullptr);
    }

    const qnnp_status setupStatus = qnnp_setup_softargmax_nc_q8(
        this->qnnpackOperator_,
        X.t.size(0) /* batch size */,
        X.t.template data<uint8_t>(),
        X.t.numel() / X.t.size(0) /* X stride */,
        Y->t.template mutable_data<uint8_t>(),
        X.t.numel() / X.t.size(0) /* Y stride */);
    CAFFE_ENFORCE(
        setupStatus == qnnp_status_success,
        "failed to setup QNNPACK SoftArgMax operator");

#if defined(FBCODE_CAFFE2) || !defined(USE_INTERNAL_PTHREADPOOL_IMPL)
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, nullptr /* thread pool */);
#else
    pthreadpool_t threadpool =
        reinterpret_cast<pthreadpool_t>(ws_->GetThreadPool());
    const qnnp_status runStatus =
        qnnp_run_operator(this->qnnpackOperator_, threadpool);
#endif
    CAFFE_ENFORCE(
        runStatus == qnnp_status_success,
        "failed to run QNNPACK SoftArgMax operator");

    return true;
  }

 private:
  Workspace* ws_;
  // QNNPACK SoftArgMax operator
  qnnp_operator_t qnnpackOperator_{nullptr};
};

} // namespace int8

} // namespace caffe2

#endif // CAFFE2_OPERATORS_INT8_SOFTMAX_OP_H_

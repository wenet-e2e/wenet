#ifndef CAFFE2_OPERATORS_DROPOUT_OP_H_
#define CAFFE2_OPERATORS_DROPOUT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class DropoutOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit DropoutOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
  }

  bool RunOnDevice() override;

 protected:
  float ratio_;
  bool is_test_;
  // Input: X; Output: Y, mask.
};

template <typename T, class Context>
class DropoutGradientOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit DropoutGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        ratio_(this->template GetSingleArgument<float>("ratio", 0.5)),
        is_test_(
            this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)) {
    CAFFE_ENFORCE_GE(ratio_, 0);
    CAFFE_ENFORCE_LT(ratio_, 1);
  }

  bool RunOnDevice() override;

 protected:
  float ratio_;
  bool is_test_;
  // Input: dY, mask; Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_DROPOUT_OP_H_

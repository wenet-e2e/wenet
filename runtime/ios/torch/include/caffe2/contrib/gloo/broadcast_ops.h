#pragma once

#include <algorithm>

#include "caffe2/contrib/gloo/common.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/types.h"

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>

namespace caffe2 {
namespace gloo {

template <class Context>
class BroadcastOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  BroadcastOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        root_(OperatorBase::template GetSingleArgument<int>("root", 0)),
        ws_(ws),
        status_blob_(
            OperatorBase::GetSingleArgument<std::string>("status_blob", "")) {
    if (status_blob_ != "") {
      ws_->CreateBlob(status_blob_);
    }
  }

  virtual ~BroadcastOp() {}

  bool RunOnDevice() override {
    std::call_once(once_, [&] { initialize(); });

    // If any parameter has changed in between runs, the initialized
    // algorithm is invalid and cannot be used.
    update(current_);
    CAFFE_ENFORCE(current_ == init_, "Inputs/outputs have changed");

    try {
      algorithm_->run();
    } catch (::gloo::IoException& ioe) {
      LOG(ERROR) << "Caught gloo IO exception: " << ioe.what();
      if (status_blob_ != "") {
        signalFailure(ws_->GetBlob(status_blob_), ioe);
        return false;
      } else {
        throw;
      }
    }
    return true;
  }

 protected:
  void initialize() {
    // Store which inputs/outputs this instance initialized with
    update(init_);

    // Verify inputs == outputs
    CAFFE_ENFORCE_EQ(init_.inputs.size(), init_.outputs.size());
    for (auto i = 0; i < init_.inputs.size(); i++) {
      CAFFE_ENFORCE_EQ(init_.inputs[i], init_.outputs[i]);
    }

    // Verify tensors all have same size
    size_t size = Input(1).numel();
    for (auto i = 2; i < InputSize(); i++) {
      CAFFE_ENFORCE_EQ(Input(i).numel(), size);
    }

    // Verify tensors all have same size
    TypeMeta meta = Input(1).dtype();
    for (auto i = 2; i < InputSize(); i++) {
      CAFFE_ENFORCE(Input(i).dtype() == meta);
    }

    // Finally initialize the algorithm
    initializeAlgorithm();
  }

  void initializeAlgorithm();

  const int root_;
  std::once_flag once_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;

  // Captures the parameters passed to Gloo when first initialized.
  // An instance is updated every time this op runs and is compared
  // to the reference instance for equality. If any parameter has
  // changed from run to run, the initialized algorithm is invalid.
  void update(GlooParameters& params) {
    params.context = OperatorBase::Input<std::shared_ptr<::gloo::Context>>(0);
    params.inputs.resize(InputSize() - 1);
    params.outputs.resize(OutputSize());
    for (auto i = 0; i < params.inputs.size(); i++) {
      params.inputs[i] = Input(i + 1).raw_data();
      params.outputs[i] = Output(i)->raw_mutable_data();
    }
    params.size = Output(0)->numel();
    params.meta = Output(0)->dtype();
  }

  GlooParameters init_;
  GlooParameters current_;
  Workspace* ws_;
  std::string status_blob_;
};

} // namespace gloo
} // namespace caffe2

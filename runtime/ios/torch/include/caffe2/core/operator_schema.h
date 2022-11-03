#ifndef CAFFE2_CORE_OPERATOR_SCHEMA_H_
#define CAFFE2_CORE_OPERATOR_SCHEMA_H_

#include <climits>
#include <functional>
#include <initializer_list>
#include <ostream>
#include <set>
#include <vector>
#include <unordered_map>

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/filler.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

// A const value returned by OpSchema::CalculateOutput() if the number of
// output cannot be determined.
constexpr int kCannotComputeNumOutputs = -1;

/**
 * @brief A class to record the schema of an op.
 *
 * OpSchema records the common interface of an op specified by its name. This
 * is optional for each operator implemented in Caffe2 but is strongly
 * recommended.
 *
 * To register an OpSchema, one can use the macro OPERATOR_SCHEMA(name) and
 * then append the various functions in the class. For example, for an op
 * that takes in two inputs, one output, and the first input and output
 * could be in-place, can be written as
 *
 *     OPERATOR_SCHEMA(name)
 *         .NumInputs(2).NumOutputs(1).AllowInplace({{0, 0}});
 */
class TORCH_API OpSchema {
 public:
  OpSchema() : OpSchema("unknown", "unknown", 0) {}
  OpSchema(const string& type, const string& file, const int line);

  /**
   * @brief Returns the file that the op schema is registered from.
   */
  inline const string& file() const {
    return file_;
  }

  /**
   * @brief Returns the line in file that the op schema is registered from.
   */
  inline int line() const {
    return line_;
  }

  /**
   * @brief Returns the docstring of the op schema.
   */
  inline const char* doc() const {
    return doc_.empty() ? nullptr : doc_.c_str();
  }

  /**
   * @brief Verifies if an operator definition protobuf matches the pattern
   * specified in the schema.
   */
  bool Verify(const OperatorDef& def) const;

  // Functions to set the property of the operator schemas.
  // Sets the number of inputs, either a fixed number or a min and a max.

  /**
   * @brief A single input.
   */
  OpSchema& NumInputs(int n);
  /**
   * @brief Input could be in range [min, max], inclusive.
   */
  OpSchema& NumInputs(int min, int max);
  /**
   * @brief Input could be one of the values specified in allowed_input_nums.
   */
  OpSchema& NumInputs(set<int> allowed_input_nums);
  /**
   * @brief Input is checked with a specified function.
   */
  OpSchema& NumInputs(std::function<bool(int)> func);

  // Sets the number of outputs, either a fixed number, a min and a max,
  // or a function that takes in the input number and produces an output
  // number. Use only one function in the set below.
  /**
   * @brief A single output.
   */
  OpSchema& NumOutputs(int n);
  /**
   * @brief Output could be in range [min, max], inclusive.
   */
  OpSchema& NumOutputs(int min, int max);
  /**
   * @brief Output could be one of the values specified in allowed_output_nums.
   */
  OpSchema& NumOutputs(set<int> allowed_output_nums);
  /**
   * @brief Output is checked with a specified function.
   */
  OpSchema& NumOutputs(std::function<bool(int)> func);

  /**
   * @brief Relationship between inputs and outputs is checked with a specified
   * function.
   */
  OpSchema& NumInputsOutputs(std::function<bool(int, int)> func);

  // Set the function that can calculate the number of output based on the
  // number of input. Use only one function in the set below.
  /**
   * @brief Set the output calculator to a user-defined function.
   */
  OpSchema& OutputCalculator(std::function<int(int)> calc);
  /**
   * @brief Set the number of outputs to be the same as the number of inputs.
   */
  OpSchema& SameNumberOfOutput();

  // Sets the rule to allow optional in-place operation.
  OpSchema& AllowInplace(std::function<bool(int, int)> inplace);
  OpSchema& AllowInplace(set<std::pair<int, int>> inplace);
  OpSchema& AllowOneToOneInplace();
  // Sets the rule to enforce in-place operation.
  OpSchema& EnforceInplace(std::function<bool(int, int)> inplace);
  OpSchema& EnforceInplace(set<std::pair<int, int>> inplace);
  OpSchema& EnforceOneToOneInplace();

  // Functions to deal with type and shape inference. Basically, this registers
  // a function that takes in an OperatorDef and a series of input type and
  // shape specified by TensorProto objects (whose data fields are empty), and
  // produces a series of output type and shape.
  typedef std::function<
      vector<TensorShape>(const OperatorDef&, const vector<TensorShape>&)>
      TensorInferenceFunctionType;

  /**
   * @brief Sets the tensor inference function, which is a std::function object
   * defined in operator_schema.h.
   */
  OpSchema& TensorInferenceFunction(TensorInferenceFunctionType function);

  /**
   * A wrapper that makes an infer tensor function to return unknown
   * shape for all outputs if any one of the inputs has unknown shape
   */

  static TensorInferenceFunctionType NeedsAllInputShapes(
      TensorInferenceFunctionType f);

  /**
   * @brief Sets the corresponding onnx schema name
   */
  OpSchema& InheritOnnxSchema(const std::string& onnx_schema_name);

  /**
   * @brief Shortcut to InheritOnnxSchema(type_)
   */
  OpSchema& InheritOnnxSchema() {
    return InheritOnnxSchema(type_);
  }

  /**
   * @brief Sets the tensor inference function to produce the same output as
   * the input.
   */
  OpSchema& IdenticalTypeAndShape();
  OpSchema& IdenticalTypeAndShapeOfInput(int idx);
  OpSchema& IdenticalTypeAndShapeOfInputDim(int idx, int dim);
  OpSchema& IdenticalTypeAndShapeOfMultipleInputs(const vector<int>& indices);
  OpSchema& ScalarType(::caffe2::TensorProto_DataType dt);

  /**
   * @brief A function to allow one to infer the type and shape from the op
   * schema.
   */
  inline vector<TensorShape> InferTensor(
      const OperatorDef& def,
      const vector<TensorShape>& input_type_shape) const {
    CAFFE_ENFORCE(
        Verify(def),
        "(InferTensor) Operator def did not pass schema checking: ",
        ProtoDebugString(def));
    return tensor_inference_function_(def, input_type_shape);
  }

  /*
   * @brief A struct to store various cost information about
   * an operator such as FLOPs, total memory use and parameters.
   */
  struct Cost {
    uint64_t flops{0}; // Floating point operations.
    uint64_t bytes_read{0}; // Total memory read.
    uint64_t bytes_written{0}; // Total memory written.
    uint64_t params_bytes{0}; // Memory read for parameters.
  };
  /**
   * @brief Registers a function that takes in an OperatorDef
   * and a series of input shapes and returns the total "cost"
   * required to run the operator via struct by value.
   */
  typedef std::function<
      struct Cost(const OperatorDef&, const vector<TensorShape>&)>
      CostInferenceFunctionType;

  /**
   * @brief Register the Cost inference function.
   */
  OpSchema& CostInferenceFunction(CostInferenceFunctionType function);

#if 0 // def _MSC_VER
  /**
   * @brief Register the Cost inference function via a pointer.
   */
  template <typename T,
            typename = std::enable_if<
                std::is_same<CostInferenceFunctionType&&, T>:value
                >:type>
  inline OpSchema& CostInferenceFunction(T func) {
    // Note: This is here in order to resolve an MSVC compiler issue: it
    // does not automatically convert a function pointer to a std::function,
    // and needs an explicit conversion.
    return CostInferenceFunction(CostInferenceFunctionType(func));
  }
#endif // _MSC_VER

  bool HasCostInferenceFunction() const {
    return !!cost_inference_function_;
  }

  inline struct Cost InferCost(
      const OperatorDef& def,
      const vector<TensorShape>& input_tensor_shape) const {
    CAFFE_ENFORCE(
        cost_inference_function_, "Cost inference function not defined.");
    return (*cost_inference_function_)(def, input_tensor_shape);
  }

  // Functions to do documentation for the operator schema.
  OpSchema& SetDoc(const string& doc);

  struct Argument {
    Argument(const char* name, const char* description, bool required)
        : name_{name}, description_{description}, required_{required} {}

    const char* name() const {
      return name_;
    }

    const char* description() const {
      return description_;
    }

    bool is_required() const {
      return required_;
    }

   private:
    const char* name_;
    const char* description_;
    const bool required_;
  };

  OpSchema&
  Arg(const char* name, const char* description, bool required = false);

#define DECLARE_STANDARD_ARG(name, str)     \
  static const char* Arg_##name; \
  OpSchema& Arg##name(const char* description);

  DECLARE_STANDARD_ARG(IsTest, is_test)

#undef DECLARE_STANDARD_ARG

  OpSchema& Input(const int n, const char* name, const char* description);
  OpSchema& Output(const int n, const char* name, const char* description);
  // Calls the passed function with `this` as an argument. Useful for
  // adding docs for templated/macro ops.
  OpSchema& FillUsing(std::function<void(OpSchema&)> populator);

  // Remove from documentation
  OpSchema& Private();

  // This op can pass data across devices
  OpSchema& InputsCanCrossDevices();

  /**
   * @brief A function to allow one to get the number of outputs based on the
   * number of inputs, if this schema supports it.
   */
  int CalculateOutput(int num_input) const;

  const std::string& onnx_schema() const {
    return onnx_schema_;
  }

  int min_input() const {
    return min_input_;
  }

  int max_input() const {
    return max_input_;
  }

  int min_output() const {
    return min_output_;
  }

  int max_output() const {
    return max_output_;
  }

  bool num_inputs_allowed(int x) const {
    return num_inputs_allowed_(x);
  }

  bool num_outputs_allowed(int x) const {
    return num_outputs_allowed_(x);
  }

  bool num_inputs_outputs_allowed(int x, int y) const {
    return num_inputs_outputs_allowed_(x, y);
  }

  int inf() const {
    return std::numeric_limits<int>::max();
  }

  bool inplace_enforced(int x, int y) const {
    return inplace_enforced_(x, y);
  }

  TORCH_API friend std::ostream& operator<<(std::ostream& out, const OpSchema& schema);

  const std::vector<Argument>& args() const {
    return args_;
  }

  const std::vector<std::pair<const char*, const char*>>& input_desc() const {
    return input_desc_;
  }
  const std::vector<std::pair<const char*, const char*>>& output_desc() const {
    return output_desc_;
  }
  bool private_op() {
    return private_;
  }
  bool inputs_can_cross_devices() const {
    return inputs_can_cross_devices_;
  }

  /**
   * @brief Returns the required device location of inputs and outputs.
   */
  using DeviceInferenceFunctionType = std::function<
      std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>(
          const OperatorDef& def)>;

  OpSchema& DeviceInferenceFunction(DeviceInferenceFunctionType function);

  /**
   * @brief Infer required device location of an op's inputs and outputs
   */
  inline std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>
  InferDevice(const OperatorDef& def) const {
    return device_inference_function_(def);
  }

  // The helper is build sparse input with values, keys, weights and lengths;
  // e.g.:
  // values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
  // keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
  // weights = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  //            \_____/  \________/  \__/
  // lengths =    [3,        4,       2]
  OpSchema& WeightedValueKeyLengthInputFillers(
      size_t value_index,
      size_t key_index,
      size_t length_index,
      size_t weight_index);

  // The helper is build sparse input with values, keys, weights and lengths;
  // e.g.:
  // values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
  // keys    = [0, 1, 4, 0, 1, 2, 5, 1, 2]
  //            \_____/  \________/  \__/
  // lengths =    [3,        4,       2]
  OpSchema& ValueKeyLengthInputFillers(
      size_t value_index,
      size_t key_index,
      size_t length_index);

  // The helper is build sparse input with values and lengths; e.g.:
  // values  = [1, 2, 3, 2, 4, 6, 7, 3, 6]
  //            \_____/  \________/  \__/
  // lengths =    [3,        4,       2]
  OpSchema& ValueLengthInputFillers(size_t value_index, size_t length_index);

  OpSchema& DisallowInputFillers();

  std::vector<TensorFiller> InputFillers(
      const std::vector<std::vector<int64_t>>& shapes) const;

 private:
  std::vector<TensorFiller> SupplyDenseFillers(
      const std::vector<std::vector<int64_t>>& shapes);

 private:
  string type_;
  string file_;
  string doc_;
  string onnx_schema_;
  std::vector<Argument> args_{};
  std::vector<std::pair<const char*, const char*>> input_desc_{};
  std::vector<std::pair<const char*, const char*>> output_desc_{};
  int line_ = 0;
  int min_input_ = 0;
  int max_input_ = std::numeric_limits<int>::max();
  int min_output_ = 0;
  int max_output_ = std::numeric_limits<int>::max();
  bool private_ = false;
  bool inputs_can_cross_devices_ = false;
  std::function<bool(int)> num_inputs_allowed_ = [](int) { return true; };
  std::function<bool(int)> num_outputs_allowed_ = [](int) { return true; };
  std::function<bool(int, int)> num_inputs_outputs_allowed_ = [](int, int) {
    return true;
  };
  std::function<int(int)> calculate_output_;
  // In default, any in-place operation is neither allowed nor enforced.
  std::function<bool(int, int)> inplace_allowed_ = [](int, int) {
    return false;
  };
  std::function<bool(int, int)> inplace_enforced_ = [](int, int) {
    return false;
  };
  TensorInferenceFunctionType tensor_inference_function_;
  std::unique_ptr<CostInferenceFunctionType> cost_inference_function_ = nullptr;
  DeviceInferenceFunctionType device_inference_function_;

  std::function<std::vector<TensorFiller>(
      const std::vector<std::vector<int64_t>>&)>
      filler_supplier_ =
          [this](const std::vector<std::vector<int64_t>>& shapes) {
            return SupplyDenseFillers(shapes);
          };
};

/**
 * @brief A registry to hold all the operator schemas.
 */
class TORCH_API OpSchemaRegistry {
 public:
  static OpSchema&
  NewSchema(const string& key, const string& file, const int line) {
    auto& m = map();
    auto it = m.find(key);
    if (it != m.end()) {
      const auto& schema = it->second;
      std::ios_base::Init init;
      std::cerr << "Trying to register schema with name " << key
                << " from file " << file << " line " << line
                << ", but it is already registered from file " << schema.file()
                << " line " << schema.line();
      abort();
    }
    m.emplace(key, OpSchema(key, file, line));
    return m[key];
  }

  static const OpSchema* Schema(const string& key) {
    auto& m = map();
    auto it = m.find(key);
    if (it != m.end()) {
      return &it->second;
    } else {
      return nullptr;
    }
  }

 private:
  // OpSchemaRegistry should not need to be instantiated.
  OpSchemaRegistry() = delete;

  /**
   * @brief Returns the underlying string to OpSchema map.
   *
   * You should not manually manipulate the map object returned. Instead, use
   * the macros defined such as OPERATOR_SCHEMA to register your operator
   * schema.
   *
   * We wrap it inside a function to avoid the static initialization order
   * fiasco.
   */
  static CaffeMap<string, OpSchema>& map();
};

// Helper function for creating simple tensorproto with dimension and type
template <typename T_I = int>
inline TensorShape CreateTensorShape(
    vector<T_I> dims,
    ::caffe2::TensorProto_DataType dt) {
  TensorShape ts;
  for (T_I d : dims) {
    ts.add_dims(d);
  }
  ts.set_data_type(dt);
  return ts;
}

// Helper function
inline vector<int64_t> GetDimsVector(const TensorShape& shape) {
  vector<int64_t> dims;
  for (auto d : shape.dims()) {
    dims.push_back(d);
  }
  return dims;
}

// Helper function
inline uint64_t nElemFromDim(const TensorShape& X, int dim = 0) {
  CAFFE_ENFORCE_GE(dim, 0, "Invalid maximum index specified");

  uint64_t nElem = 1;
  for (int i = dim; i < X.dims_size(); ++i) {
    nElem *= X.dims(i);
  }
  return nElem;
}

// Helper function
inline uint64_t nElemBetweenDim(const TensorShape& X, int start, int stop) {
  CAFFE_ENFORCE_GE(start, 0, "Invalid maximum index specified");
  CAFFE_ENFORCE_LE(stop, X.dims_size(), "Invalid maximum index specified");

  uint64_t nElem = 1;
  for (int i = start; i < stop; ++i) {
    nElem *= X.dims(i);
  }
  return nElem;
}

// Helper function for infer op inputs and outputs device information.
inline std::pair<std::vector<DeviceOption>, std::vector<DeviceOption>>
InferOpInputOutputDevice(const OperatorDef& op) {
  auto op_schema = OpSchemaRegistry::Schema(op.type());
  if (op_schema) {
    // op_schema found
    return op_schema->InferDevice(op);

  } else {
    // No schema for op.type registered
    auto temp_schema = OpSchema();
    return temp_schema.InferDevice(op);
  }
}

template <uint64_t OpsPerPoint>
OpSchema::Cost PointwiseCostInference(
    const OperatorDef& /* unused */,
    const vector<TensorShape>& inputs) {
  struct OpSchema::Cost c;
  const TensorShape X = inputs[0];
  uint64_t nElemX = nElemFromDim(X);
  uint64_t nElemRead = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    nElemRead += nElemFromDim(inputs[i]);
  }

  c.flops = nElemX * OpsPerPoint;
  c.bytes_read = nElemRead * sizeof(X.data_type());
  c.bytes_written = nElemX * sizeof(X.data_type());
  return c;
}

} // namespace caffe2

#ifndef CAFFE2_NO_OPERATOR_SCHEMA

#define OPERATOR_SCHEMA(name)                                       \
  C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; \
  static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     \
      &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

#else // CAFFE2_NO_OPERATOR_SCHEMA

#define OPERATOR_SCHEMA(name)                                       \
  C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; \
  static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     \
      1 ? nullptr : &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

#endif // CAFFE2_NO_OPERATOR_SCHEMA

#ifdef CAFFE2_NO_GRADIENT_OPS

#define GRADIENT_OPERATOR_SCHEMA(name)                              \
  C10_EXPORT void CAFFE2_PLEASE_ADD_OPERATOR_SCHEMA_FOR_##name(){}; \
  static OpSchema* C10_ANONYMOUS_VARIABLE(name) CAFFE2_UNUSED =     \
      1 ? nullptr : &OpSchemaRegistry::NewSchema(#name, __FILE__, __LINE__)

#else

#define GRADIENT_OPERATOR_SCHEMA(name) OPERATOR_SCHEMA(name)

#endif
#endif // CAFFE2_CORE_OPERATOR_SCHEMA_H_

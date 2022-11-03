#pragma once

#include <c10/util/StringUtil.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/alias_info.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <unordered_map>

namespace c10 {

// schema as used in the compiler for resolving function calls and reporting
// errors. These objects should be constructed from C10 schema once those
// are available.

struct Argument;
struct FunctionSchema;

bool operator==(const Argument& lhs, const Argument& rhs);

struct Argument {
  Argument(
      std::string name = "",
      TypePtr type = nullptr,
      c10::optional<int32_t> N = c10::nullopt,
      c10::optional<IValue> default_value = c10::nullopt,
      bool kwarg_only = false,
      c10::optional<AliasInfo> alias_info = c10::nullopt)
      : name_(std::move(name)),
        type_(type ? type : TensorType::get()),
        N_(std::move(N)),
        default_value_(std::move(default_value)),
        kwarg_only_(kwarg_only),
        alias_info_(std::move(alias_info)) {
  }
  const std::string& name() const {
    return name_;
  }
  const TypePtr& type() const {
    return type_;
  }
  c10::optional<int32_t> N() const {
    return N_;
  }
  const c10::optional<IValue>& default_value() const {
    return default_value_;
  }
  bool kwarg_only() const {
    return kwarg_only_;
  }
  const c10::optional<AliasInfo>& alias_info() const {
    return alias_info_;
  }
  bool is_inferred_type() const {
    bool is_inferred_type = false;
    TORCH_INTERNAL_ASSERT(type_);
    if (auto pt = type_->cast<TensorType>()) {
      if (pt->isInferredType()) {
        is_inferred_type = true;
      }
    }
    return is_inferred_type;
  }

  std::string formatTypeMismatchMsg(const std::string& actual_type) const {
    std::string inferred_type_hint;
    if (is_inferred_type()) {
      inferred_type_hint = c10::str(
          "Inferred '",
          name(),
          "' to be of type 'Tensor' ",
          "because it was not annotated with an explicit type.\n");
    }
    return c10::str(
        "Expected a value of type '",
        type()->repr_str(),
        "' for argument '",
        name(),
        "' but instead found type '",
        actual_type,
        "'.\n",
        inferred_type_hint);
  }

  Argument cloneWithType(TypePtr new_type) const {
    return Argument(
        name_,
        std::move(new_type),
        N_,
        default_value_,
        kwarg_only_,
        alias_info_);
  }

  // this function checks whether this Argument is backward compatible with
  // the old one. we consider the following cases are backward compatible:
  //   1) two arguments are equal
  //   2) this arg's type should be subtype of old
  //   3) this arg must provide the same default value if old arg has one,
  bool isBackwardCompatibleWith(
      const Argument& old,
      std::ostream* why_not=nullptr) const;

 private:
  std::string name_;
  TypePtr type_;
  // for list types, an optional statically known length for the list
  // e.g. for int[3]: type = ListType::ofInts(), N = 3
  // If present, this will allow scalars to be broadcast to this length to
  // become a list.
  c10::optional<int32_t> N_;

  c10::optional<IValue> default_value_;
  // is this only specifiable as a keyword argument?
  bool kwarg_only_;
  c10::optional<AliasInfo> alias_info_;
};

inline bool operator==(const Argument& lhs, const Argument& rhs) {
  return lhs.name() == rhs.name()
          && *lhs.type() == *rhs.type()
          && lhs.N() == rhs.N()
          && lhs.default_value() == rhs.default_value()
          && lhs.kwarg_only() == rhs.kwarg_only()
          && lhs.alias_info() == rhs.alias_info();
}

bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs);

struct FunctionSchema {
  FunctionSchema(
      std::string name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : name_({std::move(name), std::move(overload_name)}),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)),
        is_vararg_(is_vararg),
        is_varret_(is_varret) {
    checkSchema();
  }

  FunctionSchema(
      Symbol name,
      std::string overload_name,
      std::vector<Argument> arguments,
      std::vector<Argument> returns,
      bool is_vararg = false,
      bool is_varret = false)
      : FunctionSchema(
            name.toQualString(),
            std::move(overload_name),
            std::move(arguments),
            std::move(returns),
            is_vararg,
            is_varret) {
    checkSchema();
  }

  // Checks whether this schema is backward compatible with the old one.
  // The following conditions must be true:
  // [Function structure] The new schema's name, overload-name, varargs, and
  //      return arity are the same.
  // [Output Narrowing] The new schema's output type must be the same class
  //      or inherit from the old schema's output type.
  // [Argument count] The new schema must have at least as many arguments as
  //      the old schema (considering the list of positional and kwargs).
  // [Arg Compatibility] Every argument in the old schema has a corresponding
  //      argument in the new schema that:
  //        * is at the same position.
  //        * has the same name.
  //        * is either positional, or kwarg and the old argument was kwarg.
  //        * has the same type, or the old argument's type inherits from the
  //          new argument's type.
  // [Default Values] Every new argument must have a default value.
  // E.g.
  //   OK    f_new(a, b, c=1) => f_old(a, b)
  //   NOK   f_new(a, c=1, *, b) => f_old(a, *, b)
  //   OK    f_new(a, b, *, c) => f_old(a, *, b, c)
  //   NOK   f_new(a, *, b, c) -> f_old(a, b, *, c)
  //   NOK   f_new(a, *, c, b) => f_old(a, *, b, c)
  //   OK    f_new(a, *, b, c, d=1) => f_old(a, *, b, c)
  bool isBackwardCompatibleWith(
      const FunctionSchema& old,
      std::ostream* why_not = nullptr) const;

 private:
  OperatorName name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
  // if true then this schema takes an arbitrary number of additional arguments
  // after the argument specified in arguments
  // currently this is used primarily to represent 'primitive' operators whose
  // arguments are not checked by schema
  bool is_vararg_;
  bool is_varret_;

  // if no alias information is directly specified, what kind of "default"
  // alias information should we infer?
  // NB: due to alias analysis kind merging, this may be nullopt.  Eventually
  // this should always be set no matter what
  c10::optional<AliasAnalysisKind> alias_kind_;

  void checkArg(const IValue& value, const Argument& argument, optional<size_t> pos) const;

  void checkSchema() const {
    bool seen_default_arg = false;
    for (const auto& arg : arguments()) {
      if (arg.default_value()) {
        seen_default_arg = true;
      } else {
        // we have historically serialized broadcasting lists wo/default values,
        // so to not break BC allow lists here
        if (arg.type()->kind() == ListType::Kind) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(
            !seen_default_arg || arg.kwarg_only(),
            "Non-default positional argument follows default argument. Parameter ",
            arg.name(),
            " in ",
            *this);
      }
    }
  }

 public:

  void dump() const;

  const OperatorName& operator_name() const {
    return name_;
  }
  const std::string& name() const {
    return name_.name;
  }
  const std::string& overload_name() const {
    return name_.overload_name;
  }
  const std::vector<Argument>& arguments() const {
    return arguments_;
  }
  const std::vector<Argument>& returns() const {
    return returns_;
  }
  bool is_vararg() const {
    return is_vararg_;
  }
  bool is_varret() const {
    return is_varret_;
  }
  bool is_mutable() const {
    return std::any_of(
        arguments_.cbegin(), arguments_.cend(), [](const Argument& arg) {
          const auto& aliasInfo = arg.alias_info();
          return aliasInfo && aliasInfo.value().isWrite();
        });
  }

  c10::optional<int> argumentIndexWithName(const std::string& name) const {
    for(size_t i = 0; i < arguments().size(); ++i) {
      if(name == arguments()[i].name())
        return i;
    }
    return c10::nullopt;
  }
  FunctionSchema cloneWithName(std::string name, std::string overload_name) const {
    return FunctionSchema(
        std::move(name),
        std::move(overload_name),
        arguments(),
        returns(),
        is_vararg(),
        is_varret()
        );
  }
  FunctionSchema cloneWithArguments(std::vector<Argument> new_arguments) const {
    return FunctionSchema(
        name(),
        overload_name(),
        std::move(new_arguments),
        returns(),
        is_vararg(),
        is_varret());
  }
  FunctionSchema cloneWithReturns(std::vector<Argument> new_returns) const {
    return FunctionSchema(
        name(),
        overload_name(),
        arguments(),
        std::move(new_returns),
        is_vararg(),
        is_varret());
  }

  std::string formatTypeMismatchMsg(
      const Argument& expected,
      const std::string& actual_type,
      c10::optional<size_t> position = c10::nullopt,
      c10::optional<std::string> value = c10::nullopt) const;

  FunctionSchema cloneWithRemappedTypes(
      const std::function<TypePtr(TypePtr)> type_map) const;

  // Check that inputs have the correct types and appends any missing default
  // values.
  void checkAndNormalizeInputs(
      std::vector<IValue>& inputs,
      const std::unordered_map<std::string, IValue>& kwargs =
          std::unordered_map<std::string, IValue>{}) const;

  std::string findErrorInKwargs(const std::vector<std::string>& kwargs) const;

  bool hasAnyAliasInfo() const {
    for (const auto& arg : arguments_) {
      if (arg.alias_info().has_value()) {
        return true;
      }
    }
    for (const auto& ret : returns_) {
      if (ret.alias_info().has_value()) {
        return true;
      }
    }
    return false;
  }


  // TODO remove the mutation here
  bool isDefaultAliasAnalysisKind() const {
    return !alias_kind_;
  }
  AliasAnalysisKind aliasAnalysis() const {
    return alias_kind_.value_or(AliasAnalysisKind::CONSERVATIVE);
  }
  void setAliasAnalysis(AliasAnalysisKind v) {
    alias_kind_ = v;
  }

  c10::optional<c10::string_view> getNamespace() const {
    return name_.getNamespace();
  }

  // Returns true if we successfully set the namespace (as there
  // was none set, and false otherwise)
  bool setNamespaceIfNotSet(const char* ns) {
    return name_.setNamespaceIfNotSet(ns);
  }

  // can a function with this schema be substituted for a function of rhs's
  // schema and have the program typecheck?
  // as_method - if true, treat this schema as a method and ignore
  // the first argument, which will be the object in both cases
  bool isSubtypeOf(const FunctionSchema& rhs, bool as_method, std::ostream* why_not=nullptr) const;
};

inline bool operator==(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return lhs.name() == rhs.name()
     && lhs.overload_name() == rhs.overload_name()
     && lhs.arguments() == rhs.arguments()
     && lhs.returns() == rhs.returns()
     && lhs.is_vararg() == rhs.is_vararg()
     && lhs.is_varret() == rhs.is_varret();
}

inline bool operator!=(const FunctionSchema& lhs, const FunctionSchema& rhs) {
  return !(lhs == rhs);
}

// print out Argument, which is compatible with FunctionSchema parser
// full format: Type(alias)? name=default_value
inline std::ostream& operator<<(std::ostream& out, const Argument& arg) {

  // for adjusting the ? position.
  // in schema, we have Tensor?(a!) input, and t(a!)?.
  // however, t?(a!) doesn't work with schema parser.
  // so we always use Type(alias)? format
  auto type = arg.type();
  bool is_opt = type->kind() == OptionalType::Kind;
  auto unopt_type = is_opt ? type->castRaw<OptionalType>()->getElementType() : type;

  if (unopt_type->kind() == ListType::Kind && arg.N()) {
    // sized lists get size N from arg, not type
    auto list = unopt_type->cast<c10::ListType>();
    out << list->getElementType()->str() << "[" << *arg.N() << "]";
  } else {
    out << unopt_type->str();
  }

  if (arg.alias_info()) {
    out << arg.alias_info().value();
  }

  if (is_opt) {
    out << "?";
  }

  if (!arg.name().empty()) {
    out << " " << arg.name();
  }

  if (arg.default_value()) {
    out << "=";
    if (type->kind() == c10::TypeKind::StringType || (unopt_type->kind() == c10::TypeKind::StringType && !arg.default_value().value().isNone())) {
      printQuotedString(out, arg.default_value().value().toStringRef());
    } else {
      out << arg.default_value().value();
    }
  }

  return out;
}

inline std::ostream& operator<<(std::ostream& out, const FunctionSchema& schema);

inline std::string toString(const FunctionSchema& schema) {
  std::ostringstream str;
  str << schema;
  return str.str();
}

} // namespace c10

#include <ATen/core/function_schema_inl.h>

#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <torch/csrc/onnx/onnx.h>

#include <ostream>

namespace ONNX_NAMESPACE {
class ModelProto;
}

namespace torch {
namespace jit {

// This map is used to keep track of parameters that should be exported
// externally. When `defer_weight_export` is true, the returned map contains
// kv pairs that map {external reference name} -> {at::Tensor to be exported}.
// It is the responsibility of the caller to export these appropriately.
//
// For example, when exporting to a zip archive, the caller may write out files
// for each entry in the export map, with the filename being the key and the
// file contents being the raw tensor data.
using RawDataExportMap = std::unordered_map<std::string, at::Tensor>;

using SymbolDimMap = std::map<c10::ShapeSymbol, std::string>;

TORCH_API std::tuple<
    std::shared_ptr<::ONNX_NAMESPACE::ModelProto>,
    RawDataExportMap,
    SymbolDimMap>
export_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    const std::unordered_map<
        std::string,
        std::unordered_map<int64_t, std::string>>& dynamic_axes,
    bool defer_weight_export = false,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool strip_doc_string = true,
    bool keep_initializers_as_inputs = true,
    const std::map<std::string, int>& custom_opsets = {},
    bool add_node_names = true,
    bool use_external_data_format = false,
    const std::string& onnx_file_path = std::string());

TORCH_API std::string serialize_model_proto_to_string(
    const std::shared_ptr<::ONNX_NAMESPACE::ModelProto>& model_proto);

TORCH_API void check_onnx_proto(const std::string& proto_string);

// Serializer for both oldsyle and unified format TorchScript serialization
class TORCH_API ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(
      caffe2::serialize::PyTorchStreamWriter& export_writer)
      : writer_(export_writer), current_source_range_tag_(0) {}

  void writeFiles(const std::string& code_dir);
  void serialize(
      const Module& module,
      const ExtraFilesMap& extra_files,
      bool bytecode_format,
      bool save_mobile_debug_info);
  void serialize_unified_format(Module& module, uint64_t script_module_id);

  ~ScriptModuleSerializer() = default;

 private:
  void convertNamedType(const c10::NamedTypePtr& class_type);
  void convertTypes(const at::NamedTypePtr& root_type);
  void writeExtraFiles(const Module& module, const ExtraFilesMap& extra_files);
  void writeMobileMetadata(
      const Module& module,
      const ExtraFilesMap& extra_files);
  void writeByteCode(const Module& module, bool save_mobile_debug_info);
  void writeArchive(
      const IValue& value,
      const std::string& archive_name,
      const std::string& archive_dir,
      const std::string& tensor_dir,
      bool tensor_cdata_naming_scheme = false);
  void updateSourceRangeTags(const SourceRangeRecords& ranges);

  caffe2::serialize::PyTorchStreamWriter& writer_;
  std::vector<at::IValue> constant_table_;

  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;
  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;

  // Uniquely identifies a SourceRange in a model.
  // SourceRanges are associated with Nodes of Graphs.
  // However for mobile deployment we dont intend to ship
  // full JIT with capabilities of reading code and constructing
  // graphs.
  // Instead we serialize the Code generated from graph of the methods.
  // Code is serialized in bytecode format that contains instructions
  // corresponding to the nodes of the graph. Since original graph is gone, the
  // question is how do we identify where the ops, in serialized bytecode, come
  // from in original model code. We do this in two parts.
  // 1. Associate a unique tag to SourceRange.
  // 2. Serialize this unique_tag.
  //  2.1 Meaning save <byte_offset, source_range_tag, source range> instead of
  //      <byte_offset, source range>
  // 3. During serializing model for mobile, i.e. bytecode generation,
  //    save unique tag of SourceRange corresponding to the Node.
  // 4. During deserialization, read all the debug_pkl, to construct a map
  //    of <unique_tag, SourceRange> and use tag saved with OPs in bytecode
  //    to lookup the source range.
  // Strictly speaking we will serialize InlinedCallStack directly, which
  // contains SourceRange. This way we have access to entire callstack and not
  // just source information about where the node is, since bytecode inlines the
  // graph before saving it.
  SourceRangeTagMap source_range_tags_;
  int64_t current_source_range_tag_;
};

// For testing purposes
TORCH_API std::string pretty_print_onnx(
    const std::shared_ptr<Graph>& graph,
    const std::map<std::string, at::Tensor>& initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    ::torch::onnx::OperatorExportTypes operator_export_type =
        ::torch::onnx::OperatorExportTypes::ONNX,
    bool google_printer = false,
    bool keep_initializers_as_inputs = true,
    const std::map<std::string, int>& custom_opsets = {},
    bool add_node_names = true);

TORCH_API void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false);

TORCH_API void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false);

TORCH_API void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& metadata = ExtraFilesMap(),
    bool bytecode_format = false,
    bool save_mobile_debug_info = false);

// Write the bytes of a pickle archive and the tensors referenced inside that
// archive
TORCH_API void writeArchiveAndTensors(
    const std::string& archive_name,
    const char* pickle_bytes,
    size_t size,
    const std::vector<at::Tensor>& tensors,
    caffe2::serialize::PyTorchStreamWriter& out);

// Surrounding system can install an additional hook to produce extra files
// with metadata based on environment every time a module is serialized.
using ExportModuleExtraFilesHook = std::function<ExtraFilesMap(const Module&)>;
TORCH_API void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook);

using ExportModuleMobileInfoConverter =
    std::function<c10::Dict<std::string, std::string>(
        const Module&,
        const std::unordered_map<std::string, std::string>&)>;
TORCH_API void SetExportModuleMobileInfoConverter(
    ExportModuleMobileInfoConverter converter);

/**
 * Generates new bytecode for a Script module and returns what the op list
 * would be for a LiteScriptModule based off the current code base. If you
 * have a LiteScriptModule and want to get the currently present
 * list of ops call _export_operator_list instead.
 */
TORCH_API std::vector<std::string> export_opnames(const Module& m);

} // namespace jit
} // namespace torch

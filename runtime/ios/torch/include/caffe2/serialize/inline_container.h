#pragma once

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <istream>
#include <mutex>
#include <ostream>

#include <c10/core/Allocator.h>
#include <c10/core/Backend.h>

#include "caffe2/serialize/istream_adapter.h"
#include "caffe2/serialize/read_adapter_interface.h"
#include "caffe2/serialize/versions.h"

extern "C" {
typedef struct mz_zip_archive mz_zip_archive;
}

// PyTorch containers are a special zip archive with the following layout
// archive_name.zip contains:
//    archive_name/
//        version # a file with a single decimal number written in ascii,
//                # used to establish the version of the archive format
//        model.json # overall model description, this is a json output of
//                   # ModelDef from torch.proto
//        # the following names are by convention only, model.json will
//        # refer to these files by full names
//        tensors/
//          0 # flat storage for tensor data, meta-data about shapes, etc. is
//            # in model.json
//          1
//          ...
//        # code entries will only exist for modules that have methods attached
//        code/
//          archive_name.py # serialized torch script code (python syntax, using
//          PythonPrint) archive_name_my_submodule.py # submodules have separate
//          files
//
// The PyTorchStreamWriter also ensures additional useful properties for these
// files
// 1. All files are stored uncompressed.
// 2. All files in the archive are aligned to 64 byte boundaries such that
//    it is possible to mmap the entire file and get an aligned pointer to
//    tensor data.
// 3. We universally write in ZIP64 format for consistency.

// The PyTorchStreamReader also provides additional properties:
// 1. It can read zip files that are created with common
//    zip tools. This means that even though our writer doesn't compress files,
//    the reader can still read files that were compressed.
// 2. It provides a getRecordOffset function which returns the offset into the
//    raw file where file data lives. If the file was written with
//    PyTorchStreamWriter it is guaranteed to be 64 byte aligned.

// PyTorchReader/Writer handle checking the version number on the archive format
// and ensure that all files are written to a archive_name directory so they
// unzip cleanly.

// When developing this format we want to pay particular attention to the
// following use cases:
//
// -- Reading --
// 1) Reading with full random access
//   a) Reading with file api's such as fread()
//   b) mmaping the file and jumping around the mapped region
// 2) Reading with 1-pass sequential access
//      -> A reader will need to build up a data structure of parsed structures
//         as it reads
//
// -- Writing --
// 1) Writing with full random access
// 2) Writing with 1-pass sequential access
//      -> We must take care not to require updating values that have already
//         been written. We place the variable-length index at the end and do
//         not put any indicies into the header to fulfill this constraint.

// The model.json, which contains all the metadata information,
// should be written as the last file. One reason is that the size of tensor
// data is usually stable. As long as the shape and type of the tensor do not
// change, the size of the data won't change. On the other sied, the size of the
// serialized model is likely to change, so we store it as the last record, and
// we don't need to move previous records when updating the model data.

// The zip format is sufficiently flexible to handle the above use-case.
// it puts its central directory at the end of the archive and we write
// model.json as the last file when writing after we have accumulated all
// other information.

namespace caffe2 {
namespace serialize {

class TORCH_API PyTorchStreamReader final {
 public:
  explicit PyTorchStreamReader(const std::string& file_name);
  explicit PyTorchStreamReader(std::istream* in);
  explicit PyTorchStreamReader(std::shared_ptr<ReadAdapterInterface> in);

  // return dataptr, size
  std::tuple<at::DataPtr, size_t> getRecord(const std::string& name);
  size_t getRecordOffset(const std::string& name);
  bool hasRecord(const std::string& name);
  std::vector<std::string> getAllRecords();

  ~PyTorchStreamReader();
  uint64_t version() const {
    return version_;
  }

 private:
  void init();
  size_t read(uint64_t pos, char* buf, size_t n);
  void valid(const char* what, const char* info = "");
  size_t getRecordID(const std::string& name);

  friend size_t
  istream_read_func(void* pOpaque, uint64_t file_ofs, void* pBuf, size_t n);
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::string archive_name_plus_slash_;
  std::shared_ptr<ReadAdapterInterface> in_;
  int64_t version_;
  std::mutex reader_lock_;
};

class TORCH_API PyTorchStreamWriter final {
 public:
  explicit PyTorchStreamWriter(std::string archive_name);
  explicit PyTorchStreamWriter(
      const std::function<size_t(const void*, size_t)>& writer_func);

  void setMinVersion(const uint64_t version);

  void writeRecord(
      const std::string& name,
      const void* data,
      size_t size,
      bool compress = false);
  void writeEndOfFile();

  const std::vector<std::string>& getAllWrittenRecords();

  bool finalized() const {
    return finalized_;
  }

  const std::string& archiveName() {
    return archive_name_;
  }

  ~PyTorchStreamWriter();

 private:
  void setup(const std::string& file_name);
  void valid(const char* what, const char* info = "");
  size_t current_pos_ = 0;
  std::vector<std::string> files_written;
  std::unique_ptr<mz_zip_archive> ar_;
  std::string archive_name_;
  std::string archive_name_plus_slash_;
  std::string padding_;
  std::ofstream file_stream_;
  std::function<size_t(const void*, size_t)> writer_func_;
  uint64_t version_ = kProducedFileFormatVersion;
  bool finalized_ = false;
  bool err_seen_ = false;
  friend size_t ostream_write_func(
      void* pOpaque,
      uint64_t file_ofs,
      const void* pBuf,
      size_t n);
};

namespace detail {
// Writer-specific constants
constexpr uint64_t kFieldAlignment = 64;

// Returns a record to be appended to the local user extra data entry in order
// to make data beginning aligned at kFieldAlignment bytes boundary.
size_t getPadding(
    size_t cursor,
    size_t filename_size,
    size_t size,
    std::string& padding_buf);
}

} // namespace serialize
} // namespace caffe2

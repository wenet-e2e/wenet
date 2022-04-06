// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#ifndef DECODER_SEARCH_INTERFACE_H_
#define DECODER_SEARCH_INTERFACE_H_

namespace wenet {

#include <vector>

enum SearchType {
  kPrefixBeamSearch = 0x00,
  kWfstBeamSearch = 0x01,
};

class SearchInterface {
 public:
  virtual ~SearchInterface() {}
  virtual void Search(const std::vector<std::vector<float>>& logp) = 0;
  virtual void Reset() = 0;
  virtual void FinalizeSearch() = 0;

  virtual SearchType Type() const = 0;
  // N-best inputs id
  virtual const std::vector<std::vector<int>>& Inputs() const = 0;
  // N-best outputs id
  virtual const std::vector<std::vector<int>>& Outputs() const = 0;
  // N-best likelihood
  virtual const std::vector<float>& Likelihood() const = 0;
  // N-best timestamp
  virtual const std::vector<std::vector<int>>& Times() const = 0;
};

}  // namespace wenet

#endif  // DECODER_SEARCH_INTERFACE_H_

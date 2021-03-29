// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: zhendong.peng@mobvoi.com (Zhendong Peng)

#ifndef DECODER_DECODER_INTERFACE_H_
#define DECODER_DECODER_INTERFACE_H_

#include <string>
#include <utility>
#include <vector>

#include "decoder/ctc_prefix_beam_search.h"

namespace wenet {

struct DecodeOptions {
  int chunk_size = 16;
  int num_left_chunks = -1;
  CtcPrefixBeamSearchOptions ctc_search_opts;
};

struct WordPiece {
  std::string word;
  int start = -1;
  int end = -1;

  WordPiece(std::string word, int start, int end) {
    this->word = std::move(word);
    this->start = start;
    this->end = end;
  }

  friend std::ostream& operator<<(std::ostream& ostream,
                                  const WordPiece& word_piece) {
    return ostream << "{word:" << word_piece.word
                   << ", start_time: " << word_piece.start
                   << " end_time: " << word_piece.end
                   << "}";
  }
};

struct DecodeResult {
  float score = -kFloatMax;
  std::string sentence;
  std::vector<WordPiece> word_pieces;

  static bool CompareFunc(const DecodeResult& a, const DecodeResult& b) {
    return a.score > b.score;
  }

  friend std::ostream& operator<<(std::ostream& ostream,
                                  const DecodeResult& decodeResult) {
    ostream << "{score: " << decodeResult.score
            << ", sentence: " << decodeResult.sentence
            << ", word_pieces: [";
    for (const WordPiece& word_piece : decodeResult.word_pieces) {
      ostream << word_piece << ", ";
    }
    return ostream << "]}";
  }
};

class DecoderInterface {
 public:
  DecoderInterface() {}
  virtual ~DecoderInterface() {}

  virtual bool Decode() = 0;
  virtual void Reset() = 0;
};

}  // namespace wenet

#endif  // DECODER_DECODER_INTERFACE_H_

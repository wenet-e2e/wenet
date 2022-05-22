// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)

#include <memory>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "decoder/params.h"
#include "frontend/wav.h"
#include "utils/flags.h"
#include "utils/string.h"

DEFINE_string(text, "", "kaldi style text input file");
DEFINE_string(wav_scp, "", "kaldi style wav scp");
DEFINE_double(is_penalty, 1.0,
              "insertion/substitution penalty for align insertion");
DEFINE_double(del_penalty, 1.0, "deletion penalty for align insertion");
DEFINE_string(result, "", "result output file");
DEFINE_string(timestamp, "", "timestamp output file");

namespace wenet {

const char* kDeletion = "<del>";
// Is: Insertion and substitution
const char* kIsStart = "<is>";
const char* kIsEnd = "</is>";

bool MapToLabel(const std::string& text,
                std::shared_ptr<fst::SymbolTable> symbol_table,
                std::vector<int>* labels) {
  labels->clear();
  // Split label to char sequence
  std::vector<std::string> chars;
  SplitUTF8StringToChars(text, &chars);
  for (size_t i = 0; i < chars.size(); i++) {
    // ▁ is special symbol for white space
    std::string label = chars[i] != " " ? chars[i] : "▁";
    int id = symbol_table->Find(label);
    if (id != -1) {  // fst::kNoSymbol
      // LOG(INFO) << label << " " << id;
      labels->push_back(id);
    }
  }
  return true;
}

std::shared_ptr<fst::SymbolTable> MakeSymbolTableForFst(
    std::shared_ptr<fst::SymbolTable> isymbol_table) {
  LOG(INFO) << isymbol_table;
  CHECK(isymbol_table != nullptr);
  auto osymbol_table = std::make_shared<fst::SymbolTable>();
  osymbol_table->AddSymbol("<eps>", 0);
  CHECK_EQ(isymbol_table->Find("<blank>"), 0);
  osymbol_table->AddSymbol("<blank>", 1);
  for (int i = 1; i < isymbol_table->NumSymbols(); i++) {
    std::string symbol = isymbol_table->Find(i);
    osymbol_table->AddSymbol(symbol, i + 1);
  }
  osymbol_table->AddSymbol(kDeletion, isymbol_table->NumSymbols() + 1);
  osymbol_table->AddSymbol(kIsStart, isymbol_table->NumSymbols() + 2);
  osymbol_table->AddSymbol(kIsEnd, isymbol_table->NumSymbols() + 3);
  return osymbol_table;
}

void CompileCtcFst(std::shared_ptr<fst::SymbolTable> symbol_table,
                   fst::StdVectorFst* ofst) {
  ofst->DeleteStates();
  int start = ofst->AddState();
  ofst->SetStart(start);
  CHECK_EQ(symbol_table->Find("<eps>"), 0);
  CHECK_EQ(symbol_table->Find("<blank>"), 1);
  ofst->AddArc(start, fst::StdArc(1, 0, 0.0, start));
  // Exclude kDeletion and kInsertion
  for (int i = 2; i < symbol_table->NumSymbols() - 3; i++) {
    int s = ofst->AddState();
    ofst->AddArc(start, fst::StdArc(i, i, 0.0, s));
    ofst->AddArc(s, fst::StdArc(i, 0, 0.0, s));
    ofst->AddArc(s, fst::StdArc(0, 0, 0.0, start));
  }
  ofst->SetFinal(start, fst::StdArc::Weight::One());
  fst::ArcSort(ofst, fst::StdOLabelCompare());
}

void CompileAlignFst(std::vector<int> labels,
                     std::shared_ptr<fst::SymbolTable> symbol_table,
                     fst::StdVectorFst* ofst) {
  ofst->DeleteStates();
  int deletion = symbol_table->Find(kDeletion);
  int insertion_start = symbol_table->Find(kIsStart);
  int insertion_end = symbol_table->Find(kIsEnd);

  int start = ofst->AddState();
  ofst->SetStart(start);
  // Filler State
  int filler_start = ofst->AddState();
  int filler_end = ofst->AddState();
  for (int i = 2; i < symbol_table->NumSymbols() - 3; i++) {
    ofst->AddArc(filler_start, fst::StdArc(i, i, FLAGS_is_penalty, filler_end));
  }
  ofst->AddArc(filler_end, fst::StdArc(0, 0, 0.0, filler_start));

  int prev = start;
  // Alignment path and optional filler
  for (size_t i = 0; i < labels.size(); i++) {
    int cur = ofst->AddState();
    // 1. Insertion or Substitution
    ofst->AddArc(prev, fst::StdArc(0, insertion_start, 0.0, filler_start));
    ofst->AddArc(filler_end, fst::StdArc(0, insertion_end, 0.0, prev));
    // 2. Correct
    ofst->AddArc(prev, fst::StdArc(labels[i], labels[i], 0.0, cur));
    // 3. Deletion
    ofst->AddArc(prev, fst::StdArc(0, deletion, FLAGS_del_penalty, cur));

    prev = cur;
  }
  // Optional add endding filler
  ofst->AddArc(prev, fst::StdArc(0, insertion_start, 0.0, filler_start));
  ofst->AddArc(filler_end, fst::StdArc(0, insertion_end, 0.0, prev));
  ofst->SetFinal(prev, fst::StdArc::Weight::One());
  fst::ArcSort(ofst, fst::StdILabelCompare());
}

}  // namespace wenet

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitDecodeResourceFromFlags();
  CHECK(decode_resource->unit_table != nullptr);

  auto wfst_symbol_table =
      wenet::MakeSymbolTableForFst(decode_resource->unit_table);
  // wfst_symbol_table->WriteText("fst.txt");
  // Reset symbol_table to on-the-fly generated wfst_symbol_table
  decode_resource->symbol_table = wfst_symbol_table;

  // Compile ctc FST
  fst::StdVectorFst ctc_fst;
  wenet::CompileCtcFst(wfst_symbol_table, &ctc_fst);
  // ctc_fst.Write("ctc.fst");

  std::unordered_map<std::string, std::string> wav_table;
  std::ifstream wav_is(FLAGS_wav_scp);
  std::string line;
  while (std::getline(wav_is, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    CHECK_EQ(strs.size(), 2);
    wav_table[strs[0]] = strs[1];
  }

  std::ifstream text_is(FLAGS_text);
  std::ofstream result_os(FLAGS_result, std::ios::out);
  std::ofstream timestamp_out;
  if (!FLAGS_timestamp.empty()) {
    timestamp_out.open(FLAGS_timestamp, std::ios::out);
  }
  std::ostream& timestamp_os =
      FLAGS_timestamp.empty() ? std::cout : timestamp_out;

  while (std::getline(text_is, line)) {
    std::vector<std::string> strs;
    wenet::SplitString(line, &strs);
    if (strs.size() < 2) continue;
    std::string key = strs[0];
    LOG(INFO) << "Processing " << key;
    if (wav_table.find(key) != wav_table.end()) {
      strs.erase(strs.begin());
      std::string text = wenet::JoinString(" ", strs);
      std::vector<int> labels;
      wenet::MapToLabel(text, wfst_symbol_table, &labels);
      // Prepare FST for alignment decoding
      fst::StdVectorFst align_fst;
      wenet::CompileAlignFst(labels, wfst_symbol_table, &align_fst);
      // align_fst.Write("align.fst");
      auto decoding_fst = std::make_shared<fst::StdVectorFst>();
      fst::Compose(ctc_fst, align_fst, decoding_fst.get());
      // decoding_fst->Write("decoding.fst");
      // Preapre feature pipeline
      wenet::WavReader wav_reader;
      if (!wav_reader.Open(wav_table[key])) {
        LOG(WARNING) << "Error in reading " << wav_table[key];
        continue;
      }
      int num_samples = wav_reader.num_samples();
      CHECK_EQ(wav_reader.sample_rate(), FLAGS_sample_rate);
      auto feature_pipeline =
          std::make_shared<wenet::FeaturePipeline>(*feature_config);
      feature_pipeline->AcceptWaveform(wav_reader.data(), num_samples);
      feature_pipeline->set_input_finished();
      decode_resource->fst = decoding_fst;
      LOG(INFO) << "num frames " << feature_pipeline->num_frames();
      wenet::AsrDecoder decoder(feature_pipeline, decode_resource,
                                *decode_config);
      while (true) {
        wenet::DecodeState state = decoder.Decode();
        if (state == wenet::DecodeState::kEndFeats) {
          decoder.Rescoring();
          break;
        }
      }
      std::string final_result;
      std::string timestamp_str;
      if (decoder.DecodedSomething()) {
        const wenet::DecodeResult& result = decoder.result()[0];
        final_result = result.sentence;
        std::stringstream ss;
        for (const auto& w : result.word_pieces) {
          ss << " " << w.word << " " << w.start << " " << w.end;
        }
        timestamp_str = ss.str();
      }
      result_os << key << " " << final_result << std::endl;
      timestamp_os << key << " " << timestamp_str << std::endl;
      LOG(INFO) << key << " " << final_result;
    } else {
      LOG(WARNING) << "No wav file for " << key;
    }
  }
  return 0;
}

#include "boost/json/src.hpp"
#include "decoder/params.h"
#include "post_processor/post_processor.h"
#include "utils/string.h"
#include "utils/timer.h"
#include "utils/utils.h"

#include "runtime_wrapper.h"

namespace json = boost::json;

const char *kDeletion = "<del>";
// Is: Insertion and substitution
const char *kIsStart = "<is>";
const char *kIsEnd = "</is>";

// serialize Decode Result, this function should be in Decode
std::string SerializeResult(const std::vector<wenet::DecodeResult> &results,
                            bool finish, int n = 1) {
  json::array nbest;
  for (auto &path : results) {
    json::object jpath({{"sentence", path.sentence}});
    if (finish) {
      json::array word_pieces;
      for (auto &word_piece : path.word_pieces) {
        json::object jword_piece({{"word", word_piece.word},
                                  {"start", word_piece.start},
                                  {"end", word_piece.end}});
        word_pieces.emplace_back(jword_piece);
      }
      jpath.emplace("word_pieces", word_pieces);
    }
    nbest.emplace_back(jpath);

    if (nbest.size() == n) {
      break;
    }
  }
  return json::serialize(nbest);
}

std::once_flag once_nitialized_;

std::shared_ptr<wenet::FeaturePipelineConfig>
InitFeaturePipelineConfigFromParams(const Params &params) {
  auto feature_config = std::make_shared<wenet::FeaturePipelineConfig>(
      params.num_bins, params.sample_rate);
  return feature_config;
}

std::shared_ptr<wenet::DecodeOptions> InitDecodeOptionsFromParams(
    const Params &params) {
  auto decode_config = std::make_shared<wenet::DecodeOptions>();
  decode_config->chunk_size = params.chunk_size;
  decode_config->num_left_chunks = params.num_left_chunks;
  decode_config->ctc_weight = params.ctc_weight;
  decode_config->reverse_weight = params.ctc_weight;
  decode_config->rescoring_weight = params.reverse_weight;
  decode_config->ctc_wfst_search_opts.max_active = params.max_active;
  decode_config->ctc_wfst_search_opts.min_active = params.min_active;
  decode_config->ctc_wfst_search_opts.beam = params.beam;
  decode_config->ctc_wfst_search_opts.lattice_beam = params.lattice_beam;
  decode_config->ctc_wfst_search_opts.acoustic_scale = params.acoustic_scale;
  decode_config->ctc_wfst_search_opts.blank_skip_thresh =
      params.blank_threshold;
  decode_config->ctc_wfst_search_opts.nbest = params.nbest;
  decode_config->ctc_prefix_search_opts.first_beam_size =
      params.first_beam_size;
  decode_config->ctc_prefix_search_opts.second_beam_size =
      params.second_beam_size;
  return decode_config;
}

std::shared_ptr<wenet::DecodeResource> InitDecodeResourceFromParams(
    const Params &params) {
  auto resource = std::make_shared<wenet::DecodeResource>();

  auto model_path = params.model_path;
  auto num_threads = params.num_threads;
  LOG(INFO) << "Reading model " << model_path << " to use " << num_threads
            << " threads";
  auto model = std::make_shared<wenet::TorchAsrModel>();
  model->Read(model_path, num_threads);
  resource->model = model;

  std::shared_ptr<fst::Fst<fst::StdArc>> fst = nullptr;
  if (params.fst_path != "") {
    auto fst_path = params.fst_path;
    LOG(INFO) << "Reading fst " << fst_path;
    fst.reset(fst::Fst<fst::StdArc>::Read(fst_path));
    CHECK(fst != nullptr);
  }
  resource->fst = fst;

  auto dict_path = params.dict_path;
  LOG(INFO) << "Reading symbol table " << dict_path;
  auto symbol_table =
      std::shared_ptr<fst::SymbolTable>(fst::SymbolTable::ReadText(dict_path));
  resource->symbol_table = symbol_table;

  std::shared_ptr<fst::SymbolTable> unit_table = nullptr;
  if (params.unit_path != "") {
    auto unit_path = params.unit_path;
    LOG(INFO) << "Reading unit table " << unit_path;
    unit_table = std::shared_ptr<fst::SymbolTable>(
        fst::SymbolTable::ReadText(unit_path));
    CHECK(unit_table != nullptr);
  } else if (fst == nullptr) {
    LOG(INFO) << "Using symbol table as unit table";
    unit_table = symbol_table;
  }
  resource->unit_table = unit_table;

  if (params.context_path != "") {
    auto context_path = params.context_path;
    LOG(INFO) << "Reading context " << context_path;
    std::vector<std::string> contexts;
    std::ifstream infile(context_path);
    std::string context;
    while (getline(infile, context)) {
      contexts.emplace_back(wenet::Trim(context));
    }

    wenet::ContextConfig config;
    config.context_score = params.context_score;
    resource->context_graph = std::make_shared<wenet::ContextGraph>(config);
    resource->context_graph->BuildContextGraph(contexts, symbol_table);
  }

  wenet::PostProcessOptions post_process_opts;
  post_process_opts.language_type = params.language_type == 0
                                        ? wenet::kMandarinEnglish
                                        : wenet::kIndoEuropean;
  post_process_opts.lowercase = params.lower_case;
  resource->post_processor =
      std::make_shared<wenet::PostProcessor>(std::move(post_process_opts));
  return resource;
}

SimpleAsrModelWrapper::SimpleAsrModelWrapper(const Params &params) {
  std::call_once(once_nitialized_,
                 [&]() { google::InitGoogleLogging("wenet python wrapper:"); });

  // for now only support torchScript model
  model_backend_ = torchScript;
  feature_config_ = InitFeaturePipelineConfigFromParams(params);
  decode_config_ = InitDecodeOptionsFromParams(params);
  decode_resource_ = InitDecodeResourceFromParams(params);
}

std::string SimpleAsrModelWrapper::Recognize(char *pcm, int num_samples,
                                             int n_best) {
  return this->RecognizeMayWithLocalFst(pcm, num_samples, n_best, nullptr);
}

// TODO: split function
std::string SimpleAsrModelWrapper::RecognizeMayWithLocalFst(
    char *pcm, int num_samples, int nbest,
    std::shared_ptr<wenet::DecodeResource> local_decode_resource) {
  std::vector<float> pcm_data(num_samples);
  const int16_t *pdata = reinterpret_cast<const int16_t *>(pcm);
  for (int i = 0; i < num_samples; i++) {
    pcm_data[i] = static_cast<float>(*pdata);
    pdata++;
  }
  auto feature_pipeline =
      std::make_shared<wenet::FeaturePipeline>(*feature_config_);
  feature_pipeline->AcceptWaveform(pcm_data);
  feature_pipeline->set_input_finished();
  // resource_->decode_resource->fst = decoding_fst;
  LOG(INFO) << "num frames " << feature_pipeline->num_frames();

  auto decode_resource = decode_resource_;
  if (local_decode_resource != nullptr) {
    decode_resource = local_decode_resource;
  }
  wenet::AsrDecoder decoder(feature_pipeline, decode_resource, *decode_config_);
  while (true) {
    wenet::DecodeState state = decoder.Decode();
    if (state == wenet::DecodeState::kEndFeats) {
      decoder.Rescoring();
      break;
    }
  }

  if (decoder.DecodedSomething()) {
    return SerializeResult(decoder.result(), true, nbest);
  }
  return std::string();
}

void StreammingAsrWrapper::DecodeThreadFunc(int nbest) {
  while (true) {
    auto state = decoder_->Decode();
    if (state == wenet::DecodeState::kEndFeats) {
      decoder_->Rescoring();
      std::string result = SerializeResult(decoder_->result(), true, nbest);
      {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result_ = std::move(result);
        stop_recognition_ = true;
      }
      has_result_.notify_one();
      break;
    } else if (state == wenet::DecodeState::kEndpoint) {
      decoder_->Rescoring();
      std::string result = SerializeResult(decoder_->result(), true, nbest);
      {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result_ = std::move(result);
        if (continuous_decoding_) {
          decoder_->ResetContinuousDecoding();
        } else {
          stop_recognition_ = true;
          has_result_.notify_one();
          break;
        }
      }
      // If it's not continuous decoidng, continue to do next recognition
      has_result_.notify_one();
      // otherwise stop the recognition
    } else {
      stop_recognition_ = false;
      if (decoder_->DecodedSomething()) {
        std::string result = SerializeResult(decoder_->result(), false, 1);
        {
          std::lock_guard<std::mutex> lock(result_mutex_);
          result_ = std::move(result);
        }
        has_result_.notify_one();
      }
    }
  }
}

void StreammingAsrWrapper::AccepAcceptWaveform(char *pcm, int num_samples,
                                               bool final) {
  if (pcm == nullptr || num_samples == 0) {
    return;
  }
  // bytes to wavform
  std::vector<float> pcm_data(num_samples);
  const int16_t *pdata = reinterpret_cast<const int16_t *>(pcm);
  for (int i = 0; i < num_samples; i++) {
    pcm_data[i] = static_cast<float>(*pdata);
    pdata++;
  }
  feature_pipeline_->AcceptWaveform(pcm_data);
  if (final && !stop_recognition_) {
    feature_pipeline_->set_input_finished();
    stop_recognition_ = true;
  }
}

void StreammingAsrWrapper::Reset(int nbest, bool continuous_decoding) {
  // CHECK(!decode_thread_->joinable());
  if (!stop_recognition_) {
    feature_pipeline_->set_input_finished();
  }
  if (decode_thread_->joinable()) {
    decode_thread_->join();
  }

  continuous_decoding_ = continuous_decoding;
  stop_recognition_ = false;
  result_.clear();
  feature_pipeline_->Reset();
  decoder_->Reset();

  decode_thread_ = std::make_unique<std::thread>(
      &StreammingAsrWrapper::DecodeThreadFunc, this, nbest);
}

bool StreammingAsrWrapper::GetInstanceResult(std::string &result) {
  bool is_final = false;
  {
    std::unique_lock<std::mutex> lock(result_mutex_);
    if (!result_.empty()) {
      result = result_;
      is_final = stop_recognition_;

      result_.clear();
      return is_final;
    } else {
      while (result_.empty()) {
        has_result_.wait(lock);
        result = result_;
        is_final = stop_recognition_;
        break;
      }
      result_.clear();
    }
  }
  return is_final;
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
                   fst::StdVectorFst *ofst) {
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

LabelCheckerWrapper::LabelCheckerWrapper(
    std::shared_ptr<SimpleAsrModelWrapper> model)
    : model_(model), ctc_fst_(std::make_shared<fst::StdVectorFst>()) {
  auto decode_resource = model_->decode_resource();
  CHECK(decode_resource->unit_table != nullptr);

  wfst_symbol_table_ = MakeSymbolTableForFst(decode_resource->unit_table);
  CompileCtcFst(wfst_symbol_table_, ctc_fst_.get());
}

bool MapToLabel(const std::vector<string> &chars,
                std::shared_ptr<fst::SymbolTable> symbol_table,
                std::vector<int> *labels) {
  labels->clear();
  // Split label to char sequence
  for (size_t i = 0; i < chars.size(); i++) {
    std::string label = chars[i];
    int id = symbol_table->Find(label);
    if (id != -1) {  // fst::kNoSymbol
      // LOG(INFO) << label << " " << id;
      labels->push_back(id);
    }
  }
  return true;
}

void CompileAlignFst(std::vector<int> &labels,
                     std::shared_ptr<fst::SymbolTable> symbol_table,
                     fst::StdVectorFst *ofst, float is_penalty,
                     float del_penalty) {
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
    ofst->AddArc(filler_start, fst::StdArc(i, i, is_penalty, filler_end));
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
    ofst->AddArc(prev, fst::StdArc(0, deletion, del_penalty, cur));

    prev = cur;
  }
  // Optional add endding filler
  ofst->AddArc(prev, fst::StdArc(0, insertion_start, 0.0, filler_start));
  ofst->AddArc(filler_end, fst::StdArc(0, insertion_end, 0.0, prev));
  ofst->SetFinal(prev, fst::StdArc::Weight::One());
  fst::ArcSort(ofst, fst::StdILabelCompare());
}

std::string LabelCheckerWrapper::Check(char *pcm, int num_samples,
                                       std::vector<std::string> &chars,
                                       float is_penalty, float del_penalty) {
  if (chars.empty()) {
    return model_->Recognize(pcm, num_samples, 1);
  }

  std::vector<int> labels;
  MapToLabel(chars, wfst_symbol_table_, &labels);
  // Prepare FST for alignment decoding
  fst::StdVectorFst align_fst;
  // TODO:  parameter refine later
  CompileAlignFst(labels, wfst_symbol_table_, &align_fst, is_penalty,
                  del_penalty);
  auto local_decode_resource =
      std::make_shared<wenet::DecodeResource>(*(model_->decode_resource()));

  auto decoding_fst = std::make_shared<fst::StdVectorFst>();
  fst::Compose(*ctc_fst_, align_fst, decoding_fst.get());
  local_decode_resource->fst = decoding_fst;
  local_decode_resource->symbol_table = wfst_symbol_table_;
  return model_->RecognizeMayWithLocalFst(pcm, num_samples, 1,
                                          local_decode_resource);
}

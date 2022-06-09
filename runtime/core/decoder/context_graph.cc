// Copyright (c) 2021 Mobvoi Inc (Zhendong Peng)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "decoder/context_graph.h"
#include "decoder/hw_scorer.h"

#include <utility>

#include "fst/determinize.h"

#include "utils/string.h"

namespace wenet {

ContextGraph::ContextGraph(ContextConfig config) : config_(config) {}

// std::vector<std::string> split(std::string const &input) {
//     std::istringstream buffer(input);
//     std::vector<std::string> ret((std::istream_iterator<std::string>(buffer)),
//                                  std::istream_iterator<std::string>());
//     return ret;
// }
int a =1;
std::vector<string> split(const string& str, const string& delim) {
    std::vector<string> res;
    if("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char * strs = new char[str.length() + 1] ; //不要忘了
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p) {
        string s = p; //分割获得的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }

    return res;
}






void ContextGraph::BuildContextGraph(
    const std::vector<std::string>& query_contexts,
    const std::shared_ptr<fst::SymbolTable>& symbol_table) {
  CHECK(symbol_table != nullptr) << "Symbols table should not be nullptr!";
  start_tag_id_ = symbol_table->AddSymbol("<context>");
  end_tag_id_ = symbol_table->AddSymbol("</context>");
  symbol_table_ = symbol_table;
  if (query_contexts.empty()) {
    if (graph_ != nullptr) graph_.reset();
    return;
  }

  std::unique_ptr<fst::StdVectorFst> ofst(new fst::StdVectorFst());
  // State 0 is the start state and the final state.
  int start_state = ofst->AddState();
  ofst->SetStart(start_state);
  ofst->SetFinal(start_state, fst::StdArc::Weight::One());

  LOG(INFO) << "Contexts count size: " << query_contexts.size();
  int count = 0;
  std::vector<std::vector<int>> hwlist;
  for (const auto& context : query_contexts) {
    if (context.size() > config_.max_context_length) {
      LOG(INFO) << "Skip long context: " << context;
      continue;
    }

    if (++count > config_.max_contexts) break;

    std::vector<std::string> words;
    // Split context to words by symbol table, and build the context graph.
    //按照空格，对热词进行拆分
    //bool no_oov = SplitUTF8StringToWords(Trim(context), symbol_table, &words);
    bool no_oov = true ;
    words=split(Trim(context)," ") ;
    //LOG(INFO) <<  words.size() ;
    // for( int i = 0; i < words.size(); i++ ) {
    //     LOG(INFO) <<  words[i] <<" " ;
    //     }
    //LOG(INFO) << endl ;

    if (!no_oov) {
      LOG(WARNING) << "Ignore unknown word found during compilation.";
      continue;
    }
    hwlist.push_back({});
    for (size_t i = 0; i < words.size(); ++i) {
      int word_id = symbol_table_->Find(words[i]);
      hwlist.back().push_back(word_id);
    }
/*
    int prev_state = start_state;
    int next_state = start_state;
    float escape_score = 0;
    for (size_t i = 0; i < words.size(); ++i) {
      int word_id = symbol_table_->Find(words[i]);
      float score = config_.context_score * UTF8StringLength(words[i]);
      next_state = (i < words.size() - 1) ? ofst->AddState() : start_state;
      ofst->AddArc(prev_state,
                   fst::StdArc(word_id, word_id, score, next_state));
      // Add escape arc to clean the previous context score.
      if (i > 0) {
        // ilabel and olabel of the escape arc is 0 (<epsilon>).
        ofst->AddArc(prev_state, fst::StdArc(0, 0, -escape_score, start_state));
      }
      prev_state = next_state;
      escape_score += score;
    }*/
  }

  //std::cout<<"vdebug:"<<hwlist.size()<<std::endl;
  hwscore_=new victor::HWScorer(hwlist,config_.context_score);

/*   std::unique_ptr<fst::StdVectorFst> det_fst(new fst::StdVectorFst());
  fst::Determinize(*ofst, det_fst.get());
  graph_ = std::move(det_fst); */



}

int ContextGraph::GetNextState(int cur_state, int word_id, float* score,
                               bool* is_start_boundary, bool* is_end_boundary) {


  *is_start_boundary = false;
  *is_end_boundary = false;   //I ignore these because they are not accurate anyway
  if(refreash_cache)
  {
      cache_result_=hwscore_->score(cur_state);
      refreash_cache=false;
  }

  int next_state = 0;

  if (cache_result_.outputw_2_scrore_next_states->find(word_id) == cache_result_.outputw_2_scrore_next_states->end())
  {
        next_state = 0;
        *score=cache_result_.return_cost;
  }
  else
  {
    next_state = (*cache_result_.outputw_2_scrore_next_states)[word_id].second;
    *score = (*cache_result_.outputw_2_scrore_next_states)[word_id].first;
  }



  return next_state;
}

}  // namespace wenet

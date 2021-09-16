%module swig_decoders
%{
#include "scorer.h"
#include "ctc_beam_search_decoder.h"
#include "decoder_utils.h"
#include "path_trie.h"
%}

%include "std_vector.i"
%include "std_pair.i"
%include "std_string.i"
%include "path_trie.h"
%import "decoder_utils.h"

namespace std {
    %template(DoubleVector) std::vector<double>;
    %template(IntVector) std::vector<int>;
    %template(StringVector) std::vector<std::string>;
    %template(VectorOfStructVectorDouble) std::vector<std::vector<double> >;
    %template(VectorOfStructVectorInt) std::vector<std::vector<int>>;
    %template(FloatVector) std::vector<float>;
    %template(Pair) std::pair<float, std::vector<int>>;
    %template(PairFloatVectorVector)  std::vector<std::pair<float, std::vector<int>>>;
    %template(PairDoubleVectorVector) std::vector<std::pair<double, std::vector<int>>>;
    %template(PairDoubleVectorVector2) std::vector<std::vector<std::pair<double, std::vector<int>>>>;
    %template(DoubleVector3) std::vector<std::vector<std::vector<double>>>;
    %template(IntVector3) std::vector<std::vector<std::vector<int>>>;
    %template(TrieVector) std::vector<PathTrie*>;
    %template(BoolVector) std::vector<bool>;
}
%template(IntDoublePairCompSecondRev) pair_comp_second_rev<int, double>;
%template(StringDoublePairCompSecondRev) pair_comp_second_rev<std::string, double>;
%template(DoubleStringPairCompFirstRev) pair_comp_first_rev<double, std::string>;

%include "scorer.h"
%include "path_trie.h"
%include "ctc_beam_search_decoder.h"

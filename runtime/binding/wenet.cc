#include <stdlib.h>

#include "wenet.h"
#include "wenet_wrapper/interface/runtime_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif
cParams *wenet_params_init() {
  auto params_ptr =
      new Params("", "", "", 1, "", "", 0.0, 80, 16000, -1, -1, 0.5, 0.5, 0.0,
                 0, 1.0, 10, 10, 1, 1, 1, 7000, 200, 16, 10, 0, true);
  return (cParams *)params_ptr;
}
void wenet_params_free(cParams *cparams) {
  if (cparams == NULL) {
    return;
  }
  Params *pp = (Params *)cparams;
  delete pp;
}

void wenet_params_set_ctc_opts(cParams *pcparm, int blank, int first_beam_size,
                               int second_beam_size) {
  if (pcparm == NULL) {
    return;
  }
  Params *pp = (Params *)pcparm;
  pp->blank = blank;
  pp->first_beam_size = first_beam_size;
  pp->second_beam_size = second_beam_size;
  return;
}
void wenet_params_set_wfst_opts(cParams *pcparm, int max_active, int min_active,
                                int beam, double lattice_beam,
                                double acoustic_scale, double blank_skip_thresh,
                                int nbest) {
  if (pcparm == NULL) {
    return;
  }
  Params *pp = (Params *)pcparm;
  pp->max_active = max_active;
  pp->min_active = min_active;
  pp->beam = beam;
  pp->lattice_beam = lattice_beam;
  pp->acoustic_scale = acoustic_scale;
  pp->blank_skip_thresh = blank_skip_thresh;
  pp->nbest = nbest;
  return;
}

void wenet_params_set_decode_opts(cParams *pcparm, int chunk_size,
                                  double ctc_weight, double rescoring_weight,
                                  double reverse_weight) {
  if (pcparm == NULL) {
    return;
  }
  Params *pp = (Params *)pcparm;
  pp->chunk_size = chunk_size;
  pp->ctc_weight = ctc_weight;
  pp->rescoring_weight = rescoring_weight;
  pp->reverse_weight = reverse_weight;
  return;
}

void wenet_params_set_model_opts(cParams *pcparm, char *model_path,
                                 char *dict_path, int num_threads) {
  if (pcparm == NULL) {
    return;
  }
  Params *pp = (Params *)pcparm;
  pp->model_path = model_path;
  pp->dict_path = dict_path;
  pp->num_threads = num_threads;
  return;
}
void wenet_params_set_feature_pipeline_opts(cParams *pcparm, int num_bins,
                                            int sample_rate) {
  if (pcparm == NULL) {
    return;
  }
  Params *pp = (Params *)pcparm;
  pp->num_bins = num_bins;
  pp->sample_rate = sample_rate;
  return;
}

Model *wenet_init(const cParams *pcparm) {
  if (pcparm == nullptr) {
    return nullptr;
  }
  const Params *pp = (Params *)pcparm;
  auto m = new SimpleAsrModelWrapper(*pp);
  return (Model *)m;
}
void wenet_free(Model *model) {
  if (model == nullptr) {
    return;
  }
  SimpleAsrModelWrapper *m = (SimpleAsrModelWrapper *)model;
  delete m;
}
// caller should call free result
char *wenet_recognize(Model *model, char *data, int n_samples, int nbest) {
  SimpleAsrModelWrapper *m = (SimpleAsrModelWrapper *)model;
  std::string result(std::move(m->Recognize(data, n_samples, nbest)));

  auto cstr = result.c_str();
  char *res = (char *)malloc(result.size() + 1);
  memcpy(res, cstr, result.size());
  res[result.size()] = '\0';

  return res;
}

StreammingDecoder *streamming_decoder_init(Model *model, int nbest,
                                           int continuous_decoding) {
  if (model == nullptr) {
    return nullptr;
  }
  auto m = std::shared_ptr<SimpleAsrModelWrapper>(
      reinterpret_cast<SimpleAsrModelWrapper *>(model));
  auto decoder = new StreammingAsrWrapper(m, nbest, bool(continuous_decoding));
  return (StreammingDecoder *)decoder;
}
void streamming_decoder_free(StreammingDecoder *decoder) {
  if (decoder == nullptr) {
    return;
  }
  StreammingAsrWrapper *d = (StreammingAsrWrapper *)decoder;
  delete d;
  return;
}
void streamming_decoder_accept_waveform(StreammingDecoder *decoder, char *pcm,
                                        int num_samples, int final) {
  if (decoder == nullptr) {
    return;
  }
  StreammingAsrWrapper *d = (StreammingAsrWrapper *)decoder;
  d->AccepAcceptWaveform(pcm, num_samples, final);
  return;
}
// caller responsebile free result
int streamming_decoder_get_instance_result(StreammingDecoder *decoder,
                                           char **text) {
  if (decoder == nullptr || text == nullptr) {
    return 1;
  }
  StreammingAsrWrapper *d = (StreammingAsrWrapper *)decoder;
  std::string result;
  bool is_final = d->GetInstanceResult(result);

  auto cstr = result.c_str();
  char *res = (char *)malloc(result.size() + 1);
  memcpy(res, cstr, result.size());
  res[result.size()] = '\0';
  *text = res;
  return (int)is_final;
}
void streamming_decoder_reset(StreammingDecoder *decoder, int nbest,
                              int continuous_decoding) {
  if (decoder == nullptr) {
    return;
  }
  StreammingAsrWrapper *d = (StreammingAsrWrapper *)decoder;
  d->Reset(nbest, bool(continuous_decoding));
  return;
}

LabelChecker *label_checker_init(Model *model) {
  if (model == nullptr) {
    return nullptr;
  }
  auto m = std::shared_ptr<SimpleAsrModelWrapper>(
      reinterpret_cast<SimpleAsrModelWrapper *>(model));
  auto checker = new LabelCheckerWrapper(m);
  return (LabelChecker *)checker;
}
char *label_checker_check(LabelChecker *checker, char *pcm, int num_samples,
                          char **plabels, int n_labels, float is_penalty,
                          float del_penalty) {
  if (checker == nullptr) {
    return nullptr;
  }
  auto cker = (LabelCheckerWrapper *)checker;
  std::vector<string> labels(plabels, plabels + n_labels);
  auto result = cker->Check(pcm, num_samples, labels, is_penalty, del_penalty);

  auto cstr = result.c_str();
  char *res = (char *)malloc(result.size() + 1);
  memcpy(res, cstr, result.size());

  res[result.size()] = '\0';
  return res;
}

void label_checker_free(LabelChecker *checker) {
  if (checker == nullptr) {
    return;
  }

  LabelCheckerWrapper *cker = (LabelCheckerWrapper *)checker;
  delete cker;
  return;
}

#ifdef __cplusplus
}
#endif

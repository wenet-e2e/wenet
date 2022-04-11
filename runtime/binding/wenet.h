#ifndef _WENET_H
#define _WENET_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cParams cParams;
cParams *wenet_params_init();
void wenet_params_free(cParams *cParams);

// wfst beam search option
void wenet_params_set_wfst_opts(cParams *pcparm, int max_active, int min_active,
                                int beam, double lattice_beam,
                                double acoustic_scale, double blank_skip_thresh,
                                int nbest);

// ctc beam seatch option
void wenet_params_set_ctc_opts(cParams *pcparam, int blank, int first_beam_size,
                               int second_beam_size);
void wenet_params_set_wfst_opts(cParams *pcparm, int max_active, int min_active,
                                int beam, double lattice_beam,
                                double acoustic_scale, double blank_skip_thresh,
                                int nbest);

void wenet_params_set_model_opts(cParams *pcparm, char *model_path,
                                 char *dict_path, int num_threads);

void wenet_params_set_feature_pipeline_opts(cParams *pcparm, int num_bins,
                                            int sample_rate);
void wenet_params_set_decode_opts(cParams *pcparm, int chunk_size,
                                  double ctc_weight, double rescoring_weight,
                                  double reverse_weight);

typedef struct model Model;
Model *wenet_init(const cParams *cparams);
void wenet_free(Model *model);
// caller should cal free
char *wenet_recognize(Model *model, char *data, int n_samples, int nbest);

typedef struct streamming_decoder StreammingDecoder;
StreammingDecoder *streamming_decoder_init(Model *model, int nbest,
                                           int continuous_decoding);
void streamming_decoder_free(StreammingDecoder *decoder);
void streamming_decoder_accept_waveform(StreammingDecoder *decoder, char *pcm,
                                        int num_samples, int final);
// caller responsebile free text
int streamming_decoder_get_instance_result(StreammingDecoder *decoder,
                                           char **text);
void streamming_decoder_reset(StreammingDecoder *decoder, int nbest,
                              int continuous_decoding);

typedef struct label_checker LabelChecker;
LabelChecker *label_checker_init(Model *model);
// caller should cal free
char *label_checker_check(LabelChecker *checker, char *pcm, int num_samples,
                          char **plabels, int n_labels, float is_penalty,
                          float del_penalty);

void label_checker_free(LabelChecker *checker);
#ifdef __cplusplus
}
#endif

#endif

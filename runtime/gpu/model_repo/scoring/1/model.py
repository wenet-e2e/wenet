import triton_python_backend_utils as pb_utils
import numpy as np
import multiprocessing
from swig_decoders import ctc_beam_search_decoder_batch, \
    Scorer, PathTrie, TrieVector, map_batch
import json
import os

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        # Convert Triton types to numpy types
        self.out0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        # Get INPUT configuration
        batch_log_probs = pb_utils.get_input_config_by_name(
            model_config, "batch_log_probs")
        self.beam_size = batch_log_probs['dims'][-1]

        encoder_config = pb_utils.get_input_config_by_name(
            model_config, "encoder_out")
        self.data_type = pb_utils.triton_string_to_numpy(
            encoder_config['data_type'])

        self.feature_size = encoder_config['dims'][-1]

        self.lm = None
        self.init_ctc_rescore(self.model_config['parameters'])
        print('Initialized Rescoring!')

    def init_ctc_rescore(self, parameters):
        num_processes = multiprocessing.cpu_count()
        cutoff_prob = 0.9999
        blank_id = 0
        alpha = 2.0
        beta = 1.0
        bidecoder = 0
        lm_path, vocab_path = None, None
        for li in parameters.items():
            key, value = li
            value = value["string_value"]
            if key == "num_processes":
                num_processes = int(value)
            elif key == "blank_id":
                blank_id = int(value)
            elif key == "cutoff_prob":
                cutoff_prob = float(value)
            elif key == "lm_path":
                lm_path = value
            elif key == "alpha":
                alpha = float(value)
            elif key == "beta":
                beta = float(value)
            elif key == "vocabulary":
                vocab_path = value
            elif key == "bidecoder":
                bidecoder = int(value)

        self.num_processes = num_processes
        self.cutoff_prob = cutoff_prob
        self.blank_id = blank_id
        _, vocab = self.load_vocab(vocab_path)
        if lm_path and os.path.exists(lm_path):
            self.lm = Scorer(alpha, beta, lm_path, vocab)
            print("Successfully load language model!")
        self.vocabulary = vocab
        self.bidecoder = bidecoder
        sos = eos = len(vocab) - 1
        self.sos = sos
        self.eos = eos

    def load_vocab(self, vocab_file):
        """
        load lang_char.txt
        """
        id2vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                char, id = line.split()
                id2vocab[int(id)] = char
        vocab = [0] * len(id2vocab)
        for id, char in id2vocab.items():
            vocab[id] = char
        return id2vocab, vocab

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.

        batch_encoder_out, batch_encoder_lens = [], []
        batch_log_probs, batch_log_probs_idx = [], []
        batch_count = []
        batch_root = TrieVector()
        batch_start = []
        root_dict = {}

        encoder_max_len = 0
        hyps_max_len = 0
        total = 0
        for request in requests:
            # Perform inference on the request and append it to responses list...
            in_0 = pb_utils.get_input_tensor_by_name(request, "encoder_out")
            in_1 = pb_utils.get_input_tensor_by_name(request, "encoder_out_lens")
            in_2 = pb_utils.get_input_tensor_by_name(request, "batch_log_probs")
            in_3 = pb_utils.get_input_tensor_by_name(request, "batch_log_probs_idx")

            batch_encoder_out.append(in_0.as_numpy())
            encoder_max_len = max(encoder_max_len, batch_encoder_out[-1].shape[1])

            cur_b_lens = in_1.as_numpy()
            batch_encoder_lens.append(cur_b_lens)
            cur_batch = cur_b_lens.shape[0]
            batch_count.append(cur_batch)

            cur_b_log_probs = in_2.as_numpy()
            cur_b_log_probs_idx = in_3.as_numpy()
            for i in range(cur_batch):
                cur_len = cur_b_lens[i]
                cur_probs = cur_b_log_probs[i][0:cur_len, :].tolist()  # T X Beam
                cur_idx = cur_b_log_probs_idx[i][0:cur_len, :].tolist()  # T x Beam
                batch_log_probs.append(cur_probs)
                batch_log_probs_idx.append(cur_idx)
                root_dict[total] = PathTrie()
                batch_root.append(root_dict[total])
                batch_start.append(True)
                total += 1

        score_hyps = ctc_beam_search_decoder_batch(batch_log_probs,
                                                   batch_log_probs_idx,
                                                   batch_root,
                                                   batch_start,
                                                   self.beam_size,
                                                   min(total, self.num_processes),
                                                   blank_id=self.blank_id,
                                                   space_id=-2,
                                                   cutoff_prob=self.cutoff_prob,
                                                   ext_scorer=self.lm)
        all_hyps = []
        all_ctc_score = []
        max_seq_len = 0
        for seq_cand in score_hyps:
            # if candidates less than beam size
            if len(seq_cand) != self.beam_size:
                seq_cand = list(seq_cand)
                seq_cand += (self.beam_size - len(seq_cand)) * [(-float("INF"), (0,))]

            for score, hyps in seq_cand:
                all_hyps.append(list(hyps))
                all_ctc_score.append(score)
                max_seq_len = max(len(hyps), max_seq_len)

        beam_size = self.beam_size
        feature_size = self.feature_size
        hyps_max_len = max_seq_len + 2
        in_ctc_score = np.zeros((total, beam_size), dtype=self.data_type)
        in_hyps_pad_sos_eos = np.ones(
            (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos
        if self.bidecoder:
            in_r_hyps_pad_sos_eos = np.ones(
                (total, beam_size, hyps_max_len), dtype=np.int64) * self.eos

        in_hyps_lens_sos = np.ones((total, beam_size), dtype=np.int32)

        in_encoder_out = np.zeros((total, encoder_max_len, feature_size),
                                  dtype=self.data_type)
        in_encoder_out_lens = np.zeros(total, dtype=np.int32)
        st = 0
        for b in batch_count:
            t = batch_encoder_out.pop(0)
            in_encoder_out[st:st + b, 0:t.shape[1]] = t
            in_encoder_out_lens[st:st + b] = batch_encoder_lens.pop(0)
            for i in range(b):
                for j in range(beam_size):
                    cur_hyp = all_hyps.pop(0)
                    cur_len = len(cur_hyp) + 2
                    in_hyp = [self.sos] + cur_hyp + [self.eos]
                    in_hyps_pad_sos_eos[st + i][j][0:cur_len] = in_hyp
                    in_hyps_lens_sos[st + i][j] = cur_len - 1
                    if self.bidecoder:
                        r_in_hyp = [self.sos] + cur_hyp[::-1] + [self.eos]
                        in_r_hyps_pad_sos_eos[st + i][j][0:cur_len] = r_in_hyp
                    in_ctc_score[st + i][j] = all_ctc_score.pop(0)
            st += b
        in_encoder_out_lens = np.expand_dims(in_encoder_out_lens, axis=1)
        in_tensor_0 = pb_utils.Tensor("encoder_out", in_encoder_out)
        in_tensor_1 = pb_utils.Tensor("encoder_out_lens", in_encoder_out_lens)
        in_tensor_2 = pb_utils.Tensor("hyps_pad_sos_eos", in_hyps_pad_sos_eos)
        in_tensor_3 = pb_utils.Tensor("hyps_lens_sos", in_hyps_lens_sos)
        input_tensors = [in_tensor_0, in_tensor_1, in_tensor_2, in_tensor_3]
        if self.bidecoder:
            in_tensor_4 = pb_utils.Tensor("r_hyps_pad_sos_eos", in_r_hyps_pad_sos_eos)
            input_tensors.append(in_tensor_4)
        in_tensor_5 = pb_utils.Tensor("ctc_score", in_ctc_score)
        input_tensors.append(in_tensor_5)

        inference_request = pb_utils.InferenceRequest(
            model_name='decoder',
            requested_output_names=['best_index'],
            inputs=input_tensors)

        inference_response = inference_request.exec()
        if inference_response.has_error():
            raise pb_utils.TritonModelException(inference_response.error().message())
        else:
            # Extract the output tensors from the inference response.
            best_index = pb_utils.get_output_tensor_by_name(inference_response,
                                                            'best_index')
            best_index = best_index.as_numpy()
            hyps = []
            idx = 0
            for cands, cand_lens in zip(in_hyps_pad_sos_eos, in_hyps_lens_sos):
                best_idx = best_index[idx][0]
                best_cand_len = cand_lens[best_idx] - 1  # remove sos
                best_cand = cands[best_idx][1: 1 + best_cand_len].tolist()
                hyps.append(best_cand)
                idx += 1

            hyps = map_batch(hyps, self.vocabulary,
                             min(multiprocessing.cpu_count(), len(in_ctc_score)))
            st = 0
            for b in batch_count:
                sents = np.array(hyps[st:st + b])
                out0 = pb_utils.Tensor("OUTPUT0", sents.astype(self.out0_dtype))
                inference_response = pb_utils.InferenceResponse(output_tensors=[out0])
                responses.append(inference_response)
                st += b
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

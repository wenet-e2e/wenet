#!/usr/bin/env bash

# ./tools/decode.sh --nj $nj \
#   --beam 15.0 --lattice_beam 7.5 --max_active 7000 \
#   --blank_skip_thresh 0.98 --ctc_weight 0.5 --rescoring_weight 1.0 \
#   --chunk_size $chunk_size \
#   --fst_path data/lang_test/TLG.fst \
#   ${test_scp} ${test_text} ${jit_path} \
#   data/lang_test/words.txt $dir/lm_with_runtime_${name}

wfst_decode_opts="--fst_path TLG.fst"
wfst_decode_opts="$wfst_decode_opts --beam 15.0"
wfst_decode_opts="$wfst_decode_opts --lattice_beam 7.5"
wfst_decode_opts="$wfst_decode_opts --max_active 7000"
wfst_decode_opts="$wfst_decode_opts --min_active 200"
wfst_decode_opts="$wfst_decode_opts --acoustic_scale 1.0"
wfst_decode_opts="$wfst_decode_opts --blank_skip_thresh 0.98"

./decoder_main \
     --rescoring_weight 0.0 \
     --ctc_weight 1.0 \
     --reverse_weight 0.0 \
     --chunk_size -1 \
     --wav_scp data/wav.scp \
     --model_path final_static_quant.zip \
     --dict_path words.txt \
     --result res.text

     # $wfst_decode_opts \

./decoder_main \
     --rescoring_weight 1.0 \
     --ctc_weight 0.5 \
     --reverse_weight 0.0 \
     --chunk_size -1 \
     --wav_scp data/wav.scp \
     --model_path final_quant.zip \
     --dict_path words.txt \
     $wfst_decode_opts \
     --result res.text

./decoder_main \
     --rescoring_weight 1.0 \
     --ctc_weight 0.5 \
     --reverse_weight 0.0 \
     --chunk_size -1 \
     --wav_scp data/wav.scp \
     --model_path final.zip \
     --dict_path words.txt \
     $wfst_decode_opts \
     --result res.text

## Fintune Paraformer
### 1 Downlaod model and cmvn and config
```bash
wget --user-agent="Mozilla/5.0" -c "https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=v1.0.4&FilePath=model.pb" -O model.pt
wget --user-agent="Mozilla/5.0" -c "https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=v1.0.4&FilePath=am.mvn" -O am.mvn
wget --user-agent="Mozilla/5.0" -c "https://www.modelscope.cn/api/v1/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/repo?Revision=v1.0.4&FilePath=config.yaml" -O config.yaml
```
### 2 Convert config to wenet's style
```
PROJECT_ROOT=...
OUTPUT_DIR=...
export PYTHONPATH="${PROJECT_ROOT}"
python3 wenet/paraformer/convert_paraformer_to_wenet_config.py \
            --paraformer_config config.yaml \
            --paraformer_cmvn am.mvn \
            --output_dir  ${OUTPUT_DIR}
```
### 4 (Optioanl) Convert Ali model to wenet jit to enable wenet cli
```bash
PROJECT_ROOT=...
OUTPUT_DIR=...
export PYTHONPATH="${PROJECT_ROOT}"
python3 wenet/bin/export_jit.py \
        --config ${OUTPUT_DIR}/train.yaml \
        --checkpoint model.pt \
        --output_file ${OUTPUT_DIR}/final.zip
```
## 5 fintune 
```bash
## run train.py , set train.yaml as config, and set model.pt as checkpoint
```

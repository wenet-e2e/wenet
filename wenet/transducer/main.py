import os

import torch
import yaml
from wenet.transducer.transducer import init_transducer_asr_model

with open("test.yaml", 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)

configs['cmvn_file'] = None

model = init_transducer_asr_model(configs)

print(model)

script_model = torch.jit.script(model)
script_model.save(os.path.join(".", 'init.zip'))

input = torch.ones(2, 100, 80)
input_length = torch.ones(2, dtype=torch.int) * 100

text = torch.ones(2, 3, dtype=torch.int) * 3
text_length = torch.ones(2, dtype=torch.int) * 3

print(model(input, input_length, text, text_length))

model.greedy_search(input[:1, :, :], input_length[:1])

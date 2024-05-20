from dataclasses import dataclass
from typing import Optional


@dataclass
class Template:
    # one turn :{system_format}{user_text_format}{user_audio_format}{assistant_format}
    # multi turns:
    #    {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
    system: Optional[str]

    prefix_user: str
    suffix_user: str
    assistant: str

    bos: str
    eos: str


audio_gemma = Template(
    '',
    '<start_of_turn>user\n{content}\n',
    '<end_of_turn>\n<start_of_turn>model\n',
    '{content}<end_of_turn>\n',
    '<bos>',
    '<eos>',
)

audio_llama3 = Template(
    '<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    '<|start_header_id|>user<|end_header_id|>\n\n{content}\n\n',
    '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    '{content}<|eot_id|>',
    '<|begin_of_text|>',
    '<|end_of_text|>',
)



WENET_LLM_Template = {
    "audio_gemma": audio_gemma,
    'audio_llama3': audio_llama3,
}

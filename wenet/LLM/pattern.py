from dataclasses import dataclass
from typing import Optional


@dataclass
class Pattern:
    # one turn :{system_format}{user_format}{assistant_format}
    # multi turns:
    #    {system_format}{user_format}{assistant_format}{user_format}{assistant_format}...
    system: Optional[str]
    user: str
    assistant: str

    bos: str
    eos: str


gemma = Pattern(
    '',
    '<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n',
    '{content}<end_of_turn>\n',
    '<bos>',
    '<eos>',
)

llama3 = Pattern(
    '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>',
    '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n',
    '{content}<|eot_id|>',
    '<|begin_of_text|>',
    '<|end_of_text|>',
)
WENET_LLM_PATTERN = {"gemma": gemma, 'llama3': llama3}

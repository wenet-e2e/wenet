# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gradio as gr
import wenet

# TODO: add hotword
chs_model = wenet.load_model('chinese')
en_model = wenet.load_model('english')


def recognition(audio, lang='CN'):
    if audio is None:
        return "Input Error! Please enter one audio!"
    # NOTE: model supports 16k sample_rate
    if lang == 'CN':
        ans = chs_model.transcribe(audio)
    elif lang == 'EN':
        ans = en_model.transcribe(audio)
    else:
        return "ERROR! Please select a language!"

    if ans is None:
        return "ERROR! No text output! Please try again!"
    txt = ans['text']
    return txt


# input
inputs = [
    gr.inputs.Audio(source="microphone", type="filepath", label='Input audio'),
    gr.Radio(['EN', 'CN'], label='Language')
]

output = gr.outputs.Textbox(label="Output Text")

text = "Speech Recognition in WeNet | 基于 WeNet 的语音识别"

# description
description = (
    "Wenet Demo ! This is a speech recognition demo that supports Mandarin and English !"  # noqa
)

article = (
    "<p style='text-align: center'>"
    "<a href='https://github.com/wenet-e2e/wenet' target='_blank'>Github: Learn more about WeNet</a>"  # noqa
    "</p>")

interface = gr.Interface(
    fn=recognition,
    inputs=inputs,
    outputs=output,
    title=text,
    description=description,
    article=article,
    theme='huggingface',
)

interface.launch(enable_queue=True)

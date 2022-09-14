import json
import gradio as gr
import numpy as np
import wenetruntime as wenet

wenet.set_log_level(2)
decoder = wenet.Decoder(lang='chs')

def recognition(audio):
    sr, y = audio
    assert sr in [48000, 16000]
    if sr == 48000:  # Optional resample to 16000
        y = (y / max(np.max(y), 1) * 32767)[::3].astype("int16")
    ans = decoder.decode(y.tobytes(), True)
    return json.loads(ans)

text = "Speech Recognition in WeNet | 基于 WeNet 的语音识别"
gr.Interface(recognition, inputs="mic", outputs="json",
             description=text).launch()

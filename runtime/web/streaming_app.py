import json
import gradio as gr
import numpy as np
import torch
import wenetruntime as wenet

torch.manual_seed(777)  # for lint

wenet.set_log_level(2)
decoder = wenet.Decoder(lang='chs')

def recognition(audio):
    sr, y = audio
    assert sr in [48000, 16000]
    if sr == 48000:  # Optional resample to 16000
        y = (y / max(np.max(y), 1) * 32767)[::3].astype("int16")
    ans = decoder.decode(y.tobytes(), False)
    if ans == "":
        return ans
    ans = json.loads(ans)
    text = ans["nbest"][0]["sentence"]
    return text

print("\n===> Loading the ASR model ...")
print("===> Warming up by 100 randomly-generated audios ... Please wait ...\n")
for i in range(100):
    audio_len = np.random.randint(16000 * 3, 16000 * 10)  # 3~10s
    audio = np.random.randint(-32768, 32768, size=audio_len, dtype=np.int16)
    ans = decoder.decode(audio.tobytes(), True)
    print("Processed the {}-th audio.".format(i + 1))

with gr.Blocks() as demo:
    gr.Markdown("Streaming Speech Recognition in WeNet | 基于 WeNet 的流式语音识别")
    with gr.Row():
        inputs = gr.Microphone(streaming=True)
        outputs = gr.Textbox(label="Output:")
    inputs.stream(fn=recognition, inputs=inputs, outputs=outputs,
                  show_progress="hidden")
demo.launch()

import subprocess
import torch
from scipy.io.wavfile import read as read_forpipe
import io
import numpy as np

def _process_pipe_input(
    pipestr: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    channels_first: bool = True,
    speed: float = 1.0
):
    """
    Load waveform data from pipe command. used in function _extract_feature()

    The input is a pipe command, usually the format can be processed by Kaldi. 

    Args:
        pipestr: a shell pipe string (like "cat ./1.wav | sox -t wav - -r 16000 -c 1 -b 16 -t wav - |")
        frame_offset: the begin frame index of the segment wav
        num_frames: the frame length of the segment wav (default=-1, means no segment)
        channels_first: if true, the first dim of waveform means channels number
        speed: 
    Returns:
        (waveform, sample_rate)

    """
    if speed == 1.0:
        input_commend=pipestr+" cat - "
    else:
        input_commend=pipestr+" cat - |"+" sox -t wav - -t wav - speed " + format(speed, '.1f')
        # input_commend=pipestr+" cat - |"+" sox -t wav - -t wav - stretch " + format(speed, '.1f')
        # here will output a warning "sox WARN wav: Length in output .wav header will be wrong since can't seek to fix it"
    p=subprocess.Popen(input_commend,shell=True,stdout=subprocess.PIPE)
    out=p.stdout.read()
    # here, input_wav is a bytes object representing the wav object
    sample_rate, waveform = read_forpipe(io.BytesIO(out))
    waveform=np.array(waveform)
    if(len(waveform.shape)==1):
        waveform=np.expand_dims(waveform,axis=0)
    else:
        waveform=waveform.T
    
    if num_frames != -1:
        waveform=waveform[:,frame_offset:frame_offset+num_frames]
    if not channels_first:
        waveform=waveform.transpose(0,1)

    return torch.Tensor(waveform),sample_rate

if __name__ == '__main__':
    pipestr="cat ./SSB00010010.wav | sox -t wav - -r 16000 -c 1 -b 16 -t wav - |"
    waveform1, sample_rate1 = _process_pipe_input(pipestr,speed=1.0)
    waveform2, sample_rate2 = _process_pipe_input(pipestr,speed=1.1)
    print("")
    
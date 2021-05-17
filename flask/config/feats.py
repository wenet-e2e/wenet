#coding=utf-8

class FeatConfig(object):
    def __init__(self):
        self.frame_shift_ms = 10
        self.frame_length_ms = 25
        self.sample_rate = 8000
        self.num_mels = 40
        self.n_mfcc = 40
        self.fmax = 4000
        self.fmin = 20
        self.energy_floor = 0.1
        self.volume_scale = 1.0
        self.use_pitch = False
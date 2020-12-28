import sys
import random
import math

import torchaudio
import torch

# [TODO] time related distortion

def db2amp(db):
    return pow(10, db / 20)

def amp2db(amp):
    return 20 * math.log10(amp)

#  db域上的通用的多项式变换
def make_poly_distortion(distortion_conf):
    # x = a * x^m * (1-x)^n + x
    # a = [1, m*n]
    # m = [1, 4]
    # n = [1, 4]
    a = distortion_conf['a']
    m = distortion_conf['m']
    n = distortion_conf['n']

    def poly_distortion(x):
        # print(type(x))
        abs_x = abs(x)
        if abs_x < 0.000001:
            x = x
        else:
            db_norm = amp2db(abs_x) / 100 + 1
            if db_norm < 0:
                db_norm = 0
            db_norm = a * pow(db_norm, m) * pow((1 - db_norm), n) + db_norm
            if db_norm > 1:
                db_norm = 1
            db = (db_norm - 1) * 100
            amp = db2amp(db)
            if amp >= 0.9997:
                amp = 0.9997
            if x > 0:
                x = amp
            else:
                x = -amp
        return x
    return poly_distortion

def dither(x):
    return x + 1.0

# 二次函数
def make_quad_distortion():
    return make_poly_distortion({'a' : 1, 'm' : 1, 'n' : 1})


def make_amp_mask(db_mask=[(-110, -95),
                           (-90, -80), (-65, -60), (-50, -30), (-15, 0)]):
    amp_mask = [(db2amp(db[0]), db2amp(db[1])) for db in db_mask]
    return amp_mask

default_mask = make_amp_mask()

# gen amp slots in -100db ~ 0db.
def make_max_distortion(conf):
    max_db = conf['max_db']
    if max_db:
        max_amp = db2amp(max_db)  # < 0.997
    else:
        max_amp = 0.997

    def max_distortion(x):
        if x > 0:
            x = max_amp
        elif x < 0:
            x = -max_amp
        else:
            x = 0.0
        return x
    return max_distortion

def generate_amp_mask(mask_num):
    a = [0] * 2 * mask_num
    a[0] = 0
    m = []
    for i in range(1, 2 * mask_num):
        a[i] = a[i - 1] + random.uniform(0.5, 1)
    max_val = a[2 * mask_num - 1]
    for i in range(0, mask_num):
        l = ((a[2 * i] - max_val) / max_val) * 100
        r = ((a[2 * i + 1] - max_val) / max_val) * 100
        m.append((l, r))
    return make_amp_mask(m)

# gen amp slots in -100db ~ 0db.
def make_fence_distortion(conf):
    mask_number = conf['mask_number']
    max_db = conf['max_db']
    max_amp = db2amp(max_db)  # 0.997
    if mask_number <= 0 :
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    def fence_distortion(x):
        is_in_mask = False
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return max_amp
            if not is_in_mask:
                return 0.0
        return x

    return fence_distortion



# gen amp slots in -100db ~ 0db.
def make_jag_distortion(conf):
    mask_number = conf['mask_number']
    if mask_number <= 0 :
        positive_mask = default_mask
        negative_mask = make_amp_mask([(-50, 0)])
    else:
        positive_mask = generate_amp_mask(mask_number)
        negative_mask = generate_amp_mask(mask_number)

    def jag_distortion(x):
        is_in_mask = False
        if x > 0:
            for mask in positive_mask:
                if x >= mask[0] and x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        elif x < 0:
            abs_x = abs(x)
            for mask in negative_mask:
                if abs_x >= mask[0] and abs_x <= mask[1]:
                    is_in_mask = True
                    return x
            if not is_in_mask:
                return 0.0
        return x

    return jag_distortion

def fast_fence_distortion(x, positive_mask=default_mask, negative_mask=make_amp_mask([(-50, 0)])):
    is_in_mask = False
    if x > 0:
        for mask in positive_mask:
            if x >= mask[0] and x <= mask[1]:
                is_in_mask = True
                return 0.997
        if not is_in_mask:
            return 0.0
    elif x < 0:
        abs_x = abs(x)
        for mask in negative_mask:
            if abs_x >= mask[0] and abs_x <= mask[1]:
                is_in_mask = True
                return 0.997
        if not is_in_mask:
            return 0.0
    return x

def fast_jag_distortion(x,
                        positive_mask=default_mask,
                        negative_mask=default_mask
                        ):
    is_in_mask = False
    if x > 0:
        for slot in positive_mask:
            if x >= slot[0] and x <= slot[1]:
                is_in_mask = True
                return x
        if not is_in_mask:
            return 0.0
    elif x < 0:
        abs_x = abs(x)
        for slot in negative_mask:
            if abs_x >= slot[0] and abs_x <= slot[1]:
                is_in_mask = True
                return x
        if not is_in_mask:
            return 0.0
    return x

# db could be positive or negative
# gain 20 db means amp  = amp * 10
# gain -20 db mean amp = amp / 10
def make_gain_db(conf):
    db = conf['db']

    def gain_db(x):
        return min(0.997, x * pow(10, db / 20))
    return gain_db

def distort(x, func, rate=0.8):
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            x[0][i] = func(float(x[0][i]))
    return x

def distort_chain(x, funcs, rate=0.8):
    for i in range(0, x.shape[1]):
        a = random.uniform(0, 1)
        if a < rate:
            for func in funcs:
                x[0][i] = func(float(x[0][i]))
    return x


def distort_wav_conf_and_save(distort_type, distort_conf, rate, wav_in, wav_out):
    x, sr = torchaudio.load(wav_in)
    x = x.detach().numpy()
    out = distort_wav_conf(x, distort_type, distort_conf, rate)
    torchaudio.save(wav_out, torch.from_numpy(out), sr)


# x is numpy
def distort_wav_conf(x, distort_type, distort_conf, rate=0.1):
    if distort_type == 'gain_db':
        gain_db = make_gain_db(distort_conf)
        x = distort(x, gain_db)
    elif distort_type == 'max_distortion':
        max_distortion = make_max_distortion(distort_conf)
        x = distort(x, max_distortion, rate=rate)
    elif distort_type == 'fence_distortion':
        fence_distortion = make_fence_distortion(distort_conf)
        x = distort(x, fence_distortion, rate=rate)
    elif distort_type == 'jag_distortion':
        jag_distortion = make_jag_distortion(distort_conf)
        x = distort(x, jag_distortion, rate=rate)
    elif distort_type == 'poly_distortion':
        poly_distortion = make_poly_distortion(distort_conf)
        x = distort(x, poly_distortion, rate=rate)
    elif distort_type == 'quad_distortion':
        quad_distortion = make_quad_distortion()
        x = distort(x, quad_distortion, rate=rate)
    elif distort_type == 'none_distortion':
        pass
    else:
        print('unsupport type')
    return x

if __name__ == "__main__":
    distort_type = sys.argv[1]
    wav_in = sys.argv[2]
    wav_out = sys.argv[3]
    conf = None
    rate = 0.1
    if distort_type == 'new_jag_distortion':
        conf = {'mask_number' : 4}
    elif distort_type == 'new_fence_distortion':
        conf = {'mask_number' : 1, 'max_db' : -30}
    elif distort_type == 'poly_distortion':
        conf = {'a' : 4, 'm' : 2, "n" : 2}
    distort_wav_conf_and_save(distort_type, conf, rate, wav_in, wav_out)

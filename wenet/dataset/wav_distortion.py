import torchaudio
import torch
import numpy
import sys
import random
import math

# todo 时间相关的扰动。
# db = -110
# pow(10, db/20)

def db2amp(db):
    return pow(10, db / 20)

def amp2db(amp):
    return 20 * math.log10(amp)

# 8 anchor
# todo 支持任意个数anchor
def make_anchors(db_anchors=[-110, -95,
                            -90, -80, -65, -60, -50, -30, -15, 0]):
    amp_anchors = [ db2amp(db) for db in db_anchors]
    return amp_anchors

##  db域上的通用的多项式变换
def make_poly_distortion(a,m,n):
    ## x = a * x^m * (1-x)^n + x
    ## a = [1, m*n]
    ## m = [1,4]
    ## n = [1,4]
    def poly_distortion(x):
        #print(type(x))
        abs_x = abs(x)
        if abs_x < 0.000001:
            x = x
        else:
            db_norm = amp2db(abs_x) / 100 + 1
            if db_norm < 0:
                db_norm=0
            db_norm = a*pow(db_norm,m)*pow((1-db_norm),n) + db_norm
            if db_norm >1:
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

## 二次函数
def make_quad_distortion():
    return make_poly_distortion(1, 1, 1)

def make_double_jag_distortion(amp_anchors=make_anchors([-110, -95,
                                            -90, -80, -65, -60, -50, -30, -15, 0])):
    ## 正负采样点锯齿
    def double_jag_distortion(x):
        ahr=amp_anchors
        abs_x = abs(x)
        #abs_x = abs(x)
        if (abs_x >= ahr[0] and abs_x <= ahr[1]) or (abs_x >= ahr[2] and abs_x <= ahr[3]) or (abs_x >= ahr[4] and abs_x <= ahr[5]) or (abs_x >= ahr[6] and abs_x <= ahr[7]) or (abs_x >= ahr[8] and abs_x <= ahr[9]):
            pass
        else:
            x = 0.0
        return x
    return double_jag_distortion

## 正采样点锯齿
def jag_distortion(x,
                        positive_amp_anchors=make_anchors([-110, -95,
                                            -90, -80, -65, -60, -50, -30, -15, 0]),
                        negtive_amp_anchor=db2amp(-50)):
    ahr=positive_amp_anchors
    #print(ahr)
    if x > 0:
        if (x >= ahr[0] and x <= ahr[1]) or (x >= ahr[2] and x <= ahr[3]) or (x >= ahr[4] and x <= ahr[5]) or (x >= ahr[6] and x <= ahr[7]) or (x >= ahr[8] and x <= ahr[9]):
            x = x
        else:
            x = 0.0
    elif x < 0:
        if abs(x) > negtive_amp_anchor:
            x = x
        else:
            x = 0.0
    else:
        x = 0.0
    return x

## 无限失真
def infinite_distortion(x,
                        positive_amp_anchors=make_anchors([-110, -95,
                                            -90, -80, -65, -60, -50, -30, -15, 0]),
                        negtive_amp_anchor=db2amp(-50)):
    ahr=positive_amp_anchors
    #print(ahr)
    if x > 0:
        if (x >= ahr[0] and x <= ahr[1]) or (x >= ahr[2] and x <= ahr[3]) or (x >= ahr[4] and x <= ahr[5]) or (x >= ahr[6] and x <= ahr[7]) or (x >= ahr[8] and x <= ahr[9]):
            x = 0.9997
        else:
            x = 0.0
    elif x < 0:
        if abs(x) > negtive_amp_anchor:
            x = -0.9997
        else:
            x = 0.0
    else:
        x = 0.0
    return x

##  负采样点无限失真
def negative_infinite_distortion(x,
                        negtive_amp_anchors=make_anchors([-110, -95,
                                            -90, -80, -65, -60, -50, -30, -15, 0]),
                        positive_amp_anchor=db2amp(-50)):
    ahr=negtive_amp_anchors
    #print(ahr)
    if x < 0:
        x=abs(x)
        if (x >= ahr[0] and x <= ahr[1]) or (x >= ahr[2] and x <= ahr[3]) or (x >= ahr[4] and x <= ahr[5]) or (x >= ahr[6] and x <= ahr[7]) or (x >= ahr[8] and x <= ahr[9]):
            x = -0.9997
        else:
            x = 0.0
    elif x > 0:
        if abs(x) > positive_amp_anchor:
            x = 0.9997
        else:
            x = 0.0
    else:
        x = 0.0
    return x

## 最大阵痛
def max_pain(x):
    if x > 0:
        x = 0.9997
    elif x < 0:
        x = -0.9997
    else:
        x = 0.0
    return x

## db could be positive or negative 
def make_gain_db(db):
    def gain_db(x):
        return x * pow(10, db / 20)
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

def distort_wav_and_save(distort_type, wav_in, wav_out):
    x, sr = torchaudio.load(wav_in)
    x = x.detach().numpy()
    out = distort_wav(distort_type, x)
    torchaudio.save(wav_out, torch.from_numpy(out), sr)

gain_db_n20 = make_gain_db(-20)
gain_db_n30 = make_gain_db(-30)
# x is numpy
def distort_wav(distort_type, x, rate=0.1):
    #
    if distort_type == 'gain_db_n20':
        x = distort(x, gain_db_n20)
    elif distort_type == 'max_pain':
        x = distort_chain(x, [max_pain, gain_db_n30], rate=rate)
    elif distort_type == 'infinite_distortion':
        x = distort_chain(x, [infinite_distortion, gain_db_n20],rate=rate)
    elif distort_type == 'negative_infinite_distortion':
        x = distort_chain(x, [negative_infinite_distortion, gain_db_n20],rate=rate)
    elif distort_type == 'jag_distortion':
        x = distort(x, jag_distortion,rate=rate)
    elif distort_type == 'double_jag_distortion':
        anchors = make_anchors([-110, -95,-90, -80, -65, -60, -50, -30, -15, 0])
        #anchors = make_anchors([-110, -95])
        double_jag_distortion=make_double_jag_distortion(anchors)
        x = distort(x, double_jag_distortion,rate=rate)
    elif distort_type == 'quad_distortion':
        quad_distortion = make_quad_distortion()
        x = distort(x, quad_distortion,rate=rate)
    elif distort_type == 'poly_distortion':
        poly_distortion = make_poly_distortion(8,2,2)
        x = distort(x, poly_distortion, rate=rate)
    elif distort_type == 'none_distortion':
        pass
    else:
        print('unsupport type')
    return x

if __name__ == "__main__":
    distort_type = sys.argv[1]
    wav_in = sys.argv[2]
    wav_out=sys.argv[3]
    distort_wav_and_save(distort_type, wav_in, wav_out)
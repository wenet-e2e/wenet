// Copyright (c) 2016 HR

#ifndef WENET_FRONTEND_FFT_H_
#define WENET_FRONTEND_FFT_H_

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

namespace wenet {

// Fast Fourier Transform

// #define M_PI 3.141592653589793238462643383279502

int fft(float* x, float* y, int n);

}  // namespace wenet

#endif  // WENET_FRONTEND_FFT_H_



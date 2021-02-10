// Copyright (c) 2016 HR

#ifndef FRONTEND_FFT_H_
#define FRONTEND_FFT_H_

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace wenet {

// Fast Fourier Transform

int fft(float* x, float* y, int n);

}  // namespace wenet

#endif  // FRONTEND_FFT_H_

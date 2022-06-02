// Copyright 2022 Horizon Inc. All Rights Reserved.
// Author: zhendong.peng@horizon.ai (Zhendong Peng)

#include <signal.h>

#include "portaudio.h"  // NOLINT

#include "decoder/params.h"
#include "utils/log.h"

DEFINE_int32(interval, 500, "callback and decode pcm data interval in ms");

int exiting = 0;
std::shared_ptr<wenet::FeaturePipeline> feature_pipeline;

void sig_routine(int dunno) {
  if (dunno == SIGINT) {
    exiting = 1;
  }
}

static int recordCallback(const void* input, void* output,
                          unsigned long framesCount,  // NOLINT
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags, void* userData) {
  const auto* pcm_data = static_cast<const int16_t*>(input);
  feature_pipeline->AcceptWaveform(pcm_data, framesCount);

  if (exiting) {
    LOG(INFO) << "Exiting loop";
    return paComplete;
  } else {
    return paContinue;
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);
  signal(SIGINT, sig_routine);

  auto decode_config = wenet::InitDecodeOptionsFromFlags();
  auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
  auto decode_resource = wenet::InitDecodeResourceFromFlags();
  feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
  wenet::AsrDecoder decoder(feature_pipeline, decode_resource, *decode_config);

  Pa_Initialize();
  PaStreamParameters params;
  params.device = Pa_GetDefaultInputDevice();
  if (params.device == paNoDevice) {
    LOG(FATAL) << "Error: No default input device.";
  }
  params.channelCount = 1;
  params.sampleFormat = paInt16;
  params.suggestedLatency =
      Pa_GetDeviceInfo(params.device)->defaultLowInputLatency;
  params.hostApiSpecificStreamInfo = NULL;

  PaStream* stream;
  int frames_per_buffer = FLAGS_sample_rate / 1000 * FLAGS_interval;
  Pa_OpenStream(&stream, &params, NULL, FLAGS_sample_rate, frames_per_buffer,
                paClipOff, recordCallback, NULL);
  Pa_StartStream(stream);

  LOG(INFO) << "=== Now recording!! Please speak into the microphone. ===";
  std::string final_result;
  while (Pa_IsStreamActive(stream)) {
    Pa_Sleep(FLAGS_interval);
    wenet::DecodeState state = decoder.Decode();
    if (decoder.DecodedSomething()) {
      LOG(INFO) << "Partial result: " << decoder.result()[0].sentence;
    }

    if (state == wenet::DecodeState::kEndpoint) {
      if (decoder.DecodedSomething()) {
        decoder.Rescoring();
        final_result.append(decoder.result()[0].sentence + ", ");
        LOG(INFO) << "Final result: " << final_result;
      }
      decoder.ResetContinuousDecoding();
    }
  }
  Pa_CloseStream(stream);
  Pa_Terminate();
  return 0;
}

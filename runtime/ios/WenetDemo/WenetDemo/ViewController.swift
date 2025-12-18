// Copyright (c) 2022 Dan Ma (1067837450@qq.com)
//
//  ViewController.swift
//  WenetDemo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import UIKit
import AVFoundation

class ViewController: UIViewController {

  @IBOutlet weak var label: UILabel!
  @IBOutlet weak var button: UIButton!

  var wenetModel: Wenet?
  var audioEngine: AVAudioEngine?
  var startRecord = false
  private var workItem: DispatchWorkItem?

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view.

    initModel()

    initRecorder()
  }

  func initModel() {
    guard let modelPath = Bundle.main.path(forResource: "final", ofType: "zip"),
          let dictPath = Bundle.main.path(forResource: "units", ofType: "txt") else {
      print("Error: Model or dictionary file not found.")
      return
    }

    wenetModel = Wenet(modelPath: modelPath, dictPath: dictPath)
    wenetModel?.reset()
    print("Model initialized successfully.")
  }

  func initRecorder() {
    audioEngine = AVAudioEngine()
    guard let inputNode = audioEngine?.inputNode else {
      print("Error: Unable to access audio input node.")
      return
    }

    let bus = 0
    let inputFormat = inputNode.outputFormat(forBus: bus)
    guard let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                           sampleRate: 16000,
                                           channels: 1,
                                           interleaved: false) else {
      print("Error: Unable to create output audio format.")
      return
    }

    guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
      print("Error: Unable to create audio converter.")
      return
    }

    inputNode.installTap(onBus: bus, bufferSize: 1024, format: inputFormat) { [weak self] buffer, _ in
      guard let self = self else { return }
      var newBufferAvailable = true
      let inputCallback: AVAudioConverterInputBlock = { _, outStatus in
        if newBufferAvailable {
          outStatus.pointee = .haveData
          newBufferAvailable = false

          return buffer
        } else {
          outStatus.pointee = .noDataNow
          return nil
        }
      }

      guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat,
                                                   frameCapacity: AVAudioFrameCount(outputFormat.sampleRate) * buffer.frameLength / AVAudioFrameCount(buffer.format.sampleRate)) else {
        print("Error: Unable to create converted buffer.")
        return
      }

      var error: NSError?
      let status = converter.convert(
        to: convertedBuffer,
        error: &error, withInputFrom: inputCallback)

      // 16000 Hz buffer
      let actualSampleCount = Int(convertedBuffer.frameLength)

      guard let floatChannelData = convertedBuffer.floatChannelData else {
        print("Error: No float channel data available.")
        return
      }

      self.wenetModel?.acceptWaveForm(floatChannelData[0],
                                      Int32(actualSampleCount))
      print("Audio data accepted by the model.")
    }
    print("Audio recorder initialized successfully.")
  }

  @IBAction func btnClicked(_ sender: Any) {
    if(!startRecord) {
      //Clear result
      self.setResult(text: "")

      //Reset model
      self.wenetModel?.reset()
      print("Model reset and result cleared.")

      //Start record
      do {
        try audioEngine?.start()
        print("Audio engine started.")
      } catch let error as NSError {
        print("Got an error starting audioEngine: \(error.domain), \(error)")
        return
      }

      //Start decode thread
      workItem = DispatchWorkItem { [weak self] in
        guard let self = self else { return }
        while !(self.workItem?.isCancelled ?? true) {
          self.wenetModel?.decode()
          DispatchQueue.main.sync {
            self.setResult(text: self.wenetModel?.get_result() ?? "")
            print("Decoding in progress.")
          }
        }
      }
      DispatchQueue.global().async(execute: workItem!)

      startRecord = true
      button.setTitle("Stop Record", for: .normal)
      print("Recording started.")
    } else {
      //Stop record
      self.audioEngine?.stop()

      //Stop decode thread
      workItem?.cancel()
      startRecord = false
      button.setTitle("Start Record", for: .normal)
      print("Recording stopped.")
    }
  }

  @objc func setResult(text: String) {
    label.text = text
    print("Result updated: \(text)")
  }
}

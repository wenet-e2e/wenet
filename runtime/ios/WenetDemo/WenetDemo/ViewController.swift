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
  var startRecord: Bool?
  private var workItem: DispatchWorkItem?

  override func viewDidLoad() {
    super.viewDidLoad()
    // Do any additional setup after loading the view.

    initModel()

    initRecorder()
  }

  func initModel() {
    let modelPath = Bundle.main.path(forResource: "final", ofType: "zip")
    let dictPath = Bundle.main.path(forResource: "units", ofType: "txt")
    wenetModel = Wenet(modelPath:modelPath, dictPath:dictPath)!

    wenetModel?.reset()
  }

  func initRecorder() {
    startRecord = false

    audioEngine = AVAudioEngine()
    let inputNode = self.audioEngine?.inputNode
    let bus = 0
    let inputFormat = inputNode?.outputFormat(forBus: bus)
    let outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                     sampleRate: 16000, channels: 1,
                                     interleaved: false)!
    let converter = AVAudioConverter(from: inputFormat!, to: outputFormat)!
    inputNode!.installTap(onBus: bus,
                          bufferSize: 1024,
                          format: inputFormat) {
      (buffer: AVAudioPCMBuffer, when: AVAudioTime) in
      var newBufferAvailable = true

      let inputCallback: AVAudioConverterInputBlock = {
        inNumPackets, outStatus in
        if newBufferAvailable {
          outStatus.pointee = .haveData
          newBufferAvailable = false

          return buffer
        } else {
          outStatus.pointee = .noDataNow
          return nil
        }
      }

      let convertedBuffer = AVAudioPCMBuffer(
        pcmFormat: outputFormat,
        frameCapacity:
          AVAudioFrameCount(outputFormat.sampleRate)
        * buffer.frameLength
        / AVAudioFrameCount(buffer.format.sampleRate))!

      var error: NSError?
      let status = converter.convert(
        to: convertedBuffer,
        error: &error, withInputFrom: inputCallback)

      // 16000 Hz buffer
      let actualSampleCount = Int(convertedBuffer.frameLength)
      guard let floatChannelData = convertedBuffer.floatChannelData
      else { return }

      self.wenetModel?.acceptWaveForm(floatChannelData[0],
                                      Int32(actualSampleCount))
    }
  }

  @IBAction func btnClicked(_ sender: Any) {
    if(!startRecord!) {
      //Clear result
      self.setResult(text: "")

      //Reset model
      self.wenetModel?.reset()

      //Start record
      do {
        try self.audioEngine?.start()
      } catch let error as NSError {
        print("Got an error starting audioEngine: \(error.domain), \(error)")
      }

      //Start decode thread
      workItem = DispatchWorkItem {
        while(!self.workItem!.isCancelled) {
          self.wenetModel?.decode()
          DispatchQueue.main.sync {
            self.setResult(text: (self.wenetModel?.get_result())!)
          }
        }
      }
      DispatchQueue.global().async(execute: workItem!)

      startRecord = true
      button.setTitle("Stop Record", for: UIControl.State.normal)
    } else {
      //Stop record
      self.audioEngine?.stop()

      //Stop decode thread
      workItem!.cancel()

      startRecord = false
      button.setTitle("Start Record", for: UIControl.State.normal)
    }
  }

  @objc func setResult(text: String) {
    label.text = text
  }
}

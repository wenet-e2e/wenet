//
//  ViewController.swift
//  WenetDemo
//
//  Created by 马丹 on 2022/10/31.
//

import UIKit
import AVFoundation

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
        
        let modelPath = Bundle.main.path(forResource: "final", ofType: "zip")
        let dictPath = Bundle.main.path(forResource: "units", ofType: "txt")
        let wenetModel = Wenet(modelPath:modelPath, dictPath:dictPath)
        
        wenetModel?.reset()
        
        do {
            guard let url = Bundle.main.url(forResource: "test", withExtension: "wav") else { return }
            let file = try AVAudioFile(forReading: url)
            if let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: file.fileFormat.sampleRate, channels: file.fileFormat.channelCount, interleaved: false), let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) {

                try file.read(into: buf)
                guard let shortChannelData = buf.floatChannelData else { return }
                let frameLength = Int(buf.frameLength)
                
                let samples = Array(UnsafeBufferPointer(start:shortChannelData[0], count:frameLength))
                print("samples")
                print(samples.count)
                print(samples.prefix(10))
                
                wenetModel?.acceptWaveForm(shortChannelData[0], Int32(frameLength))
                wenetModel?.decode()                
            }
        } catch {
            print("Audio Error: \(error)")
        }
    }
}

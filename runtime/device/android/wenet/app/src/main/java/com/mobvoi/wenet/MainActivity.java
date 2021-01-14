package com.mobvoi.wenet;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.BlockingQueue;

public class MainActivity extends AppCompatActivity {

  Button button = null;
  boolean start_record = false;

  final String LOG_TAG = "WENET";
  final int SAMPLE_RATE = 16000; // The sampling rate
  int miniBufferSize = 0; // 1280 bytes 648 byte 40ms, 0.04s
  final int MAX_QUEUE_SIZE = 2500; // 100 seconds audio, 1 / 0.04 * 100
  BlockingQueue<short[]> bufferQueue = new ArrayBlockingQueue<short[]>(MAX_QUEUE_SIZE);

  AudioRecord record = null;
  static Object recordLock = new Object();

  VoiceRectView voiceView = null;
  TextView textView = null;

  static Object queueLock = new Object();


  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    // Example of a call to a native method
    textView = (TextView)findViewById(R.id.textView);
    textView.setText(Recognize.test());

    button = findViewById(R.id.button);
    button.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        if (!start_record) {
          start_record = true;
          startRecordThread();
          button.setText("Stop Record");
        } else {
          start_record = false;
          button.setText("Start Record");
        }
      }
    });

    voiceView = (VoiceRectView) (findViewById(R.id.voiceRectView));
    initRecoder();
  }

  void initRecoder() {
    // buffer size in bytes 1280
    miniBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT);
    if (miniBufferSize == AudioRecord.ERROR || miniBufferSize == AudioRecord.ERROR_BAD_VALUE) {
      Log.e(LOG_TAG, "Audio buffer can't initialize!");
      return;
    }
    record = new AudioRecord(MediaRecorder.AudioSource.DEFAULT,
        SAMPLE_RATE,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        miniBufferSize);
    if (record.getState() != AudioRecord.STATE_INITIALIZED) {
      Log.e(LOG_TAG, "Audio Record can't initialize!");
      return;
    }
    Log.i(LOG_TAG, "Record init okay");
  }


  double calculateDb(short[] buffer) {
    double energy = 0.0;
    for (int i = 0; i < buffer.length; i++) {
      energy += buffer[i] * buffer[i];
    }
    energy /= buffer.length;
    energy = (10 * Math.log10(1 + energy)) / 100;
    energy = Math.min(energy, 1.0);
    Log.e(LOG_TAG, Double.toString(energy));
    return energy;
  }

  void startRecordThread() {
    new Thread(new Runnable() {
      @Override
      public void run() {
        record.startRecording();
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
        while (true) {
          if (!start_record) break;
          short[] buffer = new short[miniBufferSize / 2];
          int points = record.read(buffer, 0, buffer.length);
          voiceView.add(calculateDb(buffer));
          try {
            bufferQueue.put(buffer);
          } catch (InterruptedException e) {
            e.printStackTrace();
          }
        }
        record.stop();
        voiceView.zero();
      }
    }).start();
  }

  void startAsrThread() {
    new Thread(new Runnable() {
      @Override
      public void run() {
        // Send all data
        while (start_record || bufferQueue.size() > 0) {
          try {
            short [] data = bufferQueue.take();
            // 1. add data to C++ interface

            // 2. get partial result
          } catch (InterruptedException e) {
            e.printStackTrace();
          }
        }

        // Wait for final result
        while (true) {
          // get result
          if (partial) {
          } else {
            break;
          }
        }
      }
    }).start();
  }

}
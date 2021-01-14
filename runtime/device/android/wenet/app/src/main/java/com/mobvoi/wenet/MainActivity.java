package com.mobvoi.wenet;

import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {

  private static final String TAG = "Wenet";

  public static String assetFilePath(Context context, String assetName) {
    File file = new File(context.getFilesDir(), assetName);
    if (file.exists() && file.length() > 0) {
      return file.getAbsolutePath();
    }

    try (InputStream is = context.getAssets().open(assetName)) {
      try (OutputStream os = new FileOutputStream(file)) {
        byte[] buffer = new byte[4 * 1024];
        int read;
        while ((read = is.read(buffer)) != -1) {
          os.write(buffer, 0, read);
        }
        os.flush();
      }
      return file.getAbsolutePath();
    } catch (IOException e) {
      Log.e(TAG, "Error process asset " + assetName + " to file path");
    }
    return null;
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    final String modelPath = new File(assetFilePath(this, "final.zip")).getAbsolutePath();
    final String dictPath = new File(assetFilePath(this, "words.txt")).getAbsolutePath();
    final String wavPath = new File(assetFilePath(this, "test.wav")).getAbsolutePath();
    TextView tv = findViewById(R.id.sample_text);
    tv.setText(Recognize.init(modelPath, dictPath));
    Recognize.recognize(wavPath);
  }
}
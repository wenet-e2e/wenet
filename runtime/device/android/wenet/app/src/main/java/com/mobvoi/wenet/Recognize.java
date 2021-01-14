package com.mobvoi.wenet;

public class Recognize {

  static {
    System.loadLibrary("wenet");
  }

  public static native String init(String modelPath, String dictPath);
  public static native void recognize(String wavPath);
}

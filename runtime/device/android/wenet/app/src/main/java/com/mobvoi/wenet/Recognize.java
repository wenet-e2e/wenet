package com.mobvoi.wenet;

public class Recognize {
    static {
        System.loadLibrary("wenet");
    }

    public static native String test();
}

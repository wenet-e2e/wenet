# Fbank on GPU

```txt
This is a version of DataLoader that FBank calculates on the GPU.

In my project, the training speed was 28% higher than that calculated on the CPU.

NumWork for DataLoader is also supported.

Due to time constraints, I did not move the code further into the WENET. But I think this is something that you can take advantage of.

The overall code style is based on the reading idea of your WENET. So I think you should be able to read the code pretty easily.
```

The main ideas are as follows:

1. Traverse through the file to get all the WAV path, label, etc. (same as WENET)

2. Set the maximum time. This operation can be obtained when counting CMVN.

3. Set the window length and step size of STFT to calculate the number of subsequent nframes (to facilitate the generation of mask)

4. Specaug returns only one mask and passes the mask as a parameter to the model, and sets the required size of the mask (which needs to be set to the number of nframes with the expected maximum number of samples calculated).

5. The rest of the operation is very similar to that of WENET.
6. The model needs to be contained in a Module class, which contains the Fbank extraction process. See the System class in the DataSet for details

Hope to help you, thank you for your busy schedule to check my message


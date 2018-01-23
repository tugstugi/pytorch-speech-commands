Convolutional neural networks for Google speech commands data set with [PyTorch](http://pytorch.org/).

# General
Our team `but`, [Yuan Xu](https://github.com/xuyuan) and [Erdene-Ochir Tuguldur](https://github.com/tugstugi),
participated in the Kaggle competition [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
and reached the 10-th place. This repository contains a simplified and cleaned up version of our team's code.

# Features
* `1x32x32` mel-spectrogram as network input
* single network implementation both for CIFAR10 and Google speech commands data sets
* faster audio data augmentation on STFT
* Kaggle private LB scores evaluated on 150.000+ audio files

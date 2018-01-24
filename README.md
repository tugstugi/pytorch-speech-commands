Convolutional neural networks for [Google speech commands data set](https://research.googleblog.com/2017/08/launching-speech-commands-dataset.html)
with [PyTorch](http://pytorch.org/).

# General
Our team `but`, [Yuan Xu](https://github.com/xuyuan) and [Erdene-Ochir Tuguldur](https://github.com/tugstugi),
participated in the Kaggle competition [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
and reached the 10-th place. This repository contains a simplified and cleaned up version of our team's code.

# Features
* `1x32x32` mel-spectrogram as network input
* single network implementation both for CIFAR10 and Google speech commands data sets
* faster audio data augmentation on STFT
* Kaggle private LB scores evaluated on 150.000+ audio files

# Results

<table><tbody>
<th valign="bottom"><sup><sub>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</sub></sup></th>
<th valign="bottom"><sup><sub>CIFAR-10<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>test set<br/>accuracy</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands/<br/>test set<br/>accuracy<br/>with crop</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands<br/>Kaggle<br/>private LB score</sub></sup></th>
<th valign="bottom"><sup><sub>Speech Commands/<br/>Kaggle<br/>private LB score<br/>with crop</sub></sup></th>
<tr>
<td align="left"><sup><sub>VGG19 BN</sub></sup></td>
<td align="left"><sup><sub>93.56%</sub></sup></td>
<td align="left"><sup><sub>97.337235%</sub></sup></td>
<td align="left"><sup><sub>97.527432%</sub></sup></td>
<td align="left"><sup><sub>0.87454</sub></sup></td>
<td align="left"><sup><sub>0.88030</sub></sup></td>
</tr>
</tbody></table>

# Results with Mixup

Some of the networks were retrained using [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) by Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin and David Lopez-Paz.

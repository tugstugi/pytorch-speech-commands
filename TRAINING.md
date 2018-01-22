## Google Speech Commands

### Training for Kaggle

Due to time limit of the competition, we have trained the smaller nets with `adam`
and the bigger nets with `sgd` both using `ReduceLROnPlateau`.
Earlier stopping the train process will sometimes produce a better score in Kaggle.
All reported Kaggle scores are the private leader board scores.

#### VGG19 BN
* 
```sh
python train_speech_commands.py --model=vgg19_bn --optim=adam --learning-rate=0.0001 --lr-scheduler=plateau --lr-scheduler-patience=2 --max-epochs=30
```

### Training only for test set

Here, we have tried to maximize the accuracy of the test set containing only 6835 audios.

#### VGG19 BN
* accuracy: 97.351865%, accuracy with crop: 97.190929% (very bad Kaggle score)
```sh
python train_speech_commands.py --model=vgg19_bn --optim=sgd --learning-rate=0.001 --lr-scheduler=plateau --lr-scheduler-patience=5 --max-epochs=90
```

## CIFAR10

#### VGG19 BN
* accuracy: 93.56%, loss: 0.337889, epoch time: 19s
```sh
python train-cifar10.py --model=vgg19_bn --optim=sgd --learning-rate=0.1 --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=180
```

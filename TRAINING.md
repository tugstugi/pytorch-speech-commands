## Google Speech Commands

### Training for Kaggle

Due to time limit of the competition, we have trained the smaller nets with `adam`
and the bigger nets with `sgd` both using `ReduceLROnPlateau`.
Earlier stopping the train process will sometimes produce a better score in Kaggle.
All reported Kaggle scores are the private leader board scores.

#### VGG19 BN
* best validation loss model: accuracy: 97.337235%, 97.527432% with crop, Kaggle private LB score: 0.87454 and 0.88030 with crop, epoch time: 1m25s
```sh
python train_speech_commands.py --model=vgg19_bn --optim sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### WideResNet 52-10
```sh
python train_speech_commands.py --model=wideresnet52_10 --optim sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

### Training only for test set

Here, we have tried to maximize the accuracy of the test set containing only 6835 audios.

#### VGG19 BN
* best valid accuracy model: accuracy: 97.439649% and 97.542063% with crop
```sh
python train_speech_commands.py --model=vgg19_bn --optim sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

## CIFAR10

#### VGG19 BN
* accuracy: 93.56%, loss: 0.337889, epoch time: 19s
```sh
python train-cifar10.py --model=vgg19_bn --optim=sgd --learning-rate=0.1 --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=180
```

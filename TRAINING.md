## Google Speech Commands

### Training for Kaggle

Due to time limit of the competition, we have trained the smaller nets with `adam`
and the bigger nets with `sgd` both using `ReduceLROnPlateau`.
Earlier stopping the train process will sometimes produce a better score in Kaggle.
All reported Kaggle scores are the private leader board scores.

#### VGG19 BN
* accuracy: 97.337235%, 97.527432% with crop, Kaggle private LB score: 0.87454 and 0.88030 with crop, epoch time: 1m25s
```sh
python train_speech_commands.py --model=vgg19_bn --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### WideResNet 28-10D
* accuracy: 97.702999%, 97.717630% with crop, Kaggle private LB score:0.89580 and 0.89568 with crop, epoch time: 2m10s
```sh
python train_speech_commands.py --model=wideresnet28_10D --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### WideResNet 52-10
* accuracy: 98.039503%, 97.980980% with crop,  Kaggle private LB score: 0.88159 and 0.88323 with crop, epoch time: 3m55s
```sh
python train_speech_commands.py --model=wideresnet52_10 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### DenseNet-BC (L=190, k=40)
* accuracy: 97.117776%, 97.147037% with crop,  Kaggle private LB score: 0.89369 and 0.89521 with crop, epoch time: 20m
```sh
python train_speech_commands.py --model=wideresnet52_10 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=64
```

## CIFAR10

#### VGG19 BN
* accuracy: 93.56%, loss: 0.337889, epoch time: 19s
```sh
python train-cifar10.py --model=vgg19_bn --optim=sgd --learning-rate=0.1 --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=180
```

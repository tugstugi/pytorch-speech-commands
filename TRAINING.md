## Dependencies
* Anaconda 5.0.1 for Python 3.6
* PyTorch 0.3
* librosa 0.5.1
* [tnt](https://github.com/pytorch/tnt)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

## Google Speech Commands

Before training, execute `./download_speech_commands_dataset.sh` to download the speech commands data set.

#### VGG19 BN
* accuracy: 97.337235%, 97.527432% with crop, Kaggle private LB score: 0.87454 and 0.88030 with crop, epoch time: 1m25s
```sh
python train_speech_commands.py --model=vgg19_bn --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### VGG19 BN with Mixup
* accuracy: 97.483541%, 97.542063% with crop, Kaggle private LB score: 0.89521 and 0.89839 with crop, epoch time: 1m30s
```sh
python train_speech_commands.py --model=vgg19_bn --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96 --mixup
```

#### WideResNet 28-10
* accuracy: 97.937089%, 97.922458% with crop, Kaggle private LB score: 0.88546 and 0.88699 with crop, epoch time: 2m5s
```sh
python train_speech_commands.py --model=wideresnet28_10 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
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

### ResNext29 8x64
* accuracy: 97.190929%, 97.161668% with crop, Kaggle private LB score: 0.89533 and 0.89733 with crop, epoch time: 4m36
```sh
python train_speech_commands.py --model=resnext29_8_64 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

### DPN92
* accuracy: 97.190929%, 97.249451% with crop,  Kaggle private LB score: 0.89075 and 0.89286 with crop, epoch time: 3m45s
```sh
python train_speech_commands.py --model=dpn92 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=96
```

#### DenseNet-BC (L=100, k=12)
* accuracy: 97.161668%, 97.147037% with crop,  Kaggle private LB score: 0.88946 and 0.89134 with crop, epoch time: 1m30s
```sh
python train_speech_commands.py --model=densenet_bc_100_12 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=64
```

#### DenseNet-BC (L=190, k=40)
* accuracy: 97.117776%, 97.147037% with crop,  Kaggle private LB score: 0.89369 and 0.89521 with crop, epoch time: 20m (P6000)
```sh
python train_speech_commands.py --model=densenet_bc_190_40 --optim=sgd --lr-scheduler=plateau --learning-rate=0.01 --lr-scheduler-patience=5 --max-epochs=70 --batch-size=64
```

## CIFAR10

#### VGG19 BN
* accuracy: 93.56%, epoch time: 19s
```sh
python train_cifar10.py --model=vgg19_bn --optim=sgd --learning-rate=0.1 --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=180
```

#### WideResNet 28-10D
* accuracy: 96.22%, epoch time: ?
```sh
python train_cifar10.py --model=wideresnet28_10D --optim=sgd --learning-rate=0.1 --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=240 --lr-scheduler-gamma=0.2 --weight-decay=5e-4
```

#### DenseNet-BC (L=100, k=12)
* accuracy: 95.52%, epoch time: 1m17s
```sh
python train_cifar10.py --model=densenet_bc_100_12 --optim sgd --lr-scheduler=step --learning-rate=0.1 --lr-scheduler-gamma=0.1 --lr-scheduler-step=130 --max-epochs=390 --weight-decay=1e-4 --train-batch-size=64
```

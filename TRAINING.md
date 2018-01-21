## Google Speech Commands

## CIFAR10

### VGG19 BN
* accuracy: 93.56%, loss: 0.337889, epoch time: 19s
```sh
python train-cifar10.py --model=vgg19_bn --optim=sgd --lr-scheduler=step --lr-scheduler-step-size=60 --max-epochs=180
```

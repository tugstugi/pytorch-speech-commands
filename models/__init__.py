from .vgg import *
from .wideresnet import *
from .dpn import *
from .resnet import *
from .densenet import *
from .resnext import *

available_models = [
    'vgg19_bn',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'wideresnet28_10', 'wideresnet28_10D', 'wideresnet52_10',
    'resnext29_8_64',
    'dpn92',
    'densenet_bc_100_12', 'densenet_bc_250_24', 'densenet_bc_190_40'
]

def create_model(model_name, num_classes, in_channels):
    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes, in_channels=in_channels)
    if model_name == "resnet34":
        model = resnet34(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet101":
        model = resnet101(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnet152":
        model = resnet152(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet28_10":
        model = WideResNet(depth=28, widen_factor=10, dropRate=0, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet28_10D":
        model = WideResNet(depth=28, widen_factor=10, dropRate=0.3, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "wideresnet52_10":
        model = WideResNet(depth=52, widen_factor=10, dropRate=0, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "resnext29_8_64":
        model = CifarResNeXt(nlabels=num_classes, in_channels=in_channels)
    elif model_name == "dpn92":
        model = DPN92(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_100_12":
        model = DenseNet(depth=100, growthRate=12, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_250_24":
        model = DenseNet(depth=250, growthRate=24, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    elif model_name == "densenet_bc_190_40":
        model = DenseNet(depth=190, growthRate=40, compressionRate=2, num_classes=num_classes, in_channels=in_channels)
    else:
        model = vgg19_bn(num_classes=num_classes, in_channels=in_channels)
    return model

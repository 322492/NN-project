import torch
import torch.nn as nn
from torchvision import models


def ResNetBinaryClassifier(pretrained=True):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet18(weights=weights)

    # zamrażamy wszystkie warstwy
    for param in model.parameters():
        param.requires_grad = False

    # podmieniamy ostatnią warstwę na binary classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    # nowa warstwa fc domyślnie ma requires_grad=True
    return model
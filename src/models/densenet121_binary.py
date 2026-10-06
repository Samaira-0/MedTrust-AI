"""ImageNet DenseNet-121 with a dropout binary classifier returning logits."""

from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121_binary(dropout_probability=0.2):
    if not 0 <= dropout_probability < 1:
        raise ValueError("dropout_probability must be in [0, 1)")
    model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_probability),
        nn.Linear(model.classifier.in_features, 1),
    )
    # DenseNet.forward returns raw classifier outputs, without sigmoid.
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model

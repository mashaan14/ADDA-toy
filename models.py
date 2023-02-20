"""
This file contains three models needed to build ADDA:
    - LeNetEncoder: to extract features from images.
    - LeNetClassifier: to perform image classification.
    - Discriminator model: to perform adversarial adaptation that if sees encoded source and target examples cannot reliably predict their domain label.

Tzeng, E., Hoffman, J., Saenko, K., & Darrell, T. (2017). Adversarial discriminative domain adaptation.
In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7167-7176).
"""

import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """
    encoder for ADDA.
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 2)
        )

    def forward(self, input):
        out = self.layer(input)
        return out


class Classifier(nn.Module):
    """
    classifier for ADDA.
    """
    def __init__(self):
        super(Classifier, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        out = self.layer(input)
        return out


class Discriminator(nn.Module):
    """
    Discriminator model for source domain.
    """

    def __init__(self):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(2, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
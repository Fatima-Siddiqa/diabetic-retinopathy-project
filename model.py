import torch
import torch.nn as nn
from torchvision import models


class CustomHead(nn.Module):
    """Kept for reference / future training runs."""
    def __init__(self, in_features=1280, hidden=512, num_classes=5,
                 dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CORALHead(nn.Module):
    def __init__(self, in_features=1280, hidden=512,
                 num_classes=5, dropout1=0.4, dropout2=0.3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(dropout2),
        )
        self.fc = nn.Linear(hidden, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        x = self.feature(x)
        return torch.sigmoid(self.fc(x) + self.bias)


def build_model(num_classes=5, coral=False, pretrained=True,
                dropout1=0.4, dropout2=0.3):
    """
    Build EfficientNet-B0 with a custom head.

    The saved checkpoint uses a flat nn.Sequential classifier
    (keys: classifier.0, classifier.1 ... classifier.4), so we
    assign nn.Sequential directly — NOT the CustomHead wrapper —
    to keep key names compatible.

    During inference pass pretrained=False so no ImageNet download happens.
    """
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    )
    in_features = model.classifier[1].in_features

    if coral:
        model.classifier = CORALHead(in_features, 512, num_classes,
                                     dropout1, dropout2)
    else:
        # Flat Sequential — matches checkpoint keys classifier.1.*, classifier.4.*
        model.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(512, num_classes),
        )
    return model


def load_checkpoint(model, path, device="cpu"):
    """Load a saved state-dict into *model* and switch to eval mode."""
    state = torch.load(path, map_location=device, weights_only=False)

    # Support checkpoints saved as {"model_state_dict": ...} or raw state-dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
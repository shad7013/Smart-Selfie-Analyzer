import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiTaskModel(nn.Module):
    """
    Multi-task model for:
    - Age group classification
    - Gender classification
    - Emotion classification

    Backbone: ResNet18 (pretrained)
    """

    def __init__(self, num_age_classes=5, num_gender_classes=2, num_emotion_classes=7):
        super().__init__()

        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes
        self.num_emotion_classes = num_emotion_classes

        # Load pretrained backbone
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Remove FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Heads
        self.age_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_age_classes)
        )

        self.gender_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_gender_classes)
        )

        self.emotion_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotion_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        return {
            "age": self.age_head(features),
            "gender": self.gender_head(features),
            "emotion": self.emotion_head(features)
        }
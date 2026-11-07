import torch
from torch import nn

class RadarCNN(nn.Module):
    def __init__(self, classes: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 96, 3, padding=1), nn.BatchNorm1d(96), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(96, 64), nn.ReLU(), nn.Dropout(0.25), nn.Linear(64, classes))
    def forward(self, x):
        return self.classifier(self.features(x))

class DenseBaseline(nn.Module):
    def __init__(self, length: int = 256, classes: int = 8):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(2 * length, 256), nn.ReLU(),
                                 nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, classes))
    def forward(self, x): return self.net(x)

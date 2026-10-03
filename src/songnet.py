import torch
import torch.nn as nn


class SongNet(nn.Module):
    def __init__(self, n_mels=128, num_classes=8):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)

        x = x.permute(0, 2, 1)

        _, hidden = self.gru(x)

        x = hidden[-1]

        x = self.fc(x)

        return x


class SongNetPaper(nn.Module):
    """Three temporal CNN blocks, timestep logits, then song-level averaging."""

    def __init__(self, n_mels=128, num_classes=8, dropout=0.3):
        super().__init__()
        blocks = []
        for in_channels, out_channels, kernel in (
            (n_mels, 128, 5), (128, 256, 5), (256, 256, 3)
        ):
            blocks.extend([
                nn.Conv1d(in_channels, out_channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_channels), nn.ReLU(), nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
        self.cnn = nn.Sequential(*blocks)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        features = self.cnn(x).transpose(1, 2)
        return self.classifier(features).mean(dim=1)

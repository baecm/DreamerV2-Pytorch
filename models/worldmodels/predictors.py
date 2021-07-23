import torch.nn as nn


class ImagePredictor(nn.Module):
    def __init__(self):
        super(ImagePredictor, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ELU(inplace=True),
            nn.ConvTranspose2d(8, 8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.ELU(inplace=True),
            nn.Conv2d(8, 3, kernel_size=(3, 3), padding=(1, 1)),
        )

    def forward(self):
        pass


class TransitionPredictor(nn.Module):
    def __init__(self):
        super(TransitionPredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward(self):
        pass


class RewardPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024 + 512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self):
        pass


class DiscountPredictor(nn.Module):
    def __init__(self):
        super(RewardPredictor).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1024 + 512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self):
        pass

import torch.nn as nn


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32 * 32)
        return self.model(z_sample)


class CriticLoss(nn.Module):
    def __init__(self):
        super(CriticLoss, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, V, ve):
        return self.mse(V, ve)

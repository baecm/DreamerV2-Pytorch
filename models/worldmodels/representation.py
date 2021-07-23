import torch
import torch.nn as nn

from torch.distributions import *


class RepresentationModel(nn.Module):
    def __init__(self):
        super(RepresentationModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(4, 4)),  # 512, 1, 1
            nn.ELU(inplace=True)
        )
        # q (2nd part): (h, xembedded) -> z
        self.mlp = nn.Sequential(
            nn.Linear(512 + 512, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ELU(inplace=True),
            nn.Linear(1024, 1024),
        )

    def forward(self, x, h, a=None, z=None, inference=False, batch_size=None):
        embedded = self.conv(x)
        embedded = embedded.reshape(-1, 512)
        embedded = torch.cat((h, embedded), dim=1)
        z_logits = self.representation_model_mlp(embedded)
        z_sample = one_hot_categorical.OneHotCategorical(logits=z_logits.reshape(-1, 32, 32)).sample()

        if inference:
            if h is None:  # starting new sequence
                h = torch.zeros((batch_size, 512))
            else:
                h = self.gru(torch.cat((z.reshape(-1, 32 * 32), a), dim=1), h)
            return z_sample, h

        else:
            z_probs = torch.softmax(z_logits.reshape(-1, 32, 32), dim=-1)
            z_sample = z_sample + z_probs - z_probs.detach()

            return z_logits, z_sample

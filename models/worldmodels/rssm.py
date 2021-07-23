import torch
import torch.nn as nn

class RSSM(nn.Module):
    def __init__(self, num_action, hidden_size):
        super(RSSM, self).__init__()

        self.gru = nn.GRUCell(
            input_size=1024 + num_action,
            hidden_size=512,
        )

    def forward(self, a, h, z):
        return self.rssm(torch.cat((z.reshape(-1, 32 * 32), a), dim=1), h)


    def initial(self, batch_size, device):
        return torch.zeros(batch_size, 512).to(device)
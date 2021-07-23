import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, num_actions=9):
        super(Actor, self).__init__()

        self.num_actions = num_actions

        self.model = nn.Sequential(
            nn.Linear(32 * 32, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 256),
            nn.ELU(inplace=True),
            nn.Linear(256, num_actions),
            nn.Tanh()
        )

    def forward(self, z_sample):
        z_sample = z_sample.reshape(-1, 32 * 32)
        return self.model(z_sample) * 2


class ActorLoss(nn.Module):
    def __init__(self, ns=0.9, nd=0.1, ne=3e-3):
        super(ActorLoss, self).__init__()

        self.ns = ns
        self.nd = nd
        self.ne = ne

        self.anneal = 1e-5

    def forward(self, a, dist_a, V, ve):
        loss = -self.ns * dist_a.log_prob(a) * (V - ve).detach().squeeze(-1) \
               - self.nd * V.squeeze(-1) \
               - self.ne * dist_a.entropy()

        self.nd = max(0, self.nd - self.anneal)
        self.ne = max(3e-4, self.ne - self.anneal)

        return loss.mean()

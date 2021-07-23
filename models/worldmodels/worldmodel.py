import torch
import torch.nn as nn

from torch.distributions import *

from .representation import *
from .predictors import *
from .rssm import *

class WorldModel(nn.Module):
    def __init__(self, gamma, num_action=9):
        super(WorldModel, self).__init__()
        self.gamma = gamma  # discount factor
        self.num_action = num_action

        # Recurrent Model (RSSM): ((z, a), h) -> h
        self.rssm = RSSM(num_action=num_action)
        self.representation_model = RepresentationModel()
        self.transition_predictor = TransitionPredictor()
        self.r_predictor_mlp = RewardPredictor()
        self.gamma_predictor_mlp = DiscountPredictor()
        # p: (h,z) -> x_hat
        self.x_hat_predictor_mlp = nn.Sequential(
            nn.Linear(1024 + 512, 1024),
            nn.ELU(inplace=True),
        )
        self.image_predictor_conv = ImagePredictor()

    def compute_z_hat_sample(self, z_hat_logits):
        """
        In:
            z_hat_logits: logits of z_hat_t
        Out:
            z_hat_sample (with straight through gradient)
        """
        z_hat_sample = torch.distributions.one_hot_categorical.OneHotCategorical(
            logits=z_hat_logits.reshape(-1, 32, 32)
        ).sample()
        z_hat_probs = torch.softmax(z_hat_logits.reshape(-1, 32, 32), dim=-1)

        return z_hat_sample + z_hat_probs - z_hat_probs.detach()

    def compute_x_hat(self, h_z):
        """
        In:
            h_z: concat of h_t and z_t
        Out:
            x_hat: x_hat_t
        """
        x_hat = self.x_hat_predictor_mlp(h_z)
        x_hat = x_hat.reshape(-1, 64, 4, 4)
        return self.image_predictor_conv(x_hat)

    def dream(self, a, x, z, h):
        h = self.compute_h(a.shape[0], a.device, a, h, z)

        z_logits, z_sample = None, None  # No z

        z_hat_logits = self.transition_predictor(h)
        z_hat_sample = self.compute_z_hat_sample(z_hat_logits)

        x_hat = None

        h_z = torch.cat((h, z_hat_sample.reshape(-1, 32 * 32)), dim=1)
        r_hat = self.r_predictor_mlp(h_z)
        gamma_hat = self.gamma_predictor_mlp(h_z)

        # r_hat_sample = torch.distributions.normal.Normal(
        #     loc=r_hat,
        #     scale=1.0
        # ).sample()
        r_hat_sample = r_hat.detach()

        gamma_hat_sample = torch.distributions.bernoulli.Bernoulli(
            logits=gamma_hat
        ).sample() * self.gamma  # Bernoulli in {0,1}

        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (
            z_hat_sample, r_hat_sample, gamma_hat_sample)

    def train(self, a, x, z, h):
        h = self.compute_h(x.shape[0], x.device, a, h, z)

        z_logits, z_sample = self.representation_model(x, h)

        z_hat_logits = self.transition_predictor(h)
        z_hat_sample = None

        h_z = torch.cat((h, z_sample.reshape(-1, 32 * 32)), dim=1)

        r_hat = self.r_predictor_mlp(h_z)

        gamma_hat = self.gamma_predictor_mlp(h_z)

        x_hat = self.compute_x_hat(h_z)

        r_hat_sample, gamma_hat_sample = None, None

        return z_logits, z_sample, z_hat_logits, x_hat, r_hat, gamma_hat, h, (
            z_hat_sample, r_hat_sample, gamma_hat_sample)

    def forward(self, a, x, z, h=None, dream=False, inference=False):
        """
        Input:
            a: a_t-1
            x: x_t
            z: z_t-1
            h: h_t-1
        """
        if inference:  # only use embedding network, i.e. no image predictor
            return self.representation_model(x, h, a=a, z=z, inference=True)
        elif dream:
            return self.dream(a, x, z, h)
        else:
            return self.train(a, x, z, h)


class WorldModelLoss(nn.Module):
    def __init__(self, nx=1 / 64 / 64 / 3, nr=1, ng=1, nt=0.08, nq=0.1):
        super(WorldModelLoss, self).__init__()

        self.nx = nx
        self.nr = nr
        self.ng = ng
        self.nt = nt
        self.nq = nq

    def forward(self, x, r, gamma, z_logits, z_sample, x_hat, r_hat, gamma_hat, z_hat_logits):
        x_dist = normal.Normal(loc=x_hat, scale=1.0)
        r_dist = normal.Normal(loc=r_hat, scale=1.0)
        gamma_dist = bernoulli.Bernoulli(logits=gamma_hat)
        z_hat_dist = one_hot_categorical.OneHotCategorical(logits=z_hat_logits.reshape(-1, 32, 32))
        z_dist = one_hot_categorical.OneHotCategorical(logits=z_logits.reshape(-1, 32, 32).detach())
        z_sample = z_sample.reshape(-1, 32, 32)

        loss = - self.nx * x_dist.log_prob(x).mean() \
               - self.nr * r_dist.log_prob(r).mean() \
               - self.ng * gamma_dist.log_prob(gamma.round()).mean() \
               - self.nt * z_hat_dist.log_prob(z_sample.detach()).mean() \
               + self.nq * z_dist.log_prob(z_sample).mean()

        return loss

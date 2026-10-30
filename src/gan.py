"""Conditional latent GAN (WGAN-GP) on top of the text VAE.

Text is discrete, so a GAN cannot be trained on tokens directly (no gradient through argmax).
We train the GAN in the continuous latent space of the VAE instead (Adversarially Regularised
Auto-encoder idea): the generator maps (noise, intent) -> latent code z, the critic tells real
encoder latents from generated ones, and the frozen VAE decoder turns z back into a message.
"""

import logging

import torch
import torch.nn as nn

from src.vae import DEVICE

log = logging.getLogger("gan")


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, o))


class LatentGAN(nn.Module):
    def __init__(self, n_labels, zdim=64, noise_dim=64, ydim=32, hid=256):
        super().__init__()
        self.cfg = dict(n_labels=n_labels, zdim=zdim, noise_dim=noise_dim, ydim=ydim, hid=hid)
        self.noise_dim, self.zdim = noise_dim, zdim
        self.g_y = nn.Embedding(n_labels, ydim)
        self.d_y = nn.Embedding(n_labels, ydim)
        self.G = mlp(noise_dim + ydim, hid, zdim)
        self.D = mlp(zdim + ydim, hid, 1)

    def generate(self, y, noise=None):
        if noise is None:
            noise = torch.randn(len(y), self.noise_dim, device=y.device)
        return self.G(torch.cat([noise, self.g_y(y)], -1))

    def critic(self, z, y):
        return self.D(torch.cat([z, self.d_y(y)], -1)).squeeze(-1)


def gradient_penalty(gan, real, fake, y):
    a = torch.rand(real.size(0), 1, device=real.device)
    mix = (a * real + (1 - a) * fake).requires_grad_(True)
    score = gan.critic(mix, y)
    grad = torch.autograd.grad(score.sum(), mix, create_graph=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def train_gan(mu, logvar, labels, n_labels, steps=3000, batch=128, n_critic=5, lam=10.0, lr=1e-4, seed=0):
    """mu, logvar: VAE posterior of the real training messages; labels: their intent ids."""
    torch.manual_seed(seed)
    gan = LatentGAN(n_labels, zdim=mu.shape[1]).to(DEVICE)
    opt_g = torch.optim.Adam(list(gan.G.parameters()) + list(gan.g_y.parameters()), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(list(gan.D.parameters()) + list(gan.d_y.parameters()), lr=lr, betas=(0.5, 0.9))
    mu, logvar, labels = mu.to(DEVICE), logvar.to(DEVICE), torch.as_tensor(labels).to(DEVICE)
    N = len(mu)
    hist = []
    for step in range(1, steps + 1):
        for _ in range(n_critic):
            idx = torch.randint(0, N, (batch,), device=DEVICE)
            real = mu[idx] + torch.randn_like(mu[idx]) * torch.exp(0.5 * logvar[idx])
            y = labels[idx]
            fake = gan.generate(y).detach()
            d_loss = gan.critic(fake, y).mean() - gan.critic(real, y).mean() + lam * gradient_penalty(gan, real, fake, y)
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        y = labels[torch.randint(0, N, (batch,), device=DEVICE)]
        g_loss = -gan.critic(gan.generate(y), y).mean()
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        if step % 500 == 0 or step == 1:
            w = -d_loss.item()
            hist.append(dict(step=step, d_loss=d_loss.item(), g_loss=g_loss.item()))
            log.info("GAN step %d/%d  critic loss %.3f  generator loss %.3f", step, steps, d_loss.item(), g_loss.item())
    return gan, hist


@torch.no_grad()
def sample_latents(gan, label_ids):
    gan.eval()
    return gan.generate(torch.tensor(label_ids, device=DEVICE))

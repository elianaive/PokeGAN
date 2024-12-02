import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import math
import random

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=1)
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        return self.gamma*out + x

class FastGenerator(nn.Module):
    def __init__(self, ngf=64, nz=100, nc=3):
        super(FastGenerator, self).__init__()
        
        self.fc = spectral_norm(nn.Linear(nz, 4 * 4 * ngf * 8))
        
        self.main = nn.ModuleList([
            # 4x4x512 -> 8x8x256
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf * 4),
                nn.LeakyReLU(0.2, True)
            ),
            # 8x8x256 -> 16x16x128
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf * 2),
                nn.LeakyReLU(0.2, True)
            ),
            # 16x16x128 -> 32x32x64
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, True),
                SelfAttention(ngf)
            ),
            # 32x32x64 -> 64x64x3
            nn.Sequential(
                spectral_norm(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)),
                nn.Tanh()
            )
        ])
        
        # Skip layer correlations with adjusted channels
        self.skips = nn.ModuleList([
            spectral_norm(nn.Conv2d(ngf * 8, ngf * 4, 1)),  # 512 -> 256
            spectral_norm(nn.Conv2d(ngf * 4, ngf * 2, 1)),  # 256 -> 128
            spectral_norm(nn.Conv2d(ngf * 2, ngf, 1)),      # 128 -> 64
        ])

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        
        features = []
        
        for i, layer in enumerate(self.main[:-1]):
            features.append(x)
            x = layer(x)
            
            if i < len(self.skips):
                skip = self.skips[i](features[-1])
                
                # Upsample skip to match x's spatial dimensions
                if skip.shape[-2:] != x.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
                
                # Get dimensions
                b, c_skip, h, w = skip.shape
                b, c_x, h, w = x.shape
                hw = h * w
                
                # Reshape tensors for correlation
                skip_flat = skip.reshape(b, c_skip, hw)
                x_flat = x.reshape(b, c_x, hw)
                
                # Normalize features
                skip_flat = F.normalize(skip_flat, dim=2)
                x_flat = F.normalize(x_flat, dim=2)
                
                # Compute correlation
                correlation = torch.bmm(skip_flat, x_flat.transpose(1, 2))
                correlation = F.softmax(correlation / math.sqrt(hw), dim=2)
                
                # Apply correlation
                x_residual = torch.bmm(correlation, x_flat)
                x_residual = x_residual.reshape(b, c_skip, h, w)
                
                # Match channels if needed
                if c_skip != c_x:
                    channel_matcher = nn.Conv2d(c_skip, c_x, 1).to(x.device)
                    x_residual = channel_matcher(x_residual)
                
                x = x + 0.3*x_residual
        
        x = self.main[-1](x)
        return x

class FastDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
        super(FastDiscriminator, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.adv_layer = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        
        self.downsampled = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        main_features = self.main(x)
        main_out = self.adv_layer(main_features)
        down_out = self.downsampled(x)
        return (main_out.view(-1, 1), down_out.view(-1, 1))

class FastPokemonGAN:
    def __init__(self, latent_dim=100, lr_g=0.0002, lr_d=0.0004, b1=0.0, b2=0.9, n_critic=5, clip_value=0.01, device='cuda'):
        self.latent_dim = latent_dim
        self.device = device
        self.n_critic = n_critic  # number of critic iterations per generator iteration
        self.clip_value = clip_value  # for weight clipping

        self.generator = FastGenerator(nz=latent_dim).to(device)
        self.discriminator = FastDiscriminator().to(device)

        self.optimizer_G = torch.optim.RMSprop(
            self.generator.parameters(), 
            lr=lr_g
        )
        self.optimizer_D = torch.optim.RMSprop(
            self.discriminator.parameters(), 
            lr=lr_d
        )

    def train_step(self, real_imgs):
        batch_size = real_imgs.size(0)
        
        d_loss = 0
        for _ in range(self.n_critic):
            self.optimizer_D.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_imgs = self.generator(z)
            
            real_noisy = real_imgs + 0.05 * torch.randn_like(real_imgs).to(self.device)
            fake_noisy = fake_imgs + 0.05 * torch.randn_like(fake_imgs)
            
            real_main, real_down = self.discriminator(real_noisy)
            fake_main, fake_down = self.discriminator(fake_noisy.detach())
            
            # Wasserstein loss for main path
            main_loss = -torch.mean(real_main) + torch.mean(fake_main)
            
            # Wasserstein loss for downsampled path
            down_loss = -torch.mean(real_down) + torch.mean(fake_down)
            
            # Combined loss
            d_loss = (main_loss + down_loss) / 2
            
            # Gradient penalty
            gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)
            d_loss = d_loss + 10 * gradient_penalty
            
            d_loss.backward()
            self.optimizer_D.step()
            
            # Clip weights of discriminator
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.clip_value, self.clip_value)
        
        self.optimizer_G.zero_grad()
        
        # Generate new fake images
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_imgs = self.generator(z)
        
        # Get critic outputs for generated images
        fake_main, fake_down = self.discriminator(gen_imgs)
        
        # Wasserstein loss for generator (maximize critic output for fake images)
        g_loss = (-torch.mean(fake_main) - torch.mean(fake_down)) / 2
        
        g_loss.backward()
        
        # Diversity loss to prevent mode collapse?
        z2 = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_imgs2 = self.generator(z2)
        diversity_loss = -torch.mean(torch.abs(gen_imgs - gen_imgs2))
        
        g_loss = g_loss + 0.1 * diversity_loss
        
        self.optimizer_G.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gradient_penalty': gradient_penalty.item()
        }

    def compute_gradient_penalty(self, real_imgs, fake_imgs):
        """Gradient penalty for Wasserstein GAN"""
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(self.device)
        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)[0]
        
        fake = torch.ones(real_imgs.size(0), 1).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def generate_sample(self, num_samples=1):
        """Generate sample images"""
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            return self.generator(z)
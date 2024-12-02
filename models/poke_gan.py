import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_channels=3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial size: 1x1x1024
        self.init_size = 4
        self.l1 = nn.Linear(latent_dim, 1024 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            # 4x4x1024 -> 8x8x512
            nn.BatchNorm2d(1024),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8x512 -> 16x16x256
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16x256 -> 32x32x128
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32x128 -> 64x64x64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer
            nn.Conv2d(64, num_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        # Expand latent vector to initial feature map
        out = self.l1(z)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        # Generate image through conv blocks
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            # 64x64x3 -> 32x32x64
            *discriminator_block(num_channels, 64, bn=False),
            # 32x32x64 -> 16x16x128
            *discriminator_block(64, 128),
            # 16x16x128 -> 8x8x256
            *discriminator_block(128, 256),
            # 8x8x256 -> 4x4x512
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = 4
        self.adv_layer = nn.Sequential(
            nn.Linear(512 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)
        return validity

class PokemonGAN:
    def __init__(self, latent_dim=100, lr=0.0002, b1=0.5, b2=0.999, device='cuda'):
        self.latent_dim = latent_dim
        self.device = device

        # Initialize generator and discriminator
        self.generator = Generator(latent_dim).to(device)
        self.discriminator = Discriminator().to(device)

        # Initialize weights
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # Loss function
        self.adversarial_loss = nn.BCELoss()

    @staticmethod
    def _weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train_step(self, real_imgs):
        # Adversarial ground truths
        batch_size = real_imgs.size(0)
        valid = torch.ones((batch_size, 1), requires_grad=False).to(self.device)
        fake = torch.zeros((batch_size, 1), requires_grad=False).to(self.device)

        # -----------------
        #  Train Generator
        # -----------------
        self.optimizer_G.zero_grad()
        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        gen_imgs = self.generator(z)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs), valid)
        g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        self.optimizer_D.zero_grad()
        real_loss = self.adversarial_loss(self.discriminator(real_imgs), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()

        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item()
        }

    def generate_sample(self, num_samples=1):
        """Generate sample images"""
        z = torch.randn(num_samples, self.latent_dim).to(self.device)
        with torch.no_grad():
            return self.generator(z)
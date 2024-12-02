import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.act(x)
        return x

class PixelArtGenerator(nn.Module):
    def __init__(self, latent_dim=512, palette_size=32, image_size=64):
        super().__init__()
        
        self.palette_size = palette_size
        self.image_size = image_size
        
        # Create learnable color palette
        self.palette = nn.Parameter(torch.randn(palette_size, 3))
        self._initialize_palette()
        
        # Initial dense processing
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.GELU(),
            nn.Linear(4 * 4 * 512, 4 * 4 * 512),
            nn.GELU(),
        )
        
        # Main generation layers
        self.main = nn.ModuleList([
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.GroupNorm(8, 512),
            nn.GELU(),
            ResBlock(512),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            ResBlock(256),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock(128),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock(64),
        ])
        
        # Skip connections
        self.skip = nn.ModuleDict({
            '512': nn.Conv2d(512, 64, 1),
            '256': nn.Conv2d(256, 64, 1),
            '128': nn.Conv2d(128, 64, 1)
        })
        
        # Final layers
        self.to_colors = nn.Conv2d(64, palette_size, 1)
        
    def _initialize_palette(self):
        """Initialize with pixel art colors"""
        base_colors = torch.tensor([
            [0.0, 0.0, 0.0],  # Black
            [1.0, 1.0, 1.0],  # White
            [0.8, 0.0, 0.0],  # Red
            [0.0, 0.8, 0.0],  # Green
            [0.0, 0.0, 0.8],  # Blue
            [0.8, 0.8, 0.0],  # Yellow
            [0.8, 0.0, 0.8],  # Magenta
            [0.0, 0.8, 0.8],  # Cyan
        ])
        with torch.no_grad():
            self.palette[:len(base_colors)] = base_colors
    
    def quantize_colors(self, x, training=True):
        """Convert palette weights to RGB colors"""
        temperature = 1.0 if training else 3.0  # Lower temperature for better gradients
        
        if training:
            # Add noise before softmax for better exploration
            noise = torch.randn_like(x) * 0.005
            x = x + noise
        
        x = F.softmax(x * temperature, dim=1)
        x = x.permute(0, 2, 3, 1)
        
        # Use sigmoid instead of hard clamping
        colors = torch.matmul(x, self.palette)
        colors = torch.sigmoid(colors)
        
        return colors.permute(0, 3, 1, 2)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # Initial processing
        x = self.initial(z)
        x = x.view(batch_size, 512, 4, 4)
        
        # Store intermediate activations for skip connections
        features = []
        
        # Main generation with skip connections
        for layer in self.main:
            x = layer(x)
            if isinstance(layer, ResBlock):
                if x.size(1) in [512, 256, 128]:
                    features.append(self.skip[str(x.size(1))](x))
        
        # Apply skip connections
        for feat in features:
            x = x + F.interpolate(feat, size=x.shape[2:])
        
        # Convert to colors
        palette_weights = self.to_colors(x)
        rgb = self.quantize_colors(palette_weights, self.training)
        
        return rgb

class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.LeakyReLU(0.2)
        
        # Skip connection with projection if needed
        self.skip = None
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride)

    def forward(self, x):
        identity = x if self.skip is None else self.skip(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + identity
        x = self.act(x)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        
        self.main = nn.Sequential(
            # 64x64 -> 32x32
            ResBlockDiscriminator(3, 64, stride=2),
            
            # 32x32 -> 16x16
            ResBlockDiscriminator(64, 128, stride=2),
            
            # 16x16 -> 8x8
            ResBlockDiscriminator(128, 256, stride=2),
            
            # 8x8 -> 4x4
            ResBlockDiscriminator(256, 512, stride=2),
            
            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0)
        )
    
    def forward(self, x):
        return self.main(x)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(*[self.norm1(x)] * 3)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class PixelArtViTGenerator(nn.Module):
    def __init__(self, 
                 latent_dim=512, 
                 palette_size=32, 
                 image_size=64,
                 patch_size=4,
                 hidden_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 dropout=0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.palette_size = palette_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Create learnable color palette
        self.palette = nn.Parameter(torch.randn(palette_size, 3))
        
        # Initial latent to patch embeddings
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Output projection for each patch
        patch_dim = patch_size * patch_size * 3
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, patch_dim),
            nn.GELU(),
            nn.Linear(patch_dim, palette_size)
        )
        
        # Initialize color palette with some defaults
        self._initialize_palette()
        
    def _initialize_palette(self):
        """Initialize palette with some common pixel art colors"""
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
        temperature = 5.0 if training else 10.0
        x = F.softmax(x * temperature, dim=1)
        
        if training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
            x = F.softmax(x, dim=1)
            
        x = x.permute(0, 2, 3, 1)
        colors = torch.matmul(x, self.palette)
        colors = torch.clamp(colors, 0, 1)
        return colors.permute(0, 3, 1, 2)
    
    def forward(self, z):
        batch_size = z.size(0)
        
        # Project latent to initial sequence
        x = self.latent_proj(z)
        x = x.unsqueeze(1).expand(-1, self.num_patches, -1)
        
        # Add cls token and position embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Remove cls token
        x = x[:, 1:]
        
        # Project patches to palette logits
        palette_logits = self.output_proj(x)
        
        # Reshape to image format
        h = w = self.image_size // self.patch_size
        palette_logits = palette_logits.reshape(batch_size, h, w, -1)
        palette_logits = palette_logits.permute(0, 3, 1, 2)
        
        # Upsample to full resolution
        palette_logits = F.interpolate(
            palette_logits, 
            size=(self.image_size, self.image_size),
            mode='nearest'
        )
        
        # Convert to RGB using palette
        rgb = self.quantize_colors(palette_logits, self.training)
        
        # Apply grid alignment
        grid_mask = self._create_grid_mask(rgb.shape[-2:])
        rgb = rgb * grid_mask.to(rgb.device)
        
        return rgb
    
    def _create_grid_mask(self, size):
        """Create a mask that emphasizes pixel grid alignment"""
        mask = torch.ones(size[0], size[1])
        for i in range(size[0]):
            for j in range(size[1]):
                if (i % 2 == 0) or (j % 2 == 0):
                    mask[i, j] = 0.95
        return mask.unsqueeze(0).unsqueeze(0)

class Discriminator(nn.Module):
    def __init__(self, image_size=64):
        super().__init__()
        
        self.layers = nn.Sequential(
            # 64x64 -> 32x32
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            # 32x32 -> 16x16
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            # 16x16 -> 8x8
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            # 8x8 -> 4x4
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.GroupNorm(8, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            
            # 4x4 -> 1x1
            nn.Conv2d(512, 1, 4, 1, 0)
        )
        
    def forward(self, x):
        return self.layers(x)
import torch
import torch.utils.data as data
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import time

def get_image_paths(root_dir):
    """Get all valid image file paths from the root directory and its subdirectories."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
    all_paths = [path for path in Path(root_dir).rglob('*') if path.suffix.lower() in valid_extensions]
    valid_paths = [path for path in all_paths if is_valid_image(path)]  # Filter valid images
    print(f"Found {len(valid_paths)} valid images in {root_dir}")
    return valid_paths

def is_valid_image(filepath):
    """Check if the image file is valid and can be opened."""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Ensure image can be opened and decoded
        return True
    except (IOError, SyntaxError):
        print(f"Invalid image detected: {filepath}")
        return False

class PokemonDataset(data.Dataset):
    """Dataset class for loading Pokemon images."""
    
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        print(f"Dataset initialized with {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Replace with a placeholder image
            image = Image.new('RGB', (64, 64), color=(255, 0, 0))  # Red placeholder image
        
        if self.transform:
            image = self.transform(image)
        
        return image, 0  # Return 0 as label since we don't need it for GAN

def setup_pokemon_data(data_dir='data/pokemon', batch_size=32, image_size=64, num_workers=4):
    """
    Load Pokemon sprites from nested directories.
    Handles multiple subdirectories and automatically finds all image files.
    """
    start_time = time.time()
    image_paths = get_image_paths(data_dir)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    # Create dataset and dataloader
    dataset = PokemonDataset(image_paths, transform=transform)
    
    dataloader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=False
    )
    
    print(f"Dataloader setup time: {time.time() - start_time:.2f} seconds")
    return dataloader

def preview_batch(dataloader):
    """Preview a batch of images from the dataloader"""
    start_time = time.time()
    images, _ = next(iter(dataloader))
    grid = utils.make_grid(images[:16], nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
    print(f"Image shape: {images[0].shape}")
    print(f"Value range: [{images[0].min():.2f}, {images[0].max():.2f}]")
    print(f"Batch preview time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    overall_start_time = time.time()
    dataloader = setup_pokemon_data(batch_size=32, num_workers=0)
    preview_batch(dataloader)
    print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")
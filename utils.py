import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os



class SpriteDataset(Dataset):
    """Sprite dataset class"""
    def __init__(self, root, transform, target_transform):
        self.images = np.load(os.path.join(root, "sprites_1788_16x16.npy"))
        self.labels = np.load(os.path.join(root, "sprite_labels_nc_1788_16x16.npy"))
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        label = self.target_transform(self.labels[idx])
        return image, label

    def __len__(self):
        return len(self.images)

def generate_animation(intermediate_samples, t_steps, fname, n_images_per_row=8):
    """Generates animation and saves as a gif file for given intermediate samples"""
    intermediate_samples = [make_grid(x, scale_each=True, normalize=True, 
                                      nrow=n_images_per_row).permute(1, 2, 0).numpy() for x in intermediate_samples]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis("off")
    img_plot = ax.imshow(intermediate_samples[0])
    
    def update(frame):
        img_plot.set_array(intermediate_samples[frame])
        ax.set_title(f"T = {t_steps[frame]}")
        fig.tight_layout()
        return img_plot
    
    ani = FuncAnimation(fig, update, frames=len(intermediate_samples), interval=200)
    ani.save(fname)


def get_custom_context(n_samples, n_classes, device):
    """Returns custom context in one-hot encoded form"""
    context = []
    for i in range(n_classes - 1):
        context.extend([i]*(n_samples//n_classes))
    context.extend([n_classes - 1]*(n_samples - len(context)))
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().to(device)


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CustomImageToImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, transform=None, target_transform=None):
        """
        Custom dataset for image-to-image tasks
        
        Args:
            input_dir (str): Directory with input images
            output_dir (str): Directory with corresponding output images
            transform (callable, optional): Transform to apply to input images
            target_transform (callable, optional): Transform to apply to output images
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all image filenames (assumes same names in both directories)
        self.image_files = [f for f in os.listdir(input_dir) 
                           if os.path.isfile(os.path.join(input_dir, f)) 
                           and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        
        # Verify all input images have corresponding output images
        for img_file in self.image_files:
            if not os.path.exists(os.path.join(output_dir, img_file)):
                raise ValueError(f"Missing output image for {img_file}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load input image
        input_img_path = os.path.join(self.input_dir, img_name)
        input_image = Image.open(input_img_path).convert('RGB')
        
        # Load output/target image
        output_img_path = os.path.join(self.output_dir, img_name)
        output_image = Image.open(output_img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            input_image = self.transform(input_image)
        
        if self.target_transform:
            output_image = self.target_transform(output_image)
        else:
            # If no target transform is provided, at least convert to tensor
            output_image = transforms.ToTensor()(output_image)
            
        return input_image, output_image
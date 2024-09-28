import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from PIL import Image
import os

# Configuration
BATCH_SIZE = 5  # Set batch size to 5 to match the number of images
OUTPUT_DIR = 'saved_images'  # Directory to save the images

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std (for MNIST)
])

mnist_dataset = datasets.MNIST(root='datasets/', train=True, transform=transform, download=True)

# Create a subset with only 5 samples
subset_indices = torch.arange(5)  # Select the first 5 samples
mnist_subset = Subset(mnist_dataset, subset_indices)

# Create a DataLoader for the subset
subset_loader = DataLoader(mnist_subset, batch_size=BATCH_SIZE, shuffle=True)

# Iterate through the subset to get the images and labels
for batch_idx, (data, target) in enumerate(subset_loader):
    print(f"Batch {batch_idx+1}:")
    print("Data shape:", data.shape)
    print("Target shape:", target.shape)
    break  # Just to see the first batch

# Save images to local directory
def save_images(images, labels, output_dir, ncols=5):
    for i, (img, label) in enumerate(zip(images, labels)):
        img = img.squeeze(0)  # Remove the channel dimension
        img = transforms.ToPILImage()(img)  # Convert tensor to PIL image
        img.save(os.path.join(output_dir, f'image_{i}_label_{label.item()}.png'))
        print(f'Saved image_{i}_label_{label.item()}.png')

# Save the images from the subset
save_images(data, target, OUTPUT_DIR)

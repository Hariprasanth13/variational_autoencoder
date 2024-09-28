import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from model import VariationalAutoEncoder


# configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20 # constitute for more compression

# Assume your model is already trained and saved as 'vae.pth'
# Load the trained model
model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)
model.load_state_dict(torch.load('vae.pth',weights_only=True))
model.eval()  # Set the model to evaluation mode

# Preprocessing: Define a transform to convert the input image to a tensor
transform = transforms.Compose([
    transforms.Grayscale(),  # Ensure the image is grayscale (MNIST is grayscale)
    transforms.Resize((28, 28)),  # Resize to 28x28 pixels (MNIST image size)
    transforms.ToTensor(),  # Convert image to PyTorch tensor
])

# Load and preprocess an image
img = Image.open(r'C:\Projects\vae\saved_images\image2.jpg')
img = transform(img).to(DEVICE)  # Apply transformations and move to the device

# Flatten the image (28x28 -> 784) and add a batch dimension (1 image)
img = img.view(1, INPUT_DIM)

# Forward pass through the VAE
with torch.no_grad():  # No need to compute gradients for inference
    reconstructed_img, _, _ = model(img)

# Reshape the images back to 28x28 for visualization
original_img = img.view(1, 1, 28, 28)
reconstructed_img = reconstructed_img.view(1, 1, 28, 28)

# Save or Display the original and reconstructed images side by side
comparison = torch.cat([original_img, reconstructed_img])

# Save the images for comparison
save_image(comparison.cpu(), 'reconstructed_image.png', nrow=2)

# Alternatively, display using matplotlib
plt.figure(figsize=(6, 3))

# Display Original Image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(original_img.squeeze(0).squeeze(0).cpu(), cmap='gray')

# Display Reconstructed Image
plt.subplot(1, 2, 2)
plt.title('Reconstructed Image')
plt.imshow(reconstructed_img.squeeze(0).squeeze(0).cpu(), cmap='gray')

plt.show()

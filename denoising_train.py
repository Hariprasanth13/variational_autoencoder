import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from model import VariationalAutoEncoder

# configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 32*32*3 # CIFAR-10 images are 32x32 with 3 channels
H_DIM = 512
Z_DIM = 64
BATCH_SIZE = 64
LR=1e-3
NUM_EPOCHS = 3


# Add noise to the image
def add_noise(images,noise_factor = 0.2):
    noisy_images = images+ noise_factor*torch.randn(*images.shape) #torch.randn(*images.shape) generates a random tensor with the same shape as images, with values drawn from a normal distribution (mean = 0, std = 1).
    noisy_images = torch.clip(noisy_images,0.,1.) # This step ensures that the pixel values of the noisy images stay within a valid range for image data (0 to 1). The pixel values are clipped so that:Any value below 0 becomes 0.Any value above 1 becomes 1.
    return noisy_images



# Prepare the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) # Normalize for CIFAR-10.transforms.Normalize(mean, std) normalizes a tensor image by subtracting the mean and dividing by the standard deviation for each channel. Subtracting 0.5 and dividing by 0.5 essentially scales the values from [0, 1] to [-1, 1]. This is a standard normalization technique to make the neural network’s training more stable by ensuring that the input values are centered around zero and are scaled to a small range.
])

train_dataset = datasets.CIFAR10(root='./data', train=True,transform = transform,download=True)
train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE,shuffle=True)

# Model and optimizer
model = VariationalAutoEncoder(INPUT_DIM,H_DIM,Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=LR)
loss_fn = nn.BCELoss(reduction="sum") #Binary cross entropy

#Training loop

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx,(data,_) in enumerate(train_loader):
        data = data.view(-1,INPUT_DIM).to(DEVICE)
        noise_data = add_noise(data)

        optimizer.zero_grad()
        reconstructed_data,mu,sigma = model(noise_data)

        # compute loss
        reconstruction_loss = loss_fn(reconstructed_data,data)
        #kl_div = -torch.sum(1+torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))


        # Backprop
        loss = reconstruction_loss + kl_div
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_loader.dataset)}')

    

def denoise_image(image):
    model.eval()
    with torch.no_grad():
        noisy_image = add_noise(image.view(-1,INPUT_DIM).to(DEVICE))
        denoised_image,_,_ = model(noisy_image)
        return denoised_image.view(3,32,32).cpu()

# Test with single image in CIFAR 10
test_image = train_dataset[0][0] 
denoised_image = denoise_image(test_image)

# Display the original, noisy, and denoised images
plt.subplot(1,3,1)
plt.title("original image")
plt.imshow(test_image.permute(1,2,0)) # permute() is used to change the order of dimensions of a tensor. test_image.permute(1, 2, 0) rearranges the dimensions from (C, H, W) to (H, W, C), making it compatible with plt.imshow() for visualization.

plt.subplot(1, 3, 2)
plt.title("Noisy Image")
plt.imshow(add_noise(test_image).permute(1, 2, 0))

plt.subplot(1, 3, 3)
plt.title("Denoised Image")
plt.imshow(denoised_image.permute(1, 2, 0))

plt.show()
    
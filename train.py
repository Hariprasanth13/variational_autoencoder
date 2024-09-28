import torch
import torchvision.datasets as datasets
from tqdm import tqdm # for progress bar
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms # For image augmentation
from torchvision.utils import save_image
from torch.utils.data import DataLoader # Gives easier dataset management by creating mini batches 
import matplotlib.pyplot as plt

# configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
H_DIM = 200
Z_DIM = 20 # constitute for more compression
NUM_EPOCHS = 3
BATCH_SIZE = 32
LR_RATE = 3e-4 # Karpathy constant

# Dataset loading
dataset = datasets.MNIST(root = "datasets/" , train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE,shuffle=True)
model = VariationalAutoEncoder(INPUT_DIM,H_DIM,Z_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(),lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum") #Binary cross entropy

# start training
for epoch in range(NUM_EPOCHS):
    loop = tqdm(enumerate(train_loader))
    for i,(x,_) in loop:
        #Forward pass
        x = x.to(DEVICE).view(x.shape[0],INPUT_DIM) # The view() function reshapes the tensor without changing its data. Here, you're reshaping each MNIST image (28x28) into a flat vector of length 784 (28 * 28 = 784)
        x_reconstructed,mu,sigma = model(x)

        # compute loss
        reconstruction_loss = loss_fn(x_reconstructed,x)
        #kl_div = -torch.sum(1+torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
        kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))


        # Backprop
        loss = reconstruction_loss + kl_div
        optimizer.zero_grad() # Resetting the gradients to zero before computing the next gradient
        loss.backward() #Computes the gradients of the loss function with respect to the model
        optimizer.step()
        loop.set_postfix(loss = loss.item())

        #Visualize 
        # if i % 500 == 0:  # Save images every 100 batches
        #     with torch.no_grad():
        #         comparison = torch.cat([x[:8].view(-1, 1, 28, 28), x_reconstructed[:8].view(-1, 1, 28, 28)])
        #         save_image(comparison.cpu(), f'reconstruction_epoch_{epoch}_batch_{i}.png', nrow=8)

# Save the trained model's state dict after training
#torch.save(model.state_dict(), 'vae.pth')

model.to("cpu")
def inference(digit,num_examples = 1):
    model.eval()
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx+=1
        if idx==10:
            break
    
    encoding_digits =[]
    for d in range(10):
        with torch.no_grad():
            mu,sigma = model.encode(images[d].view(1,784))
        encoding_digits.append((mu,sigma))
    
    mu,sigma = encoding_digits[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma* epsilon
        out = model.decode(z)
        out = out.view(-1,1,28,28)
        save_image(out,f"generated_{digit}_ex_{example}.png")
    
inference(3,5)





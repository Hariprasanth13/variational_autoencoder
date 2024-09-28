import torch
from torch import nn
import torch.nn.functional as F


# Input image -> hidden dim -> mean,std ->parameterization trick -> Decoder -> output image
class VariationalAutoEncoder(nn.Module):
    def __init__(self,input_dim,h_dim=200,z_dim=20):
        super().__init__()
        # encoder
        self.img_2hid = nn.Linear(input_dim,h_dim)
        self.hid_2mu = nn.Linear(h_dim,z_dim)
        self.hid_2sigma = nn.Linear(h_dim,z_dim)

        #decodermyenv
        self.z_2hid = nn.Linear(z_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim,input_dim)

        self.relu = nn.ReLU()
    
    def encode(self,x):
        #q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu , sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu,sigma

        

    def decode(self,z):
        # p theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h)) #Normalizing the data between 0 and 1


    def forward(self,x):
        mu,sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparameterized)
        return x_reconstructed,mu,sigma

    
if __name__=="__main__":
    x = torch.randn(4,28*28) # 4 values of 28*28 each
    vae = VariationalAutoEncoder(input_dim=784)
    x_reconstructed,mu,sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)

 
 
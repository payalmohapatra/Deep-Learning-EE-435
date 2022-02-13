import matplotlib.pyplot as plt
from numpy.core.fromnumeric import argmax
import autograd.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from skimage import exposure
from sklearn.datasets import fetch_openml
from optimizers_only import gradient_descent_nn
from optimizers_only import gradient_descent
import autograd.numpy as np
from autograd import value_and_grad 
from autograd import hessian
from autograd.misc.flatten import flatten_func
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from torch.utils.tensorboard import SummaryWriter
import tensorboard
writer = SummaryWriter()
# import data
datapath = 'Data/'
X = np.loadtxt(datapath + 'universal_autoencoder_samples.txt', delimiter=',')
print(np.shape(X))
plt.scatter(X[0,:], X[1,:], c = 'k', s = 60, linewidth = 0.75, edgecolor = 'w')
plt.title('Original Data')
plt.xlabel('x1')
plt.ylabel('x2')
#plt.show()
batch_size = 100
X = X.T
x = torch.from_numpy(X)

class GenDataset(Dataset) :
    def __init__(self,x) :
        # data loading
        self.x = x
        
    def __getitem__(self,index) :
        return self.x[index]
    
    def __len__(self) :    
        return len(self.x)      

dataset = GenDataset(x)

loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

# Creating a PyTorch class
# 1 * 2 ==> 1 ==> 1 * 2
class AE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		
		# Building an linear encoder with Linear
		# layer followed by Relu activation function
		# 2 ==> 1
		self.encoder = torch.nn.Sequential(
			torch.nn.Linear(2, 10),
			torch.nn.Tanh(),
			torch.nn.Linear(10, 10),
			torch.nn.Tanh(),
			torch.nn.Linear(10, 1)
		)
		
		# Building an linear decoder with Linear
		# layer followed by Relu activation function
		# The Sigmoid activation function
		# outputs the value between 0 and 1
		# 1 ==> 2
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(1, 10),
			torch.nn.Tanh(),
			torch.nn.Linear(10, 10),
			torch.nn.Tanh(),
			torch.nn.Linear(10, 2)
			#torch.nn.ReLU()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		#print('Encoded =', encoded)
		decoded = self.decoder(encoded)
		#print('Decoded =',decoded)
		return decoded
# Model Initialization
model = AE()
print(model)

# Validation using MSE Loss function
loss_function = torch.nn.MSELoss().float()
loss_function = loss_function.float()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-2)
epochs = 5000
outputs = []
losses = []
#reconstructed_hist = []
# Train the model
for epoch in range(epochs):
    n_correct = 0
    reconstructed_hist = []
    for (features) in loader: 
        # Forward pass
        reconstructed = model(features.float())
        loss = loss_function(reconstructed.float(), features.float())
        loss = loss.float()
        writer.add_scalar("Loss/train", loss, epoch)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Storing the losses in a list for plotting
        losses.append(loss.detach().numpy())
        reconstructed_np = reconstructed.detach().numpy()
        reconstructed_hist.append(reconstructed_np)

    outputs.append((epochs, reconstructed))    

print(np.shape(reconstructed_hist))
reconstructed_arr = np.array(reconstructed_hist)
reconstructed_arr = np.reshape(reconstructed_arr, (100,2))
print(np.shape(reconstructed_arr))
plt.figure(3)
plt.scatter(reconstructed_arr[:,0], reconstructed_arr[:,1])
plt.title('Decoded Data')
plt.xlabel('x1')
plt.ylabel('x2')

plt.figure(2)
plt.plot(losses)
plt.title('Cost Function')
plt.xlabel('Iteration')
plt.ylabel('Cost history')
plt.show()
reconstrcuted_test_hist = []
with torch.no_grad():
    n_correct = 0
    for (features) in loader:    
        reconstructed_test = model(features.float()) 
        reconstrcuted_test_hist.append(reconstructed_test)

print(len(reconstrcuted_test_hist))
writer.close()
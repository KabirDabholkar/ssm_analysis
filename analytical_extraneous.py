import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

true_latent_size = 10
num_heldin = 50
num_heldout = 50
total_neurons = num_heldin + num_heldout
train_size = 20

np.random.seed(0)

W = np.random.normal(0,1, (total_neurons, true_latent_size))

Z = np.random.normal(0, 1, (train_size,true_latent_size))
X = Z @ W.T

print(X.shape)

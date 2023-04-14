import numpy as np
from tqdm import tqdm
from math import ceil
from chlnet import CHLNet

# load MNIST images as normalized, flattened vectors
with np.load("mnist.npz") as data:
    x_train, x_test = data["x_train"] / 255, data["x_test"] / 255
    x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
    y_train, y_test = data["y_train"], data["y_test"]
    y_train, y_test = np.eye(10)[y_train], np.eye(10)[y_test] # to one-hot

# create the network
layer_sizes = [784, 64, 64, 64, 10]
net = CHLNet(layer_sizes, gamma=0.1, lr=0.05)

# define simple training hyperparams
batch_size = 8
num_batches = ceil(len(x_train) / batch_size)
num_epochs = 1

# train the network using CHL
for i in range(num_epochs):
    perm = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]
    for j in tqdm(range(num_batches), desc=f"Epoch {i+1}/{num_epochs}"):
        x_batch = x_train[j*batch_size:(j+1)*batch_size]
        y_batch = y_train[j*batch_size:(j+1)*batch_size]
        free_eq = net.free_phase(x_batch)
        clamped_eq = net.clamped_phase(x_batch, y_batch)
        net.update(free_eq, clamped_eq)

# get predictions on test set
preds = np.argmax(net.free_phase(x_test)[-1], axis=1)
truth = np.argmax(y_test, axis=1)

# calculate accuracy on test set
num_samples = len(x_test)
num_wrong = np.count_nonzero(preds-truth)
print("Accuracy:", (num_samples - num_wrong) / num_samples)
import numpy as np

class CHLNet:
    def __init__(self, layer_sizes, gamma, lr):
        self.layer_sizes = layer_sizes
        self.gamma, self.lr = gamma, lr
        self.W = [np.random.normal(0, 0.1, size=(i, o))
                  for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.random.normal(0, 0.1, size=(1, i))
                  for i in layer_sizes[1:]]
        self.L = len(layer_sizes)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def free_phase(self, x0, T=50):
        """
        The free phase runs for `T` iterations, each time running with
        the same image(s) as input. Since there are feedback connections (i.e.
        the activations in layer k at time t are both a function of the
        activations in layer k-1 at time t *and* layer k+1 at time t-1), the
        activations differ at every pass but converge to an equilibrium. The
        default value of 50 should be enough to reach equilibrium in most cases.
        
        Args:
            x0: The input (in this case flattened MNIST images)
            T: The number of iterations (default=50)
        """
        x = [x0] + [np.zeros((len(x0), i)) for i in self.layer_sizes[1:]]
        for _ in range(T):
            for k in range(1, self.L):
                d_x = x[k-1] @ self.W[k-1] + self.b[k-1]
                if k < self.L - 1:
                    d_x += self.gamma * x[k+1] @ self.W[k].T
                d_x = self.sigmoid(d_x)
                x[k] += -x[k] + d_x
        return x
                
    def clamped_phase(self, x0, y, T=50):
        """
        The clamped phase is similar to the free phase except the output units
        are held fixed ("clamped") to their target values. The rest is the same
        as in the free phase.
        """
        x = [x0] + [np.zeros((len(x0), i)) for i in self.layer_sizes[1:-1]] + [y]
        for _ in range(T):
            for k in range(1, self.L-1):
                d_x = x[k-1] @ self.W[k-1] + self.b[k-1]
                d_x += self.gamma * x[k+1] @ self.W[k].T
                d_x = self.sigmoid(d_x)
                x[k] += -x[k] + d_x
        return x
    
    def update(self, x_free, x_clamped):
        """
        To update the parameters of the network, we need the equilibrium state
        (i.e. all of the final activations) from both the free phase and the
        clamped phase. Note that the deltas for the weights and biases between
        layer k and layer k+1 only depend on the activations in the two layers.
        There is no global loss function or backpropagation.
        """
        num_samples = len(x_free[0])
        for k in range(self.L-1):
            coeff = self.lr * self.gamma**(k-(self.L-1)) / num_samples
            self.W[k] += coeff * (x_clamped[k].T @ x_clamped[k+1] - x_free[k].T @ x_free[k+1])
            self.b[k] += coeff * np.mean(x_clamped[k+1] - x_free[k+1], axis=0)

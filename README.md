# Contrastive Hebbian Learning

A simple implementation of contrastive Hebbian learning (CHL) using a small neural network. CHL, and Hebbian learning more generally, is an alternative to backpropagation, which doesn't rely on a global loss nor error derivatives. Instead, each parameter is updated using a local learning rule that only takes into account activations in a neuron's immediate neighborhood.

CHL proceeds in two phases:

1. **The free phase** runs over $1,...,T$ time steps, each time taking the same values as input. At $t=1$, the activations of the network are identical to that of a standard MLP. For $t>1$, the activations in layer $k$ are a function of the activations in layer $k-1$ (as usual) but also the activations of layer $k+1$ in the previous time step. This means that the activations in every pass are different despite being fed the same input. However, it can be shown that they still converge to an equilibrium. $T$ is picked such that the network has enough time for this to be the case[^1].
2. **The clamped phase** is identical to the free phase except the output units are held fixed ("clamped") to their target values.

The equilibrium states of both phases are stored and used to compute the deltas for weights and biases of the network. When you do this enough times, the parameters should converge to a configuration that maps to inputs to the desired target states. For more information, see the following papers:

[^1]: Instead of picking a fixed number of iterations, it is also possible to do early stopping once equilibrium has been reached.

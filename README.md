# Contrastive Hebbian Learning

A simple implementation of contrastive Hebbian learning (CHL) using a small neural network. CHL, and Hebbian learning more generally, is an alternative to backpropagation, which doesn't rely on a global loss nor error derivatives. Instead, each parameter is updated using a local learning rule that only takes into account activations in a neuron's immediate neighborhood.

CHL proceeds in two phases:

1. **The free phase** runs over $1,...,T$ time steps, each time taking the same value(s) as input. At $t=1$, the activations of the network are identical to that of a standard MLP. For $t>1$, the activations in layer $k$ are a function of the activations in layer $k-1$ (as usual) but also the activations of layer $k+1$ in the previous time step. This is due to the network's symmetric "feedback" weights, i.e. if $W_k$ are the weights from layer $k-1$ to layer $k$, $W_k^\text{T}$ are the weights from layer $k$ to layer $k-1$. The presence of feedback connections means that the activations in every pass are different despite the network being fed the same input. However, it can be shown that they still converge to an equilibrium. $T$ is picked such that the network has enough time for this to be the case[^1].
2. **The clamped phase** is identical to the free phase except the output units are held fixed ("clamped") to their target values. In this phase, it is thus only the hidden units that are updated.

The equilibrium states of both phases are stored and used to compute the deltas for the weights and biases of the network. When you do this enough times, the parameters should converge to a configuration that maps new inputs to their desired target states. I found the following papers helpful in understanding and implementing the algorithm:

* J. R. Movellan (1991): [Contrastive Hebbian Learning in the Continuous Hopfield Model](https://inc.ucsd.edu/mplab/46/media/CHL90.pdf)
* Xie & Seung (2003): [Equivalence of Backpropagation and Contrastive Hebbian Learning in a Layered Network](https://www.researchgate.net/publication/10896451_Equivalence_of_Backpropagation_and_Contrastive_Hebbian_Learning_in_a_Layered_Network)
* Detorakis et al. (2018): [Contrastive Hebbian Learning with Random Feedback Weights](https://arxiv.org/abs/1806.07406)

[^1]: Instead of picking a fixed number of iterations, it is also possible to do early stopping once equilibrium has been reached.

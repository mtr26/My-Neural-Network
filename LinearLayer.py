import numpy as np
import lib


class Linear(lib.Layer):
    """
    Linear layer.
    """
    def __init__(self, input_dim : int, output_dim : int, activation : str ='linear'):
        assert activation in lib.ACTIVATION_FUNC.keys()
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.weight = np.random.uniform(low=-np.sqrt(1/input_dim), high=np.sqrt(1/input_dim), size=(input_dim, output_dim))
        self.biases = np.random.uniform(low=-np.sqrt(1/input_dim), high=np.sqrt(1/input_dim), size=(output_dim, 1))
        self.last_input = None
        self.last_output = None
        self.last_z = None
        self.gradient_cache = None
        self.activation = activation

    def backward(self, grad_output, lr):
        """
        Compute the gradient of the loss with respect to the input of the layer.
        """
        # Compute gradient of the loss with respect to the linear combination (z)
        grad_z = grad_output * lib.ACTIVATION_FUNC_DERIVITIVE[self.activation](self.last_z)
        
        # Compute gradients with respect to weights and biases
        grad_weight = self.last_input.T @ grad_z
        grad_bias = np.sum(grad_z, axis=0)
        
        # Reshape grad_bias to match the shape of self.biases
        grad_bias = grad_bias.reshape(self.biases.shape)
        
        # Update weights and biases
        self.weight -= lr * grad_weight
        self.biases -= lr * grad_bias
        
        # Compute gradient to propagate to the previous layer
        self.gradient_cache = grad_z @ self.weight.T
        return self.gradient_cache

    def params_nbr(self):
        """
        Return the number of parameters of the layer.
        """
        return self.input_dim * self.out_dim + self.out_dim

    def forward(self, x):
        """
        Compute the output of the layer
        """
        self.last_input = x  # x: (batch_size, input_dim)
        self.last_z = x @ self.weight + self.biases.T  # Align dimensions
        self.last_output = lib.ACTIVATION_FUNC[self.activation](self.last_z)
        return self.last_output

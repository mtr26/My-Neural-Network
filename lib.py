import numpy as np

class Layer:
    def __init__(self):
        self.forward

    def forward(self, x):
        pass

    def backward(self, grad_output):
        pass


class NN:
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def params_nbr(self):
        variables = filter((lambda x : isinstance(x, Layer)), self.__dict__.values())
        return sum([layer.params_nbr() for layer in variables])

    def backward(self, grad_output, lr):
        variables = list(filter((lambda x : isinstance(x, Layer)), self.__dict__.values()))
        
        for layer in reversed(variables):
            grad_output = layer.backward(grad_output, lr)
        



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def __sigmoid__derivitive__(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mean_square_error(output, expected_output):
    loss = np.mean((output - expected_output) ** 2)
    grad = 2 * (output - expected_output) / output.size
    return loss, grad



ACTIVATION_FUNC = {
    'linear' : (lambda x : x),
    'sigmoid' : sigmoid,
    'tanh' : ...,
    'ReLu' : ...
}


ACTIVATION_FUNC_DERIVITIVE = {
    'linear' : (lambda x : np.ones_like(x)),
    'sigmoid' : __sigmoid__derivitive__,
    'tanh' : ...,
    'ReLu' : ...
}
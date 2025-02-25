from LinearLayer import Linear
from lib import NN, mean_square_error
import numpy as np

class Model(NN):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = Linear(input_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, hidden_dim)
        self.l3 = Linear(hidden_dim, output_dim, activation='sigmoid')

    def forward(self, x):
        out = self.l1.forward(x)
        out = self.l2.forward(out)
        out = self.l3.forward(out)
        return out
    


model = Model(3, 128, 1)


# Should predict the last digit of the array (e.g [O, 1, 1] returns 1)
training_sample = np.array([[0, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0],
                            [0, 1, 1],
                            [1, 1, 1]])


Y = np.array([0, 1, 0, 1, 1]).reshape(-1, 1)



test_sample = np.array([[1, 0, 0],
                        [1, 0, 1]])


# Training loop
for epoch in range(10000):
    out = model.forward(training_sample)
    loss, grad = mean_square_error(out, Y)
    model.backward(grad, 1e-3)
    if epoch % 100 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")


print(model.forward(test_sample))

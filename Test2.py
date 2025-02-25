import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.out(out)
        return out
    

net = Network(2, 3, 4)

X = th.tensor([2, 1]).float()


print(net(X))



optimizer = optim.Adam(net.parameters())



for epoch in range(1000):
    optimizer.zero_grad()
    out = net(X)
    loss = F.mse_loss(out.float(), th.tensor([1, 2, 3, 4]).float())
    loss.backward()
    optimizer.step()



print(net(X))





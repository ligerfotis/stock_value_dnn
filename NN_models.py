from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim=28, l1_size=512, l2_size=512, output_size=10):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(input_dim, l1_size), nn.ReLU(),
                                               nn.Linear(l1_size, l2_size), nn.ReLU(),
                                               nn.Linear(l2_size, output_size))

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
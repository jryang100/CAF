import torch.nn as nn
import torch.nn.functional as F


class MLPNet(nn.Module):
    """
    NoDebias network
    """

    def __init__(self, configs):
        super(MLPNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        self.num_classes = configs["num_classes"]

    def forward(self, inputs):
        x = inputs
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x = self.softmax(x)
        return F.log_softmax(x, dim=1)


class RegNet(nn.Module):
    """
    network with fair constraint loss
    """

    def __init__(self, configs):
        super(RegNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        self.softmax_1 = nn.Linear(self.num_neurons[-1], 1)
        self.num_classes = configs["num_classes"]

    def forward(self, inputs):
        x = inputs
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        x_2 = self.softmax(x)
        x_1 = self.softmax_1(x)
        return F.log_softmax(x_2, dim=1), x_1

    def inference(self, inputs):
        x = inputs
        for hidden in self.hiddens:
            x = F.relu(hidden(x))
        logprobs = F.log_softmax(self.softmax(x), dim=1)
        return logprobs

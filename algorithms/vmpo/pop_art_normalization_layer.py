import math
from numpy import mat
import torch


class PopArtLayer(torch.nn.Module):
    def __init__(self, input_dim=256, output_dim=1, beta=1e-4):
        super(PopArtLayer, self).__init__()

        self.beta = beta
        self.input_dim = input_dim
        self.output_dim = output_dim

        # define weights and bias
        self.weights = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(output_dim))

        # register mu and sigma
        self.register_buffer('mu', torch.zeros(
            output_dim, requires_grad=False))
        self.register_buffer('sigma', torch.ones(
            output_dim, requires_grad=False))

        # reset all parameters
        self.reset_parameters()

    def forward(self, input, task=None):
        if len(input.shape) == 2:
            input = input.unsqueeze(-1)
        input_shape = input.shape
        input = input.reshape(-1, self.input_dim)

        normalized_output = input.mm(self.weights.t())
        normalized_output += self.bias.unsqueeze(
            0).expand_as(normalized_output)
        normalized_output = normalized_output.reshape(
            *input_shape[:-1], self.output_dim)

        with torch.no_grad():
            output = normalized_output * self.sigma + self.mu

        if task is not None:
            output = output.gather(-1, task.unsqueeze(-1))
            normalized_output = normalized_output.gather(
                -1, task.unsqueeze(-1))

        return [output.squeeze(-1), normalized_output.squeeze(-1)]

    def reset_parameters(self):
        # initialize weights according to Kaiming definition
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

        # initialize bias
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.weights)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def normalize(self, inputs, task=None):
        """
        task: task ids
        """
        task = torch.zeros(
            inputs.shape, dtype=torch.int64) if task is None else task
        input_device = inputs.device
        inputs = inputs.to(self.mu.device)
        mu = self.mu.expand(
            *inputs.shape, self.output_dim).gather(-1, task.unsqueeze(-1)).squeeze(-1)
        sigma = self.sigma.expand(
            *inputs.shape, self.output_dim).gather(-1, task.unsqueeze(-1)).squeeze(-1)
        output = (inputs - mu) / sigma
        return output.to(input_device)

    @torch.no_grad()
    def update_parameters(self, vs, task):
        """
        task: one hot vector of tasks
        """
        vs, task = vs.to(self.mu.device), task.to(self.mu.device)

        oldmu = self.mu
        oldsigma = self.sigma

        vs = vs * task
        n = task.sum((0, 1))
        mu = vs.sum((0, 1)) / n
        nu = torch.sum(vs ** 2, (0, 1)) / n
        sigma = torch.sqrt(nu - mu ** 2)
        sigma = torch.clamp(sigma, min=1e-2, max=1e+6)

        mu[torch.isnan(mu)] = self.mu[torch.isnan(mu)]
        sigma[torch.isnan(sigma)] = self.sigma[torch.isnan(sigma)]

        self.mu = (1 - self.beta) * self.mu + self.beta * mu
        self.sigma = (1 - self.beta) * self.sigma + self.beta * sigma

        self.weights.data = (self.weights.t() * oldsigma / self.sigma).t()
        self.bias.data = (oldsigma * self.bias + oldmu - self.mu) / self.sigma

    def state_dict(self):
        return dict(mu=self.mu,
                    sigma=self.sigma,
                    weights=self.weights.data,
                    bias=self.bias.data)

    def load_state_dict(self, state_dict):
        with torch.no_grad():
            self.mu = state_dict['mu']
            self.sigma = state_dict['sigma']
            self.weights.data = state_dict['weights']
            self.bias.data = state_dict['bias']

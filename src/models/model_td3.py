import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim=1, 
        hidden_dim=[100, 100], 
        act=nn.ReLU(),
    ):

        super(Actor, self).__init__()

        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        else:
            net = [nn.Linear(state_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

        self.net = nn.Sequential(*net)

    def forward(self, state):
        a = self.net(state)
        return torch.sigmoid(a)


class Critic(nn.Module):
    def __init__(self, 
        state_dim, 
        action_dim=1, 
        hidden_dim=[100, 100], 
        act=nn.ReLU(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):

        super(Critic, self).__init__()
    
        self.device = device
        
        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

        self.net1 = nn.Sequential(*net)


        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

        self.net2 = nn.Sequential(*net)

    def forward(self, state, action):
        return (
            self.net1(torch.cat([state, action], 1)), 
            self.net2(torch.cat([state, action], 1)),
        )

    def q1(self, state, action):
        return self.net1(torch.cat([state, action], 1))
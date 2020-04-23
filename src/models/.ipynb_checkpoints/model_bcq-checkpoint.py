import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim=1, 
        phi=0.05, 
        hidden_dim=[100, 100], 
        act=nn.LeakyReLU(),
    ):

        super(Actor, self).__init__()
        self.phi = phi

        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]
            
        elif len(hidden_dim) == 1:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]
            net += [nn.Linear(hidden_dim[0], action_dim)]
            
        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

        self.net = nn.Sequential(*net)

    def forward(self, state, action):
        a = self.net(torch.cat([state, action], 1))
        a = self.phi * torch.tanh(a)
        return (a + action).clamp(min=0)


class Critic(nn.Module):
    def __init__(self, 
        state_dim, 
        action_dim=1, 
        hidden_dim=[100, 100], 
        act=nn.LeakyReLU(),
        num_heads=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ):

        super(Critic, self).__init__()
    
        self.device = device
        #self.num_heads = num_heads
        
        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        elif len(hidden_dim) == 1:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]
            net += [nn.Linear(hidden_dim[0], action_dim)]
            
        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

        self.net1 = nn.Sequential(*net)


        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        elif len(hidden_dim) == 1:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]
            net += [nn.Linear(hidden_dim[0], action_dim)]

        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[-1], action_dim)]

                

        self.net2 = nn.Sequential(*net)

    def forward(self, state, action):
        #convex1, convex2 = torch.rand(self.num_heads), torch.rand(self.num_heads)
        #convex1, convex2 = convex1/convex1.sum(), convex2/convex2.sum()
        return (
            self.net1(torch.cat([state, action], 1)), #convex1.to(self.device)).sum(dim=1,keepdim=True),
            self.net2(torch.cat([state, action], 1)), #convex2.to(self.device)).sum(dim=1,keepdim=True),
        )

    def q1(self, state, action):
        #convex1 = torch.rand(self.num_heads)
        #convex1 = convex1/convex1.sum()
        return self.net1(torch.cat([state, action], 1)) #convex1.to(self.device)).sum(dim=1,keepdim=True)


class VAE(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim=1,
        hidden_dim=[100, 100],
        latent_dim=10,
        act=nn.LeakyReLU(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        super(VAE, self).__init__()

        self.device = device
        self.latent_dim = latent_dim

        ### Encoder ###
        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + action_dim, hidden_dim), act]

            self.mean = nn.Linear(hidden_dim, latent_dim)
            self.log_std = nn.Linear(hidden_dim, latent_dim)

        elif len(hidden_dim) == 1:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            self.mean = nn.Linear(hidden_dim[0], latent_dim)
            self.log_std = nn.Linear(hidden_dim[0], latent_dim)
        else:
            net = [nn.Linear(state_dim + action_dim, hidden_dim[0]), act]

            for d1, d2 in zip(hidden_dim, hidden_dim[1:]):
                net += [nn.Linear(d1, d2), act]

            self.mean = nn.Linear(hidden_dim[-1], latent_dim)
            self.log_std = nn.Linear(hidden_dim[-1], latent_dim)

        self.encoder = nn.Sequential(*net)

        ### Decoder ###

        if isinstance(hidden_dim, int):
            net = [nn.Linear(state_dim + latent_dim, hidden_dim), act]
            net += [nn.Linear(hidden_dim, action_dim)]

        else:
            net = [nn.Linear(state_dim + latent_dim, hidden_dim[-1]), act]

            for d1, d2 in zip(reversed(hidden_dim), reversed(hidden_dim[:-1])):
                net += [nn.Linear(d1, d2), act]

            net += [nn.Linear(hidden_dim[0], action_dim)]

        self.decoder = nn.Sequential(*net)

    def forward(self, state, action):
        z = self.encoder(torch.cat([state, action], 1))

        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)

        z = mean + std * torch.FloatTensor(
            np.random.normal(0, 1, size=(std.size()))
        ).to(self.device)

        u = self.decode(state, z)

        return u, mean, std

    def decode(self, state, z=None):
        if z is None:
            z = (
                torch.FloatTensor(
                    np.random.normal(0, 1, size=(state.size(0), self.latent_dim))
                )
                .clamp(-0.5, 0.5)
                .to(self.device)
            )

        a = self.decoder(torch.cat([state, z], 1))
        return a
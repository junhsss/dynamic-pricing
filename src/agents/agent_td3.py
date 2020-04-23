import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import torch
import wandb
import pandas as pd
import random
import gc

from src.models.model_td3 import Actor, Critic

class TD3(object):
    def __init__(
        self,
        state_dim=52,
        actor_hidden_dim=[300, 200],
        critic_hidden_dim=[300, 200],
        actor_lr=0.001,
        critic_lr=0.001,
        actor_decay=0.004,
        critic_decay=0.000001,
        discount=0.99,
        tau=0.005,
        epsilons = [0.005,0.01,0.02,0.05],
        #Twin Delayed Parameters
        policy_delay = 2,
        noise_std = 0.02,
        noise_clip = 0.01,
        batch_size=1024,
        device='cuda',
    ):
        
        self.device = device
        self.epsilons = epsilons
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size
        
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        
        self.actor = Actor(
            state_dim, action_dim=1,  hidden_dim=actor_hidden_dim, 
        ).to(self.device)
        self.actor_target = Actor(
            state_dim, action_dim=1, hidden_dim=actor_hidden_dim, 
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay
        )

        self.critic = Critic(state_dim, action_dim=1, hidden_dim=critic_hidden_dim,device=self.device).to(
            self.device
        )
        self.critic_target = Critic(state_dim, hidden_dim=critic_hidden_dim,device=self.device).to(
            self.device
        )
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay
        )

        self.read_data()
        self.make_dataloader()

    def read_data(self):

        self.actions = torch.tensor(pd.read_csv("./data/preprocessed/t_a.csv").values,dtype=torch.float32)
        self.states = torch.tensor(pd.read_csv("./data/preprocessed/t_s.csv").values,dtype=torch.float32)
        self.next_states = torch.tensor(pd.read_csv("./data/preprocessed/t_ns.csv").values,dtype=torch.float32)
        self.rewards = torch.tensor(pd.read_csv("./data/preprocessed/t_r.csv").values,dtype=torch.float32)

        self.action_evaluation = torch.tensor(pd.read_csv("./data/preprocessed/v_a.csv").values,dtype=torch.float32)
        self.state_evaluation = torch.tensor(pd.read_csv("./data/preprocessed/v_s.csv").values,dtype=torch.float32)
        self.next_state_evaluation = torch.tensor(pd.read_csv("./data/preprocessed/v_ns.csv").values,dtype=torch.float32)
        self.reward_evaluation = torch.tensor(pd.read_csv("./data/preprocessed/v_r.csv").values,dtype=torch.float32)

    def clear_data(self):

        del self.actions
        del self.states
        del self.next_states
        del self.rewards

        del self.action_evaluation
        del self.state_evaluation
        del self.next_state_evaluation
        del self.reward_evaluation
        gc.collect()

    def make_dataloader(self):

        class Tuples(torch.utils.data.Dataset):
            def __init__(self,
                         states,
                         actions,
                         rewards,
                         next_states,
                         device):
                super().__init__()
                self.states = states
                self.actions = actions
                self.rewards = rewards
                self.next_states = next_states
                self.device = device

            def __len__(self):
                return len(self.states)
            def __getitem__(self,idx):
                return self.states[idx], \
                       self.actions[idx],\
                       self.rewards[idx], \
                       self.next_states[idx]

        self.train_loader= torch.utils.data.DataLoader(Tuples(self.states,
                                                                self.actions,
                                                                self.rewards,
                                                                self.next_states,
                                                                self.device), 
                                                           batch_size = self.batch_size,
                                                           pin_memory = True,
                                                           shuffle=True)

        self.valid_loader= torch.utils.data.DataLoader(Tuples(self.state_evaluation,
                                                                self.action_evaluation,
                                                                self.reward_evaluation,
                                                                self.next_state_evaluation,
                                                                self.device),
                                                            batch_size = len(self.state_evaluation), 
                                                            shuffle = False)
        self.clear_data()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1))
            action = self.actor(state)
        return action.cpu().data.numpy().flatten()

    def train(self):
        running_critic_loss = 0
        running_actor_loss = 0

        for it, (s, a, r, ns) in enumerate(self.train_loader):
            s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
            with torch.no_grad():
                sampled_action = self.actor_target(ns)
                noise = (torch.randn_like(sampled_action)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)

                target_Q1, target_Q2 = self.critic_target(
                    ns, (sampled_action + noise).clamp(0, 1)
                )

                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = r + self.discount * target_Q

            current_Q1, current_Q2 = self.critic(s, a)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            if it % self.policy_delay == 0:

                actor_loss = -self.critic.q1(s, self.actor(s)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data
                    )

            running_critic_loss += critic_loss.cpu().item()

            if it % self.policy_delay == 0:
                running_actor_loss += actor_loss.cpu().item()

        return {
            "Critic_Loss": running_critic_loss / len(self.train_loader),
            "Actor_Loss": running_actor_loss / ( len(self.train_loader) // self.policy_delay),
        }

    def evaluate(self):
        R = [0, 0, 0, 0]
        N = [0, 0, 0, 0]

        for it, (s, a, r, ns) in enumerate(self.valid_loader):
            s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
            with torch.no_grad():
                sampled_action = self.actor(s)
                td_error = (
                    self.critic.q1(s, a)
                    - self.discount
                    * self.critic.q1(ns, sampled_action)
                    - r
                ).abs().mean()

                for i, epsilon in enumerate(self.epsilons): 
                    R[i] += (r[torch.abs(sampled_action - a) < epsilon]).sum()
                    N[i] += (r[torch.abs(sampled_action - a) < epsilon]).numel()
        
        #
            data = np.concatenate(
                (a.cpu().numpy(), \
                 (sampled_action).cpu().numpy())
               ,axis=1)
            
            table = wandb.Table(
                data=data.tolist()[:50],
                columns=[
                    "Actual Price",
                    "Sampled Price",
                ]
            )
        
        return {
            "Temporal_Difference_Error": td_error,
            "Reward_1": R[0] / N[0],
            "Reward_2": R[1] / N[1],
            "Reward_3": R[2] / N[2],
            "Reward_4": R[3] / N[3],
            "N_1":N[0] / len(self.valid_loader.dataset),
            "N_2":N[1]/ len(self.valid_loader.dataset),
            "N_3":N[2]/ len(self.valid_loader.dataset),
            "N_4":N[3]/ len(self.valid_loader.dataset),
            "Table":table
        }

    def step(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

        train_metrics = self.train()

        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

        eval_metrics = self.evaluate()

        train_metrics.update(eval_metrics)

        return train_metrics

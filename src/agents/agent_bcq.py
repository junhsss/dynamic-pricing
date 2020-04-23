import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import torch
import wandb
import pandas as pd
import random
import gc
from tqdm import tqdm

from src.models.model_bcq import Actor, Critic, VAE


class BCQ(object):
    def __init__(
        self,
        state_dim=52,
        actor_hidden_dim=[300, 200],
        critic_hidden_dim=[300, 200],
        vae_hidden_dim=[300, 200],
        vae_latent_dim=50,
        actor_lr=0.001,
        critic_lr=0.001,
        vae_lr=0.001,
        actor_decay=0.004,
        critic_decay=0.000001,
        vae_decay=0.0001,
        discount=0.99,
        tau=0.005,
        phi=0.01,
        epsilons = [0.005,0.01,0.02,0.05],
        lamb=0.75,
        #Twin Delayed Parameters
        policy_delay = 4,
        noise_std = 0.000,
        noise_clip = 0.01,
        batch_size=1024,
        clip_grad_norm= 1e6,
        reward_shaping='False',
        device='cuda'
    ):
        
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_latent_dim = vae_latent_dim
        self.vae_lr = vae_lr
        self.vae_decay = vae_decay

        self.device = device
        self.epsilons = epsilons
        self.lamb = lamb
        self.discount = discount
        self.tau = tau
        self.phi = phi
        self.batch_size = batch_size
        self.reward_shaping = reward_shaping
        self.clip_grad_norm = clip_grad_norm
        
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        
        self.actor_lr = actor_lr
        self.actor_decay = actor_decay
        
        self.critic_lr = critic_lr
        self.critic_decay = critic_decay
        
        self.actor = Actor(
            state_dim, action_dim=1, phi=phi, hidden_dim=actor_hidden_dim, 
        ).to(self.device)
        self.actor_target = Actor(
            state_dim, action_dim=1, phi=phi, hidden_dim=actor_hidden_dim, 
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

        self.vae = VAE(
            state_dim,
            action_dim=1,
            hidden_dim=vae_hidden_dim,
            latent_dim=vae_latent_dim,
            device=self.device,
        ).to(self.device)
        self.vae_optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=vae_lr, weight_decay=vae_decay
        )

        self.read_data()
        self.make_dataloader()
        self.name_vae()
        self.name_agent()

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
                         training,
                         reward_shaping,
                         device):
                super().__init__()
                self.states = states
                self.actions = actions
                self.next_states = next_states
                if training and reward_shaping:
                    self.rewards = torch.sign(rewards)
                else:
                    self.rewards = rewards
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
                                                                True,
                                                                self.reward_shaping,
                                                                self.device,), 
                                                           batch_size = self.batch_size,
                                                           pin_memory = True,
                                                           shuffle=True)

        self.valid_loader= torch.utils.data.DataLoader(Tuples(self.state_evaluation,
                                                                self.action_evaluation,
                                                                self.reward_evaluation,
                                                                self.next_state_evaluation,
                                                                False,
                                                                self.reward_shaping,
                                                                self.device),
                                                            batch_size = len(self.state_evaluation), 
                                                            shuffle = False)
        self.clear_data()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(10, 1)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            ind = q1.max(0)[1]
        return action[ind].cpu().data.numpy().flatten()

    def train(self):
        running_critic_loss = 0
        running_actor_loss = 0
        for it, (s, a, r, ns) in enumerate(self.train_loader):
            s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
            with torch.no_grad():
                state_rep = torch.FloatTensor(
                    np.repeat(ns.cpu().numpy(), 10, axis=0)
                ).to(self.device)

                sampled_action = self.actor_target(state_rep, self.vae.decode(state_rep))
                
                target_Q1, target_Q2 = self.critic_target(
                    state_rep, sampled_action + (torch.randn_like(sampled_action)*self.noise_std).clamp(-self.noise_clip, self.noise_clip)
                )

                target_Q = self.lamb * torch.min(target_Q1, target_Q2) + (1-self.lamb) * torch.max(
                    target_Q1, target_Q2
                )
                target_Q = target_Q.view(len(s), -1).max(1)[0].view(-1, 1)

                target_Q = r + self.discount * target_Q

            current_Q1, current_Q2 = self.critic(s, a)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                current_Q2, target_Q
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.clip_grad_norm
            )
            self.critic_optimizer.step()

            if it % self.policy_delay == 0:
                sampled_actions = self.vae.decode(s)
                perturbed_actions = self.actor(s, sampled_actions)

                actor_loss = -self.critic.q1(s, perturbed_actions).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.clip_grad_norm
                )
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
        R = [0, 0, 0, 0, 0]
        N = [0, 0, 0, 0, 0]
        R_VAE = [0, 0, 0, 0, 0]
        N_VAE = [0, 0, 0, 0, 0]

        for it, (s, a, r, ns) in enumerate(self.valid_loader):
            s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
            with torch.no_grad():
                sampled_actions = self.vae.decode(s)
                perturbed_actions = self.actor(s, sampled_actions)
                td_error = (
                    self.critic.q1(s, a)
                    - self.discount
                    * self.critic.q1(ns, perturbed_actions)
                    - r
                ).abs().mean()


                for i, epsilon in enumerate(self.epsilons): 
                    R[i] += (r[torch.abs(perturbed_actions - a) < epsilon]).sum()
                    N[i] += (r[torch.abs(perturbed_actions - a) < epsilon]).numel()
                    R_VAE[i] += (r[torch.abs(sampled_actions - a) < epsilon]).sum()
                    N_VAE[i] += (r[torch.abs(sampled_actions - a) < epsilon]).numel()

                R[-1] =  (r[torch.abs(perturbed_actions - a) > self.epsilons[-1]]).sum()
                N[-1] =  (r[torch.abs(perturbed_actions - a) > self.epsilons[-1]]).numel()
                R_VAE[-1] += (r[torch.abs(sampled_actions - a) > self.epsilons[-1]]).sum()
                N_VAE[-1] += (r[torch.abs(sampled_actions - a) > self.epsilons[-1]]).numel()
            
            weight_large = torch.exp(-(a - perturbed_actions).pow(2)/0.05)
            weight_medium = torch.exp(-(a - perturbed_actions).pow(2)/0.03)
            weight_small = torch.exp(-(a - perturbed_actions).pow(2)/0.01)
            
            weight_large /= weight_large.sum()
            weight_medium /= weight_medium.sum()
            weight_small /= weight_small.sum()
            
            
            data = np.concatenate(
                (a.cpu().numpy(), \
                 sampled_actions.cpu().numpy(), \
                 (perturbed_actions - sampled_actions).cpu().numpy())
               ,axis=1)
            """
            table = wandb.Table(
                data=data.tolist()[:50],
                columns=[
                    "Actual Price",
                    "Sampled Price",
                    "Price"
                ]
            )
            """
            
            deviation_from_positive_phi = np.abs((self.phi - 
            (perturbed_actions - sampled_actions).cpu().numpy())).mean()
            
            deviation_from_negative_phi = np.abs((-self.phi -
            (perturbed_actions - sampled_actions).cpu().numpy())).mean()
        
            deviation_from_zero = np.abs((
            (perturbed_actions - sampled_actions).cpu().numpy())).mean()

            signs = np.abs(
            ((perturbed_actions - sampled_actions).cpu().numpy()>0).mean() - 0.5)
            
            std = np.std((perturbed_actions - sampled_actions).cpu().numpy())

        return {
            "Temporal Difference Error" : td_error,
            "Deviation From Positive Phi" : deviation_from_positive_phi,
            "Deviation From Negative Phi" : deviation_from_negative_phi,
            "Deviation From Zero" : deviation_from_zero,
            "Signs":signs,
            "Standard Deviation" : std,
            "Reward_1": R[0] / N[0],
            "Reward_2": R[1] / N[1],
            "Reward_3": R[2] / N[2],
            "Reward_4": R[3] / N[3],
            "Reward_OUT":R[4]/N[4],
            "N_1":N[0] / len(self.valid_loader.dataset),
            "N_2":N[1]/ len(self.valid_loader.dataset),
            "N_3":N[2]/ len(self.valid_loader.dataset),
            "N_4":N[3]/ len(self.valid_loader.dataset),
            "N_4_out":N[4]/ len(self.valid_loader.dataset),
            "Reward_1_VAE": R[0] / N[0] - R_VAE[0] / N_VAE[0],
            "Reward_2_VAE": R[1] / N[1] - R_VAE[1] / N_VAE[1],
            "Reward_3_VAE": R[2] / N[2] - R_VAE[2] / N_VAE[2],
            "Reward_4_VAE": R[3] / N[3] - R_VAE[3] / N_VAE[3], 
            "Reward_OUT_VAE": R[4] / N[4] - R_VAE[4] / N_VAE[4],
            "Weighted Sum (Large)":(weight_large*r).cpu().numpy().sum(),
            "Weighted Sum (Medium)":(weight_medium*r).cpu().numpy().sum(),
            "Weighted Sum (Small)":(weight_small*r).cpu().numpy().sum(),
            #"Table":table
        }#,pd.DataFrame(data, columns = ["Actual Price", "Sampled Price", "Price"])

    def pretrain_vae(self):
        print('Pretraining VAE')

        best_performance = 1e6
        early_stopping = 200
        counter = 0
        epoch = 0

        while 1:
            epoch += 1
            for it, (s, a, r, ns) in enumerate(self.train_loader):
                s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
                
                recon, mean, std = self.vae(s, a)
                recon_loss = F.mse_loss(recon, a)
                KL_loss = (
                    -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                )
                vae_loss = recon_loss + 0.5 * KL_loss

                self.vae_optimizer.zero_grad()
                vae_loss.backward()
                self.vae_optimizer.step()

            with torch.no_grad():
                running_vae_loss = 0
                for it, (s, a, r, ns) in enumerate(self.valid_loader):
                    s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
                    recon, mean, std = self.vae(s, a)
                    recon_loss = F.mse_loss(recon, a)
                    KL_loss = (
                        -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                    )
                    vae_loss = recon_loss + 0.5 * KL_loss

                    running_vae_loss += vae_loss.item()
                running_vae_loss /= len(self.valid_loader)

                #wandb.log({"VAE Loss" : running_vae_loss})

            if running_vae_loss < best_performance:
                best_performance = running_vae_loss
                self.save_vae()
                counter = 0

            else:
                counter +=1

            if counter == early_stopping:
                break

        print(f"Pretraining Completed! Best Model at Epoch : {epoch}, Loss : {best_performance}")
        self.load_vae()
        return best_performance

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

    def name_vae(self):
        hidden_config = "_".join([str(i) for i in self.vae_hidden_dim])

        self.vae_name = "L_"+str(self.vae_latent_dim)+"_H_"+\
                        hidden_config+"_D_"+str(self.vae_decay)+\
                        "_LR_"+str(self.vae_lr) + "_BS_" + str(1024)

    def save_vae(self):
        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        if not os.path.exists('./weights/vae'):
            os.mkdir('./weights/vae')

        torch.save(self.vae.state_dict(), f'./weights/vae/{self.vae_name}.pth')

    def load_vae(self):
        self.vae.load_state_dict(torch.load(f'./weights/vae/{self.vae_name}.pth'))                      
        
    def save_progress(self):
        torch.save(self.agent.state_dict(), )
        
    def name_agent(self):
        import random
        import string

        def randomString(stringLength=8):
            letters = string.ascii_lowercase
            return ''.join(random.choice(letters) for i in range(stringLength))
        
        self.file_name = randomString()
    
    def to_data(self):
        for it, (s, a, r, ns) in enumerate(self.valid_loader):
            s, a, r, ns = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device)
            with torch.no_grad():
                sampled_actions = self.vae.decode(s)
                perturbed_actions = self.actor(s, sampled_actions)

            data = np.concatenate(
                (a.cpu().numpy(), \
                 sampled_actions.cpu().numpy(), \
                 (perturbed_actions - sampled_actions).cpu().numpy())
               ,axis=1)
            
        pd.DataFrame(data, columns = ["Actual Rate", "Sampled Rate", "Rate"]).to_csv(self.file_name,index=False)
        
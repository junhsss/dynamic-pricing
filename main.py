import argparse
import os
import wandb

import numpy as np
import pandas as pd
from itertools import repeat
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from src.agents.agent_bcq import BCQ
from src.agents.agent_td3 import TD3
import random
import subprocess

def parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.003161, type=float)
    parser.add_argument("--batch_size", default=1765, type=int)
    parser.add_argument("--epsilons", default=[0.005, 0.01, 0.015, 0.02], nargs="+", type=float)
    parser.add_argument("--device", default='cuda', type=str)

    # Agent Arguments
    parser.add_argument("--state_dim", default=60, type=int)

    parser.add_argument("--actor_hidden_dim", default=204, type=int)
    parser.add_argument("--actor_depth", default=1, type=int)
    parser.add_argument("--actor_factor", default=0.7452, type=float)
    parser.add_argument("--actor_lr", default=0.00005414, type=float)
    parser.add_argument("--actor_decay", default=0.00000784, type=float)

    parser.add_argument("--critic_hidden_dim", default=918, type=int)
    parser.add_argument("--critic_depth", default=5, type=int)
    parser.add_argument("--critic_factor", default=0.7679, type=float)
    parser.add_argument("--critic_lr", default=0.0002988, type=float)
    parser.add_argument("--critic_decay", default=0.01915, type=float)

    parser.add_argument("--vae_hidden_dim_1", default=432, type=int)
    parser.add_argument("--vae_hidden_dim_2", default=366,  type=int)
    parser.add_argument("--vae_latent_dim", default=29, type=int)
    parser.add_argument("--vae_lr", default=0.00002445, type=float)
    parser.add_argument("--vae_decay", default=0.00002579, type=float)

    parser.add_argument("--reward_shaping", action="store_true")
    parser.add_argument("--clip_grad_norm", default=1e5, type=float)

    #BCQ Arguments
    parser.add_argument("--phi", default=0.01224, type=float)
    parser.add_argument("--lamb", default=0.0006878, type=float)

    #TD3 Arguments
    parser.add_argument("--policy_delay", default=8, type=float)
    parser.add_argument("--noise_std", default=0.002383, type=float)
    parser.add_argument("--noise_clip", default=0.0006707, type=float)
    
    args = parser.parse_args()
    wandb.config.update(args)
    return args


def get_gpu_memory_map():

    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ])
    result = result.decode('utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

get_gpu_memory_map()


if not os.path.exists("./data/result/"):
    os.mkdir("./data/result/")
    
if __name__ == "__main__":
    wandb.init()
    args = parser()
    
    agent = BCQ(
        state_dim=args.state_dim,
        actor_hidden_dim=[int(float(dim)*(args.actor_factor**i)) 
                    for i,dim in enumerate([args.actor_hidden_dim] * args.actor_depth)],
        critic_hidden_dim=[int(float(dim)*(args.critic_factor**i)) 
                    for i,dim in enumerate([args.critic_hidden_dim] * args.critic_depth)],
        vae_hidden_dim=[args.vae_hidden_dim_1, args.vae_hidden_dim_2],
        vae_latent_dim=args.vae_latent_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        vae_lr=args.vae_lr,
        actor_decay=args.actor_decay,
        critic_decay=args.critic_decay,
        vae_decay=args.vae_decay,
        discount=args.discount,
        tau=args.tau,
        phi=args.phi,
        epsilons=args.epsilons,
        lamb=args.lamb,
        policy_delay=args.policy_delay,
        noise_std=args.noise_std,
        noise_clip=args.noise_clip,
        batch_size=args.batch_size,
        reward_shaping=args.reward_shaping,
        clip_grad_norm=args.clip_grad_norm,
        device='cuda:'+str(np.argmin(list(get_gpu_memory_map().values()))))

    if os.path.exists(f"./weights/vae/{agent.vae_name}.pth"):
        agent.load_vae()
    else:
        wandb.config.vae_best_performance = agent.pretrain_vae()

    early_stopping=10000
    counter = 0
    best_performance = -1e6
    for i in range(10000):
        metric = agent.step()
        wandb.log(metric)
        if metric["Weighted Sum (Medium)"] > best_performance:
            best_performance = metric["Weighted Sum (Medium)"]
            if metric["Signs"] < 0.5:
                if metric["N_4"] > 0.6:
                    counter = 0
                else:
                    counter += 1
            else:
                counter += 1
        else:
            counter += 1
   
        if counter == early_stopping:
            #data.to_csv("./data/result/"+str(counter)+".csv",index=False)
            break
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
    
    parser.add_argument("--mode", default='bcq',type=str)
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--epsilons", default=[0.005, 0.01, 0.015, 0.02], nargs="+", type=float)
    parser.add_argument("--device", default='cuda', type=str)

    # Agent Arguments
    parser.add_argument("--state_dim", default=49, type=int)
    parser.add_argument("--actor_hidden_dim_1", default=500, type=int)
    parser.add_argument("--actor_hidden_dim_2", default=500, type=int)
    parser.add_argument("--critic_hidden_dim_1", default=500, type=int)
    parser.add_argument("--critic_hidden_dim_2", default=500, type=int)
    parser.add_argument("--vae_hidden_dim_1", default=500, type=int)
    parser.add_argument("--vae_hidden_dim_2", default=500,  type=int)
    parser.add_argument("--actor_lr", default=0.0001, type=float)
    parser.add_argument("--critic_lr", default=0.0005, type=float)
    parser.add_argument("--actor_decay", default=0.001, type=float)
    parser.add_argument("--critic_decay", default=0.001, type=float)
    parser.add_argument("--reward_shaping", action="store_true")
    parser.add_argument("--clip_grad_norm", default=1e5, type=float)

    #BCQ Arguments
    parser.add_argument("--vae_latent_dim", default=20, type=int)
    parser.add_argument("--vae_lr", default=0.0001, type=float)
    parser.add_argument("--vae_decay", default=0.0001, type=float)
    parser.add_argument("--phi", default=0.04, type=float)
    parser.add_argument("--lamb", default=0.75, type=float)

    
    #TD3 Arguments
    parser.add_argument("--policy_delay", default=4, type=float)
    parser.add_argument("--noise_std", default=0.01, type=float)
    parser.add_argument("--noise_clip", default=0.005, type=float)
    
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
    
    if args.mode == "bcq":
        agent = BCQ(
            state_dim=args.state_dim,
            actor_hidden_dim=[args.actor_hidden_dim_1,args.actor_hidden_dim_2],
            critic_hidden_dim=[args.critic_hidden_dim_1,args.critic_hidden_dim_2],
            vae_hidden_dim=[args.vae_hidden_dim_1,args.vae_hidden_dim_2],
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


    if args.mode == "td3":
        agent = TD3(
            state_dim=args.state_dim,
            actor_hidden_dim=args.actor_hidden_dim,
            critic_hidden_dim=args.critic_hidden_dim,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            actor_decay=args.actor_decay,
            critic_decay=args.critic_decay,
            discount=args.discount,
            tau=args.tau,
            epsilons=args.epsilons,
            policy_delay=args.policy_delay,
            noise_std=args.noise_std,
            noise_clip=args.noise_clip,
            batch_size=args.batch_size,
            device='cuda:'+str(np.argmin(list(get_gpu_memory_map().values()))))

    if os.path.exists(f"./weights/vae/{agent.vae_name}.pth"):
        agent.load_vae()
    else:
        wandb.config.vae_best_performance = agent.pretrain_vae()

    best_performance = 1e6
    early_stopping=1000
    counter = 0
    while 1:
        metric, data = agent.step()
        wandb.log(metric)
        
        if metric["Reward_3"] < best_performance:
            best_performance = metric["Reward_3"]
            counter =0
        else:
            counter += 1
            
        if counter == early_stopping:
            data.to_csv("./data/result/"+str(counter)+".csv",index=False)
            break
        
        
        
        
        
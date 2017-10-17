import os
import sys
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from envs import create_atari_env
from model import A3C
from train import train_model

if __name__ == '__main__':

    seed = 1
    game = 'Seaquest-v0'
    num_agents = 4
    threads = []
    torch.manual_seed(seed)
    env = create_atari_env(game)
    print env.observation_space.shape
    share = A3C(env.observation_space.shape[0], env.action_space)
    share.share_memory()
    print "Beginning training of A3C on {}".format(game)
    for a in range(0, num_agents):
        th = mp.Process(target=train_model, args=(a, share))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()

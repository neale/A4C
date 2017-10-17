import math
import os
import sys

import torch
from  torch.nn.functional import softmax, log_softmax
import torch.optim as optim
from envs import create_atari_env
from model import A3C
from torch.autograd import Variable
from torchvision import datasets, transforms


def update_grad(m1, m2):

    for p, s in zip(m1.parameters(), m2.parameters()):

        if s.grad is not None:
            return
        s.grad = p.grad

def train_model(trip, m2):
    game = "Seaquest-v0"
    env = create_atari_env(game)
    torch.manual_seed(1 + trip)
    env.seed(1 + trip)
    model = A3C(env.observation_space.shape[0], env.action_space)
    loss = optim.Adam(m2.parameters(), lr=.0001)
    # loading model
    print "loading model"
    #model = torch.load('/home/neale/repos/pyA3C/models/trained/Breakout-v0ckpt.pymodel')
    model.train()
    episode_len = 0
    val , proba = [], []
    state = env.reset()
    state = torch.from_numpy(state)
    terminal = True
    while True:
        episode_len += 1
        env.render()
        model.load_state_dict(m2.state_dict())
        r, vals, entropy, probs = [], [], [], []
        if terminal:
            context = Variable(torch.zeros(1, 256))
            hidden = Variable(torch.zeros(1, 256))
        else:
            context = Variable(context.data)
            hidden = Variable(hidden.data)

        for frame in range(20):
            val, linear_out, (hidden, context) = model((Variable(state.unsqueeze(0)), (hidden, context)))
            proba = softmax(linear_out)
            log_proba = log_softmax(linear_out)
            e = -(log_proba * proba).sum(1)
            entropy.append(e)

            a = proba.multinomial().data
            log_proba = log_proba.gather(1, Variable(a))
            state, reward, terminal, _ = env.step(a.numpy())
            if (terminal == True) or (episode_len >= 1000000):
                terminal = True
            reward = min(reward, 1)
            reward = max(r, -1)
            if terminal:
                episode_len = 0
                state = env.reset()
                torch.save(model, './models/trained/'+game+'_ckpt.pymodel')
            state = torch.from_numpy(state)
            probs.append(log_proba)
            vals.append(val)
            r.append(reward)
            if terminal:
                break
        total_reward = torch.zeros(1, 1)
        if not terminal:
            value, _, _ = model((Variable(state.unsqueeze(0)), (hidden, context)))
            total_reward = value.data
        vals.append(Variable(total_reward))
        actor_loss, critic_loss = 0, 0
        toal_reward = Variable(total_reward)
        advantage_update = torch.zeros(1, 1)
        for i in reversed(range(len(r))):
            total_reward = 0.99 * total_reward + r[i]
            advantage = total_reward - vals[i]
            critic_loss = critic_loss + 0.5 * advantage.pow(2)
            update = r[i] + 0.99 * vals[i + 1].data - vals[i].data
            advantage_update = advantage_update * 0.99 * 1. + update

            actor_loss = actor_loss - probs[i] * Variable(advantage_update) - 0.01 * entropy[i]

        loss.zero_grad()
        (actor_loss + 0.5*critic_loss).backward()
        update_grad(model, m2)
        loss.step()
        print "Advantage update: {}".format(advantage_update)
        print "actor loss: {}".format(actor_loss)
        print "critic loss: {}".format(critic_loss)
        print "reward: ".format(reward)

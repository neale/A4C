import math
import os
import sys
from scipy.misc import imshow
import torch
import torch.nn.functional as F
import torch.optim as optim
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from collections import deque


def test(rank, shared_model):

    game = "Seaquest-v0"
    torch.manual_seed(1 + rank)

    env = create_atari_env(game)
    env.seed(1 + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    #model = torch.load('/home/neale/repos/pyA3C/models/trained/Seaquest-v0ckpt.pymodel')
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256), volatile=True)
            hx = Variable(torch.zeros(1, 256), volatile=True)
        else:
            cx = Variable(cx.data, volatile=True)
            hx = Variable(hx.data, volatile=True)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.numpy()


        state, reward, done, _ = env.step(action[0, 0])

        print state.shape
        imshow(state[0])

        done = done or episode_length >= 1000000
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, score {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            torch.save(model, './models/trained/'+game+'_ckpt.pymodel')

            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)

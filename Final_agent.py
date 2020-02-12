import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time
import warnings
import gym
import matplotlib.pyplot as plt
from wrappers import *
from memory import ReplayMemory,N_step_ReplayMemory,PER
from models import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transion', 
                        ('state', 'action', 'reward', 'next_state','done'))

class Final_agent:
    def __init__(self,env,policy,target,n_step=3,n_action=4,capacity=100000,batch_size=32,lr=2.5e-4,gamma=0.99,burn_in=50000,C=1000,eps_decay=1000000,alpha=0.5):
        self.env=env
        self.n_action=n_action
        self.memory=PER(capacity,n_step,gamma,alpha)
        self.device="cuda"
        self.policy=policy
        self.target=target
        self.batch_size=batch_size
        self.gamma=gamma
        self.lr=lr
        self.opt= optim.Adam(self.policy.parameters(), lr=self.lr)
        self.burn_in=burn_in
        self.C=C
        self.eps_decay=eps_decay
        self.n_step=n_step
        self.beta=0.4
        self.loss=nn.MSELoss(reduction="none")
        self.eps_priority=1e-6
        self.max_step=1e7
    def get_state(self,obs):
        state=torch.FloatTensor(np.array(obs).transpose(2,0,1)).unsqueeze(0)
        return(state)
    def get_action(self,state,eps):
        x=random.random()
        if x<eps:
            return(torch.tensor([[random.randrange(self.n_action)]], dtype=torch.long))
        else:
            with torch.no_grad():
                return(self.policy(state.to("cuda")).max(1)[1].view(1,1))
        
    def update_policy(self):
        state,action,reward,next_state,done,indexes,IS_weights=self.memory.sample(self.batch_size,self.beta)
        state=state.to("cuda")
        action=action.to("cuda")
        next_state=next_state.to("cuda")
        reward=reward.to("cuda")
        done=done.to("cuda")
        IS_weights = torch.FloatTensor(IS_weights).to("cuda")
        q=self.policy(state).gather(1,action.unsqueeze(1)).squeeze(1)
        q_max=self.target(next_state).gather(1,self.policy(next_state).max(1)[1].unsqueeze(1)).squeeze(1)
        y=(reward+(self.gamma**self.n_step)*q_max)*(1-done)+reward*done
        element_loss=self.loss(q,y)
        loss=torch.mean(IS_weights*element_loss)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.memory.update_priority(indexes,(element_loss**0.5).cpu().detach().numpy()+self.eps_priority)
        return
    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())
    def train(self,episodes):
        steps=0
        reward_list=[]
        for episode in range(episodes):
            obs=self.env.reset()
            state=self.get_state(obs)
            reward_episode=0
            done=False
            while not done:
                self.beta=0.4+min(steps/self.max_step,1)*(1-0.4)
                steps+=1
                test_eps=int(steps>self.eps_decay)
                eps=(1-steps*(1-0.1)/self.eps_decay)*(1-test_eps)+0.1*test_eps
                action=self.get_action(state,eps)
                obs,reward,done,info=env.step(action)
                reward_episode+=reward
                next_state=self.get_state(obs)
                reward = torch.tensor([reward], device="cpu", dtype=torch.float)
                action = torch.tensor([action], device="cpu", dtype=torch.long)
                done = torch.tensor([int(done)], device="cpu", dtype=int)
                self.memory.push(state,action,reward,next_state,done)
                if steps>self.burn_in:
                    self.update_policy()
                if steps>self.burn_in and steps%self.C==0:
                    self.update_target()
                state=next_state
            if episode>0 and episode % 50 == 0:
                print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps, episode, episodes, np.mean(reward_list[-50:])))
            reward_list.append(reward_episode)
        print(reward_list)
        plt.plot(list(range(episodes)),reward_list)
        plt.xlabel("episodes")
        plt.ylabel("reward")
        plt.savefig('training_reward_dqn2.png')
        self.env.close()
        return
    def save_model(self,name):
        torch.save(self.policy,name)
        return
    def load_model(self,name):
        self.policy=torch.load(name)
    def test(self,n_episodes):
        test_reward=[]
        for episode in range(n_episodes):
            obs = env.reset()
            state = self.get_state(obs)
            reward_episode = 0.0
            done=False
            while not done:
                with torch.no_grad():
                    action=self.policy(state.to("cuda")).max(1)[1].view(1,1)
                obs,reward,done,info=env.step(action)
                reward_episode+=reward
                state=self.get_state(obs)
                if done:
                    print("Finished Episode {} with reward {}".format(episode, reward_episode))
            env.close()
            test_reward.append(reward_episode)
        return (test_reward)

def test_video(agent,env, n_episodes, render=False):
    env = gym.wrappers.Monitor(env, './videos/')
    for episode in range(n_episodes):
        obs = env.reset()
        state = agent.get_state(obs)
        total_reward = 0.0
        for t in count():
            action = agent.policy(state.to('cuda')).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            obs, reward, done, info = env.step(action)

            total_reward += reward

            if not done:
                next_state = agent.get_state(obs)
            else:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {} with reward {}".format(episode, total_reward))
                break

    env.close()
    return
if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    policy = DQN(n_actions=4).to("cuda")
    target = DQN(n_actions=4).to("cuda")
    target.load_state_dict(policy.state_dict())
    #env = gym.make("SeaquestNoFrameskip-v4")
    env=gym.make("PongNoFrameskip-v4")
    env = make_env(env)
    agent=Final_agent(env,policy,target)
    try:
        agent.train(800)
    except KeyboardInterrupt:
        agent.save_model("final_model_pong")
    agent.save_model("final_model_pong")
    l=agent.test(100)
    test_video(agent,env, 1)
    print(np.mean(l))
                      
                      



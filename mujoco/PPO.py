import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal

from config import *

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5) 

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)

    def forward(self, s):
        x = torch.tanh(self.fc1(s))
        x = torch.tanh(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std.expand_as(mean)

        return mean, log_std

    def get_action(self, state):
        '''
        根据 state 采样动作，放到 cpu 上转成 numpy
        '''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample().cpu().numpy()

        return action

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        v = torch.tanh(self.fc1(state))
        v = torch.tanh(self.fc2(v))
        value = self.value(v)

        return value

class Normalize:
    '''
    用于归一化
    '''
    def __init__(self, state_dim):
        self.mean = np.zeros((state_dim,))
        self.std = np.ones((state_dim,))
        self.stdd = np.zeros((state_dim,))
        self.n = 0
        self.training = True

    def __call__(self, obs):
        obs = np.array(obs, copy=False)

        if self.training:
            self.n += 1
            if self.n == 1:
                self.mean = np.copy(obs)
            else:
                diff = obs - self.mean
                self.mean += diff / self.n
                self.stdd += diff * (obs - self.mean)
            
            if self.n > 1:
                self.std = np.sqrt(self.stdd / (self.n - 1) + 1e-10)

        norm_obs = (obs - self.mean) / (self.std + 1e-8)
        
        return np.clip(norm_obs, -5.0, 5.0)

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, ent_coef=0.01, device=None):
        self.device = device if device is not None else 'cpu'
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, hidden_dim).to(self.device)

        self.ent_coef = ent_coef

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=a_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=c_lr, weight_decay=c_l2)

    def train(self, memory, next_value, epochs=10):
        states = torch.tensor(np.array([m[0] for m in memory]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([m[1] for m in memory]), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array([m[2] for m in memory]), dtype=torch.float32).to(self.device)
        masks = torch.tensor(np.array([m[3] for m in memory]), dtype=torch.float32).to(self.device)

        values = self.critic(states)
        returns, advantages = self.compute_gae(rewards, masks, values, next_value)
        
        with torch.no_grad():
            old_mu, old_log_std = self.actor(states)
            old_std = old_log_std.exp()
            dist = Normal(old_mu, old_std)
            old_log_prob = dist.log_prob(actions).sum(-1, keepdim=True)

        T = len(states)
        for _ in range(epochs):
            arr = np.arange(T)
            np.random.shuffle(arr)

            for i in range(T // batch_size):
                batch_index = arr[batch_size * i: batch_size * (i + 1)]
                batch_states = states[batch_index]
                batch_advantages = advantages[batch_index].unsqueeze(1)
                batch_actions = actions[batch_index]
                batch_returns = returns[batch_index].unsqueeze(1)

                mean, log_std = self.actor(batch_states)
                std = log_std.exp()
                dist = Normal(mean, std)
                new_prob = dist.log_prob(batch_actions).sum(-1, keepdim=True)
                old_prob = old_log_prob[batch_index].detach()
                entropy = dist.entropy().sum(-1, keepdim=True)

                ratio = torch.exp(new_prob - old_prob)

                surr_loss1 = ratio * batch_advantages
                surr_loss2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages
                actor_loss = -torch.min(surr_loss1, surr_loss2).mean() - self.ent_coef * entropy.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()

                # 裁剪梯度
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                # 更新Critic
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values, batch_returns)
                self.critic_optim.zero_grad()
                critic_loss.backward()

                # 裁剪梯度
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

    def compute_gae(self, rewards, masks, values, next_value):
        '''
        计算 GAE
        '''
        T = len(rewards)
        advants = torch.zeros(T, device=self.device)
        
        gae = 0
        previous_value = next_value 
        
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * previous_value * masks[t] - values[t]
            gae = delta + gamma * lam * masks[t] * gae
            
            advants[t] = gae
            previous_value = values[t]
        
        returns = advants + values.flatten()
        advants = (advants - advants.mean()) / (advants.std() + 1e-8)
        return returns.detach(), advants.detach()
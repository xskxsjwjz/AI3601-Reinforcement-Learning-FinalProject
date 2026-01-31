import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

def _process_frame(frame):
    """处理 frame，支持 LazyFrame 和 numpy.ndarray"""
    if hasattr(frame, 'force'):
        return frame.force()
    else:
        return np.array(frame)

class DQN(nn.Module):
    '''
    标准 DQN 的结构类。
    '''
    def __init__(self, input_shape, d_output):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.d_output = d_output

        # 三个卷积层，输出卷积特征。
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 全连接层，直接输出 Q(s, a)
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, d_output)
        )

    # 卷积 -> 展平为 [batch_size, c*h*w] -> 全连接层 -> 输出 Q(s, a)
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        q_values = self.fc(x)
        return q_values

    def feature_size(self):
        '''
        制造全 0 向量通过卷积层测试 size。
        '''
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
class memory_buffer():
    '''
    经验回放池。
    '''
    def __init__(self, capacity, input_shape, device):
        '''
        :param capacity: 回放池的最大容量。
        :param input_shape: 输入状态的形状，atari 中为 [4, 84, 84]。
        :param device: cpu 或者 cuda。
        '''
        self.buffer = []
        self.capacity = capacity
        self.device = device
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        '''
        将五元组存入池子中。直接存储 LazyFrame 对象以节省内存。
        '''
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        '''
        一次性从缓存中采样 batch_size 个样本并移动到 device 上。
        
        :param batch_size: 一次性采样的样本数目
        '''
        # 随机采样索引
        idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:
            frame, action, reward, next_frame, done = self.buffer[idx]
            # 在采样时才转换 LazyFrame 为 tensor
            states.append(torch.from_numpy(_process_frame(frame).transpose(2, 0, 1)[None]).float())
            actions.append(action)
            rewards.append(reward)
            next_states.append(torch.from_numpy(_process_frame(next_frame).transpose(2, 0, 1)[None]).float())
            dones.append(done)
        
        # 拼接为 batch tensor，转到GPU后再归一化
        states = torch.cat(states).to(self.device).div(255)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        next_states = torch.cat(next_states).to(self.device).div(255)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)
    
class DQNAgent():
    '''
    结合 DQN 网络和经验回放池的智能体。
    '''
    def __init__(self, input_shape, action_space=[], capacity=10000, epsilon=1.0, lr=1e-4):
        self.action_space = action_space
        self.epsilon = epsilon
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.memory_buffer = memory_buffer(capacity, input_shape, self.device)
        self.policy = DQN(input_shape, action_space.n).to(self.device)
        self.target = DQN(input_shape, action_space.n).to(self.device)

        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def select_action(self, state, epsilon=None):
        '''
        带有 epsilon-greedy 的选择动作方法。
        
        :param epsilon: 探索概率。
        '''
        if epsilon is None:
            epsilon = self.epsilon

        if random.random() < epsilon:
            action = random.randrange(self.action_space.n)
        else:
            with torch.no_grad():
                q_values = self.policy(state)
                action = q_values.argmax(dim=1).item()
        return action
    
    def compute_TDLoss(self, states, actions, rewards, next_states, dones, gamma=0.99):
        '''
        计算 TD Loss。

        :param states: [batch_size, 4, 84, 84]
        '''
        # 取出当前状态下对应动作的 Q value
        q_values = self.policy(states)
        action_q_values = q_values[range(states.shape[0]), actions]

        # 假设下一 state 选择 Q value 最大的 action
        next_q_values = self.target(next_states)
        next_state_values = next_q_values.max(dim=-1)[0]

        # 如果 done 了就直接取 rewards
        td_targets = torch.where(dones, rewards, rewards + gamma * next_state_values)
        loss = F.smooth_l1_loss(action_q_values, td_targets.detach())

        return loss
    
    def learn(self, batch_size):
        '''
        从缓存中采样，计算 `TD_loss` 并限制梯度范围，更新参数
        '''
        if len(self.memory_buffer) <= batch_size:
            return 0
        states, actions, rewards, next_states, dones = self.memory_buffer.sample(batch_size)
        loss = self.compute_TDLoss(states, actions, rewards, next_states, dones)
        
        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()
        return loss.item()
    
    def observe(self, frame):
        '''
        给定一个Lazyframe或numpy数组，返回拼接的画面。
        
        :param frame: 给定的 Lazyframe 或 numpy.ndarray
        '''
        return torch.from_numpy(_process_frame(frame).transpose(2, 0, 1)[None]).float().to(self.device).div(255)
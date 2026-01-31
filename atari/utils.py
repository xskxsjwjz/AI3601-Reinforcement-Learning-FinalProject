import gymnasium as gym
import numpy as np
import cv2
from collections import deque
import ale_py

# 注册 ALE 环境
gym.register_envs(ale_py)

# 需要 FIRE 来开始游戏的环境列表
FIRE_RESET_GAMES = ['Breakout']

class Random_Reset(gym.Wrapper):
    '''
    每次 reset 后，挂机一段时间来采样初始化状态。
    '''
    def __init__(self, env, afk_max=20):
        super().__init__(env)
        self.afk_max = afk_max

    def reset(self, **kwargs):
        '''
        reset 以后先挂机随机回合，保证随机初始化
        '''
        obs, info = self.env.reset(**kwargs)
        afk_turns = self.unwrapped.np_random.integers(1, self.afk_max + 1)

        for _ in range(afk_turns):
            obs, _, terminated, truncated, info = self.env.step(0)
            done = terminated or truncated
            if done:
                obs, info = self.env.reset(**kwargs)

        return obs, info
    
class Fire_Reset(gym.Wrapper):
    '''
    自动开火以开始游戏（仅适用于需要按 FIRE 开始的游戏）
    '''
    def reset(self, **kwargs):
        '''
        开火开始游戏
        '''
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        done = terminated or truncated
        if done:
            obs, info = self.env.reset(**kwargs)

        return obs, info
    
class One_Life_Reset(gym.Wrapper):
    '''
    将每一条命分开当作一局游戏，死了直接设置 done.
    仅适用于有生命系统的游戏。
    '''
    def __init__(self, env):
        super().__init__(env)
        self.total_lives = self.env.unwrapped.ale.lives()
        self.done = True  # 初始状态设为 True，确保第一次 reset 正常
        # 检查是否需要 FIRE 来开始/继续游戏
        game_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else ''
        self.needs_fire = any(g in game_name for g in FIRE_RESET_GAMES) and 'FIRE' in env.unwrapped.get_action_meanings()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.done = done
        rest_lives = self.env.unwrapped.ale.lives()

        if rest_lives < self.total_lives and rest_lives > 0:
            terminated = True
        self.total_lives = rest_lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        '''
        如果游戏没有真正结束，则执行必要动作继续下一条命。
        对于 Breakout 等游戏，需要按 FIRE 发球。
        '''
        if self.done:
            obs, info = self.env.reset(**kwargs)
        else:
            # 先执行 NOOP 跳过死亡动画
            obs, _, _, _, info = self.env.step(0)
            # 如果游戏需要 FIRE 才能继续（如 Breakout 发球），执行 FIRE
            if self.needs_fire:
                obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
                if terminated or truncated:
                    obs, info = self.env.reset(**kwargs)
        self.total_lives = self.env.unwrapped.ale.lives()

        return obs, info
    
class Life_Loss(gym.Wrapper):
    def __init__(self, env, train=True, fire_reset=False):
        super().__init__(env)
        self.life_loss = -1 if train else 0
        self.total_lives = self.env.unwrapped.ale.lives()
        self.fire_reset = fire_reset

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rest_lives = self.env.unwrapped.ale.lives()

        if rest_lives < self.total_lives and rest_lives > 0:
            reward += self.life_loss
            obs, _, _, _, _ = self.env.step(0)
            if self.fire_reset:
                obs, _, _, _, _ = self.env.step(1)
        
        self.total_lives = rest_lives
        return obs, reward, terminated, truncated, info

class Clip_Reward(gym.RewardWrapper):
    '''
    统一 reward = -1, 0, +1
    '''
    def reward(self, reward):
        return np.sign(reward)
    
class Frame_Preprocess(gym.ObservationWrapper):
    '''
    RGB -> 灰度图 -> 转置
    '''
    def __init__(self, env, w=84, h=84, key=None):
        '''
        __init__ 的 Docstring
        
        :param w: 宽
        :param h: 高
        :param key: 如果环境返回的是一个字典，这个 key 应该从其中取出图像。
        '''
        super().__init__(env)
        self.w = w
        self.h = h
        self.key = key

        space = gym.spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

        if self.key == None:
            self.observation_space = space

        else:
            self.observation_space.spaces[self.key] = space

    def observation(self, observation):
        frame = observation if self.key is None else observation[self.key]
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (self.w, self.h), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, np.newaxis]

        if self.key is None:
            return frame
        
        observation = observation.copy()
        observation[self.key] = frame

        return observation
    
class Frame_Stack(gym.Wrapper):
    '''
    将多个图像拼在一起。
    '''
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], k)
        frame_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (frame_shape[: -1] + (frame_shape[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(observation)

        return Lazy_Frames(list(self.frames)), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return Lazy_Frames(list(self.frames)), reward, terminated, truncated, info
    
class Lazy_Frames():
    def __init__(self, frames):
        self.frames = frames
        self.out = None

    def force(self):
        if self.out is None:
            self.out = np.concatenate(self.frames, axis=-1)
            self.frames = None
        return self.out
    
    def __array__(self, dtype=None):
        out = self.force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self.force())
    
    def __getitem__(self, i):
        return self.force()[i]
    
    def count(self):
        frames = self.force()
        return frames.shape[frames.ndim - 1]
    
    def frame(self, i):
        return self.force()[i]
    
def create_env(env_name):
    '''
    创建并包装基础环境
    
    :param env_name: 环境名称
    '''
    env = gym.make(env_name)
    env = Random_Reset(env, 20)

    return env

def construct_wrappers(env, one_life_reset=False, life_loss=False, clip_reward=True, frame_stack=True, train=True):
    '''
    Wrappers 由外到里逐步执行，注意包装顺序。
    
    注意：one_life_reset 和 life_loss 不应同时启用！
    - one_life_reset=True: 每条命当作一局游戏（训练时推荐）
    - life_loss=True: 掉血给惩罚但不结束游戏（测试时可用）
    
    :param env: 基础环境
    :param one_life_reset: 是否启用单命重置（仅对有生命系统的游戏有效）
    :param life_loss: 是否启用掉血惩罚（仅对有生命系统的游戏有效）
    :param clip_reward: 是否裁剪奖励到 [-1, 0, 1]
    :param frame_stack: 是否堆叠帧
    :param train: 是否为训练模式（影响life_loss的惩罚值）
    '''
    # 检查是否有生命系统
    has_lives = env.unwrapped.ale.lives() > 0
    
    # 检查是否需要 FIRE 来开始游戏
    game_name = env.unwrapped.spec.id if hasattr(env.unwrapped, 'spec') else ''
    needs_fire = any(g in game_name for g in FIRE_RESET_GAMES) and 'FIRE' in env.unwrapped.get_action_meanings()
    
    # 只对有生命系统的游戏启用 One_Life_Reset 或 Life_Loss（二选一）
    if has_lives:
        if one_life_reset and life_loss:
            print("警告：one_life_reset 和 life_loss 不应同时启用，优先使用 one_life_reset")
        
        if one_life_reset:
            # One_Life_Reset 内部已处理 fire_reset
            env = One_Life_Reset(env)
        elif life_loss:
            # Life_Loss 需要单独的 Fire_Reset 配合
            if needs_fire:
                env = Fire_Reset(env)
            env = Life_Loss(env, train=train, fire_reset=needs_fire)
        else:
            # 如果都不启用，但游戏需要 FIRE，则添加 Fire_Reset
            if needs_fire:
                env = Fire_Reset(env)
    else:
        # 无生命系统的游戏，如果需要 FIRE 则添加 Fire_Reset
        if needs_fire:
            env = Fire_Reset(env)
    
    env = Frame_Preprocess(env)
    
    if clip_reward:
        env = Clip_Reward(env)
    
    if frame_stack:
        env = Frame_Stack(env, 4)

    return env
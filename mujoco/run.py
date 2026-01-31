"""
MuJoCo PPO 测试脚本 - 优化版
支持优雅的控制台输出、自动设备适配及视频保存
"""
import argparse
import gymnasium as gym
import torch
import numpy as np
import os
import pickle
import sys
from PPO import Actor
from config import *

class Tester:
    def __init__(self, args):
        self.args = args
        self.env_name = args.env_name
        self.config = configs.get(self.env_name)
        self.device = self._get_device()
        
        # 自动补全环境版本
        if not self.env_name.endswith('-v4'):
            self.env_name += '-v4'
            
        # 初始化环境（先判断渲染模式）
        render_mode = "human" if not args.no_render else "rgb_array"
        self.env = gym.make(self.env_name, render_mode=render_mode)
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = self.config['hidden_dim']
        
        # 加载模型与归一化参数
        self.actor = self._load_model()
        self.norm_mean, self.norm_std = self._load_norm_params()

    def _get_device(self):
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self):
        # 1. 确定模型路径
        self.model_path = self.args.model or f"result/{self.env_name}/best_PPO.pt"
        if not os.path.exists(self.model_path):
            print(f"❌ 找不到模型文件: {self.model_path}")
            sys.exit(1)
            
        actor = Actor(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        actor.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        actor.eval()
        print(f"✅ 成功加载模型: {self.model_path}")
        return actor

    def _load_norm_params(self):
        model_dir = os.path.dirname(self.model_path)
        model_name = os.path.basename(self.model_path) # 获取文件名，如 best_PPO.pt
        
        # 根据模型文件名，决定归一化文件名
        if "best" in model_name:
            norm_filename = "best_normalize.pkl"
        else:
            norm_filename = "normalize.pkl"
            
        norm_path = os.path.join(model_dir, norm_filename)
        
        if not os.path.exists(norm_path):
            print(f"⚠️ 找不到对应的 {norm_filename}，尝试加载通用 normalize.pkl")
            norm_path = os.path.join(model_dir, "normalize.pkl")

        if not os.path.exists(norm_path):
            print(f"❌ 严重错误: 找不到任何归一化参数，机器人会乱跳！")
            sys.exit(1)
            
        with open(norm_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ 归一化参数匹配成功: {norm_path}")
        return data['mean'], data['std']

    def normalize(self, state):
        state = (state - self.norm_mean) / (self.norm_std + 1e-8)
        return np.clip(state, -5, 5)

    def run(self):
        print(f"\n🚀 开始测试 {self.env_name} | 设备: {self.device} | 回合数: {self.args.episodes}")
        print("-" * 50)
        
        scores = []
        for ep in range(self.args.episodes):
            state, _ = self.env.reset()
            ep_reward = 0
            frames = []
            
            for t in range(10000): # MuJoCo 通常上限是1000
                state_norm = self.normalize(state)
                state_tensor = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    # 测试时直接取均值 mean，不进行随机采样，动作更稳
                    mean, _ = self.actor(state_tensor)
                    action = mean.cpu().numpy()[0]
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                
                if self.args.save_video and not self.args.no_render:
                    # 如果需要保存视频，注意这里通常需要渲染到rgb_array
                    pass 

                if terminated or truncated:
                    break
            
            scores.append(ep_reward)
            print(f"Episode {ep+1:2d}: Reward = {ep_reward:8.2f} | Steps = {t+1}")

        print("-" * 50)
        print(f"📊 平均得分: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        self.env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Hopper-v4')
    parser.add_argument('--model', type=str, default=None, help="模型路径")
    parser.add_argument('--episodes', type=int, default=5, help="测试多少个回合")
    parser.add_argument('--no_render', action='store_true', help="关闭可视化界面")
    parser.add_argument('--save_video', action='store_true', help="是否保存GIF")
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()

if __name__ == "__main__":
    main()
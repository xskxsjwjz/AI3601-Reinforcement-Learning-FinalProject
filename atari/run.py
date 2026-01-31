import gymnasium as gym
import argparse
import torch
import numpy as np
import time

from Dueling_DQN import Dueling_DQN, Dueling_DQNAgent
from utils import create_env, construct_wrappers

parser = argparse.ArgumentParser(description="评估训练好的 Dueling DQN 模型")
parser.add_argument('--env_name', type=str, default='VideoPinball-v5', help="输入环境名称")
parser.add_argument('--model_path', type=str, default=f'result/best_DQN.pt', help="模型文件路径")
parser.add_argument('--episodes', type=int, default=10, help="评估的回合数")
parser.add_argument('--render', default=False, help="是否渲染游戏画面")
args = parser.parse_args()

ENV_NAME = args.env_name
MODEL_PATH = f"result/{ENV_NAME}/best_DQN.pt"

def evaluate(env_name, model_path, episodes=10, render=False):
    '''
    评估训练好的模型
    
    :param env_name: 环境名称
    :param model_path: 模型文件路径
    :param episodes: 评估的回合数
    :param render: 是否渲染
    '''
    # 创建环境
    render_mode = 'human' if render else None
    env = gym.make(env_name, render_mode=render_mode)
    env = construct_wrappers(
        env, 
        one_life_reset=False,
        life_loss=True,
        clip_reward=False,
        frame_stack=True,
        train=False,
        )
    
    action_space = env.action_space
    input_shape = (4, 84, 84)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = Dueling_DQN(input_shape, action_space.n).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f"成功加载模型: {model_path}")
    except FileNotFoundError:
        print(f"模型文件不存在: {model_path}")
        print("请先运行 train.py 进行训练")
        return
    model.eval()
    
    all_rewards = []
    
    for ep in range(episodes):
        frame, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # 处理帧
            state = torch.from_numpy(
                frame.force().transpose(2, 0, 1)[None]
            ).to(device).float().div(255)
            
            with torch.no_grad():
                q_values = model(state)
                action = q_values.argmax(dim=1).item()
            
            frame, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            if render:
                time.sleep(0.02)
        
        all_rewards.append(episode_reward)
        print(f"Episode {ep + 1}: reward = {episode_reward}, steps = {steps}")
    
    env.close()
    
    print(f"\n评估结果 ({episodes} episodes):")
    print(f"平均奖励: {np.mean(all_rewards):.2f}")
    print(f"最高奖励: {np.max(all_rewards):.2f}")
    print(f"最低奖励: {np.min(all_rewards):.2f}")
    print(f"标准差: {np.std(all_rewards):.2f}")

if __name__ == '__main__':
    evaluate(f"ALE/{ENV_NAME}", MODEL_PATH, args.episodes, args.render)
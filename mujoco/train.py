import argparse
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from collections import deque
from PPO import PPO, Normalize
from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='Hopper-v4')
args = parser.parse_args()

env_name = args.env_name
save_path = f"result/{env_name}"
config = configs[env_name]
max_episodes = config['max_episodes']
batch_size = config['batch_size']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    """
    PPO训练函数
    - max_episodes: 训练多少个episode
    - max_steps: 每次累积多少步经验后学习
    - epochs: 每次学习时从经验中抽样学习多少轮
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_steps = config['max_steps']
    epochs = config['epochs']
    
    ppo = PPO(state_dim, action_dim, config['hidden_dim'], config['ent_coef'], device=device)
    normalize = Normalize(state_dim)
    
    episode_rewards = []
    eval_rewards = []
    recent_rewards = []
    best_avg_reward = -np.inf

    os.makedirs(save_path, exist_ok=True)

    print(f"="*50)
    print(f"正在训练 {env_name} 环境...")
    print(f"训练参数：{config}")
    print(f"="*50)

    episode = 0
    total_steps = 0
    
    while episode < max_episodes:
        memory = deque()
        steps_collected = 0
        batch_episode_rewards = []
        
        state, _ = env.reset()
        state = normalize(state)
        episode_reward = 0
        
        while steps_collected < max_steps:
            state_tensor = torch.from_numpy(np.array(state).astype(np.float32)).unsqueeze(0).to(ppo.device)
            action = ppo.actor.get_action(state_tensor)[0]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = normalize(next_state)
            
            mask = 0 if done else 1
            memory.append([state, action, reward, mask])
            
            episode_reward += reward
            state = next_state
            steps_collected += 1
            total_steps += 1
            
            if done:
                episode += 1
                episode_rewards.append(episode_reward)
                batch_episode_rewards.append(episode_reward)
                recent_rewards.append(episode_reward)
                if len(recent_rewards) > 10:
                    recent_rewards.pop(0)
                
                if episode >= max_episodes:
                    break
                
                state, _ = env.reset()
                state = normalize(state)
                episode_reward = 0
        
        last_state_tensor = torch.from_numpy(np.array(state).astype(np.float32)).unsqueeze(0).to(ppo.device)
        with torch.no_grad():
            next_value = ppo.critic(last_state_tensor).item()

        if len(memory) >= batch_size:
            ppo.train(memory, next_value, epochs=epochs)
        
        if len(batch_episode_rewards) > 0:
            avg_batch = np.mean(batch_episode_rewards)
            avg_recent = np.mean(recent_rewards) if recent_rewards else 0
            
            print(f"Episode: {episode:5d}/{max_episodes} | "
                  f"总步数: {total_steps:8d} | "
                  f"本轮平均: {avg_batch:8.2f} | "
                  f"最近10ep: {avg_recent:8.2f}")
            
            eval_rewards.append(avg_recent)
            
            if avg_recent > best_avg_reward:
                best_avg_reward = avg_recent

                torch.save(ppo.actor.state_dict(), f"{save_path}/best_PPO.pt")
                with open(f"{save_path}/best_normalize.pkl", 'wb') as f:
                    pickle.dump({
                        'mean': normalize.mean, 
                        'std': normalize.std, 
                        'n': normalize.n, 
                        'stdd': normalize.stdd
                    }, f)

                print(f">>> 新最佳模型! 平均奖励: {best_avg_reward:.2f}")

    torch.save(ppo.actor.state_dict(), f"{save_path}/last_PPO.pt")
    with open(f"{save_path}/normalize.pkl", 'wb') as f:
        pickle.dump({'mean': normalize.mean, 'std': normalize.std, 'n': normalize.n, 'stdd': normalize.stdd}, f)

    print(f"\n最终模型已保存至: {save_path}/last_PPO.pt")
    print(f"最佳模型已保存至: {save_path}/best_PPO.pt (最佳平均奖励: {best_avg_reward:.2f})")
    print(f"Normalize状态已保存至: {save_path}/normalize.pkl")

    # 画图
    plot_training_curve(episode_rewards, eval_rewards)
    
    env.close()
    print(f"\n{'='*60}")
    print(f"训练完成！环境: {env_name}, 总步数: {total_steps}")
    print(f"{'='*60}\n")

def main():
    train()
    
def plot_training_curve(episode_rewards, eval_rewards):
    """绘制训练曲线"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 颜色方案
    train_color = '#3498db'
    avg_color = '#e74c3c'
    eval_color = '#2ecc71'
    
    # === 左图：训练奖励 ===
    ax1 = axes[0]
    ax1.plot(episode_rewards, alpha=0.3, color=train_color, linewidth=1, label='Episode Reward')
    
    # 计算移动平均
    window = min(50, len(episode_rewards) // 4) if len(episode_rewards) > 4 else 1
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                 color=avg_color, linewidth=2, label=f'{window}-Episode Moving Avg')
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('Training Reward Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.tick_params(labelsize=10)
    
    # === 右图：评估奖励（每次学习后的10-episode平均）===
    ax2 = axes[1]
    ax2.plot(eval_rewards, color=eval_color, linewidth=2, 
             marker='o', markersize=3, alpha=0.8, label='10-Episode Avg (per update)')
    
    if len(eval_rewards) > 0:
        ax2.fill_between(range(len(eval_rewards)), eval_rewards, alpha=0.2, color=eval_color)
    
    ax2.set_xlabel('Update', fontsize=12)
    ax2.set_ylabel('Average Reward', fontsize=12)
    ax2.set_title('Average Reward per Update', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curve.png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Training curve saved to training_curve.png")
    plt.close()

if __name__ == "__main__":
    main()

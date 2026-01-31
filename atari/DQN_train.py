import argparse
import math
import os
import matplotlib.pyplot as plt
from DQN import *
from utils import *
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='VideoPinball-v5')
args = parser.parse_args()
env_name = args.env_name

# =======================================================================
#   超参数设定
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
max_frame = 500000
lr = 1e-3
update_target = 1000
batch_size = 32
check_freq = 1000
learning_start = 10000
save_freq = 100000
capacity = 10000
# =======================================================================

def set_academic_style():
    sns.set_theme(style="whitegrid")
    
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "lines.linewidth": 1.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
    })

set_academic_style()

def epsilon_decay(frame_idx):
    return epsilon_min + (epsilon_max - epsilon_min) * math.exp(-1. * frame_idx / eps_decay)
    
    
def train(env_name):
    os.makedirs('result', exist_ok=True)
    
    env = create_env(env_name)
    env = construct_wrappers(   
        env, 
        one_life_reset=False, 
        life_loss=True,
        clip_reward=False, 
        frame_stack=True,
        train=True,
    )

    frame, info = env.reset()
    action_space = env.action_space
    input_shape = frame.force().transpose(2, 0, 1).shape
    agent = DQNAgent(input_shape, action_space, capacity=capacity, epsilon=epsilon_max, lr=lr)

    episode_reward = 0
    
    frame_idx, all_rewards, losses, episode_count = [], [], [], 0
    best_reward = float('-inf')

    for i in range(max_frame):
        epsilon = epsilon_decay(i)
        state = agent.observe(frame)
        action = agent.select_action(state, epsilon)

        next_frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        agent.memory_buffer.push(frame, action, reward, next_frame, done)
        
        frame = next_frame

        loss = 0
        if len(agent.memory_buffer) >= learning_start:
            loss = agent.learn(batch_size)

        # 打印训练信息
        if i % check_freq == 0:
            avg_reward = np.mean(all_rewards[-10:]) if len(all_rewards) > 0 else 0
            print(f"frame: {i} | reward: {avg_reward:.2f} | loss: {loss:.4f} | epsilon: {epsilon:.4f} | episode: {episode_count}")

        # 更新目标网络
        if i % update_target == 0:
            agent.target.load_state_dict(agent.policy.state_dict())

        if done:
            frame_idx.append(i / 100)
            losses.append(loss)
            all_rewards.append(episode_reward)
            
            # # 保存最佳模型
            # if episode_reward > best_reward:
            #     best_reward = episode_reward
            #     torch.save(agent.policy.state_dict(), 'result/best_DQN.pt')

            episode_reward = 0
            frame, info = env.reset()
            episode_count += 1

    # 保存最终模型
    # torch.save(agent.policy.state_dict(), 'result/trained_DQN.pt')

    env.close()

    # 绘制优化的训练曲线
    fig = plt.figure(figsize=(16, 5))
    
    # 1. 原始奖励曲线
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(frame_idx, all_rewards, alpha=0.3, color='#2E86AB', linewidth=0.5)
    # 计算移动平均
    if len(all_rewards) >= 10:
        window = 10
        moving_avg = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        moving_avg_x = frame_idx[window-1:]
        ax1.plot(moving_avg_x, moving_avg, color='#A23B72', linewidth=2, label=f'Moving Avg (window={window})')
    ax1.set_xlabel('Training Steps / 100', fontsize=11)
    ax1.set_ylabel('Episode Reward', fontsize=11)
    ax1.set_title('Training Reward Curve', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 奖励分布（最近的episodes）
    ax2 = plt.subplot(1, 3, 2)
    recent_rewards = all_rewards[-100:] if len(all_rewards) >= 100 else all_rewards
    ax2.hist(recent_rewards, bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(recent_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(recent_rewards):.1f}')
    ax2.set_xlabel('Episode Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'Reward Distribution (Last {len(recent_rewards)} Episodes)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. 训练统计信息
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    stats_text = f"""
    Training Statistics
    {'='*40}
    
    Total Episodes: {episode_count}
    Total Frames: {max_frame}
    
    Best Reward: {best_reward:.2f}
    Final Avg (last 10): {np.mean(all_rewards[-10:]):.2f}
    Final Avg (last 100): {np.mean(all_rewards[-100:]):.2f}
    
    Overall Mean: {np.mean(all_rewards):.2f}
    Overall Std: {np.std(all_rewards):.2f}
    Overall Max: {np.max(all_rewards):.2f}
    Overall Min: {np.min(all_rewards):.2f}
    
    Final Epsilon: {epsilon_decay(max_frame):.4f}
    """
    
    ax3.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('result/Analasis/DQN.png', dpi=300, bbox_inches='tight')
    print(f"\n训练曲线已保存到: result/Analasis/DQN.png")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最终平均奖励 (最近100局): {np.mean(all_rewards[-100:]):.2f}")
    
    plt.show()

if __name__ == '__main__':
    train(f"ALE/{env_name}")

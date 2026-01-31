## 测试运行结果

```
Episode 1: reward = 54.0, steps = 883
Episode 2: reward = 72.0, steps = 784
Episode 3: reward = 73.0, steps = 701
Episode 4: reward = 31.0, steps = 1366
Episode 5: reward = 28.0, steps = 1754
Episode 6: reward = 47.0, steps = 1183
Episode 7: reward = 32.0, steps = 1578
Episode 8: reward = 39.0, steps = 1522
Episode 9: reward = 38.0, steps = 1587
Episode 10: reward = 68.0, steps = 908

评估结果 (10 episodes):
平均奖励: 48.20
最高奖励: 73.00
最低奖励: 28.00
标准差: 16.62

```

## 训练参数

```py
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
max_frame = 500000
lr = 1e-4
update_target = 1000
batch_size = 32
check_freq = 1000
learning_start = 10000
save_freq = 100000
capacity = 10000

env = construct_wrappers(   
        env, 
        one_life_reset=False, 
        life_loss=True,
        clip_reward=False,
        frame_stack=True,
        train=True,
    )
```
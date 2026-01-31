## 测试结果

```
Episode 1: reward = 17.0, steps = 2147
Episode 2: reward = 15.0, steps = 2490
Episode 3: reward = 11.0, steps = 2902
Episode 4: reward = 16.0, steps = 2286
Episode 5: reward = 12.0, steps = 2850
Episode 6: reward = 13.0, steps = 2478
Episode 7: reward = 17.0, steps = 2094
Episode 8: reward = 15.0, steps = 2466
Episode 9: reward = 11.0, steps = 3215
Episode 10: reward = 19.0, steps = 2001

评估结果 (10 episodes):
平均奖励: 14.60
最高奖励: 19.00
最低奖励: 11.00
标准差: 2.62
```

## 训练参数

```py
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
max_frame = 1000000
lr = 2e-4
update_target = 1000
batch_size = 32
check_freq = 1000
learning_start = 10000
save_freq = 100000
capacity = 10000


env = construct_wrappers(   
        env, 
        one_life_reset=False, 
        life_loss=False,
        clip_reward=False, 
        frame_stack=True,
        train=True,
    )
```
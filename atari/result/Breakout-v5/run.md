测试运行结果

---

```
Episode 1: reward = 18.0, steps = 664
Episode 2: reward = 16.0, steps = 462
Episode 3: reward = 26.0, steps = 771
Episode 4: reward = 21.0, steps = 512
Episode 5: reward = 9.0, steps = 449
Episode 6: reward = 14.0, steps = 610
Episode 7: reward = 8.0, steps = 419
Episode 8: reward = 35.0, steps = 890
Episode 9: reward = 14.0, steps = 643
Episode 10: reward = 35.0, steps = 840

评估结果 (10 episodes):
平均奖励: 19.60
最高奖励: 35.00
最低奖励: 8.00
标准差: 9.18

```

训练参数

---

```py
gamma = 0.99
epsilon_max = 1
epsilon_min = 0.01
eps_decay = 30000
max_frame = 500000
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
        life_loss=True,
        clip_reward=False, 
        frame_stack=True,
        train=True,
    )
```
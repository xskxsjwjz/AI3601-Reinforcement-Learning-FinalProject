```
🚀 开始测试 Hopper-v4 | 设备: cuda | 回合数: 5
--------------------------------------------------
Episode  1: Reward =  3252.00 | Steps = 1000
Episode  2: Reward =  3265.31 | Steps = 1000
Episode  3: Reward =  3254.47 | Steps = 1000
Episode  4: Reward =  3259.20 | Steps = 1000
Episode  5: Reward =  3252.22 | Steps = 1000
--------------------------------------------------
📊 平均得分: 3256.64 ± 5.05
```

```
'Hopper-v4': {
        'max_episodes': 3000,
        'max_steps': 2048,
        'batch_size': 64,
        'epochs': 10,
        'hidden_dim': 256,
        'lr': 1e-3,
        'ent_coef': 0.01,  # Hopper相对简单，可以少一点熵
    },
```
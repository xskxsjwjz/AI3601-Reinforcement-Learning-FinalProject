a_lr = 3e-4
c_lr = 3e-4
c_l2 = 1e-3
gamma = 0.99
lam = 0.98
epsilon = 0.1
batch_size = 64
hidden_dim = 64

Iter_Hopper = 5000
Iter_Humanoid = 10000
Iter_HalfCheetah = 10000
Iter_Ant = 5000

MAX_STEP = 10000
horizon = 1000

configs = {
    'Hopper-v4': {
        'max_episodes': 2000,
        'max_steps': 2048,
        'batch_size': 64,
        'epochs': 10,
        'hidden_dim': 256,
        'lr': 1e-3,
        # 'lr': 1e-2,
        'ent_coef': 0.01,  # Hopper相对简单，可以少一点熵
        # 'ent_coef': 0.5
    },
    'Humanoid-v4': {
        'max_episodes': 2000,  # 减少，但用更大的batch
        'max_steps': 1000,
        'batch_size': 256,      # 增大batch size
        'epochs': 10,           # 减少epochs防止过拟合
        'hidden_dim': 512,
        'lr': 3e-4,             # 提高学习率
        'ent_coef': 0.001,      # 添加熵正则，鼓励探索！
    },
    'HalfCheetah-v4': {
        'max_episodes': 2000,
        'max_steps': 2048,
        'batch_size': 64,
        'epochs': 10,
        'hidden_dim': 256,
        'lr': 3e-4,
        'ent_coef': 0.001,        # HalfCheetah较简单
    },
    'Ant-v4': {
        'max_episodes': 2000,   # 增加训练时间
        'max_steps': 2048,      # 减少，让更新更频繁
        'batch_size': 128,      # 增大batch
        'epochs': 10,
        'hidden_dim': 256,
        'lr': 3e-4,
        'ent_coef': 0.001,      # 添加熵正则，鼓励探索！
    }
}
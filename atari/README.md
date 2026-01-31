# Atari by Duel DQN

使用 Dueling DQN 算法在 Atari 游戏环境中进行训练和评估。

## 如何测试？

训练好的模型已经放在 `result/{env_name}` 文件夹中。只需要运行以下命令即可测试：

```bash
cd atari  # 确保当前目录在 atari 文件夹下
python run.py --env_name {填入环境名，如 Pong-v5}
```

接下来会自动加载对应的模型并测试，在终端打印测试结果.

## 不想测试?

我也准备了测试视频,存在 `result/{env_name}` 文件夹中,可以直接观看大概的训练结果.
## 如何测试

正如已经写在 atari 和 mujoco 目录中的 `README`，测试时请确保切换到相应目录下，运行下面的命令会加载训练好的最佳模型在指定环境下进行多轮测试并输出每轮奖励和平均奖励。

```bash
cd atari
python run.py --env_name PongNoFrameskip-v4
```

```bash
cd mujoco
python run.py --env_name Hopper-v4 --no_render
```
## 如何测试？

```bash
cd mujoco
python run.py --env_name Hopper-v4
```

这会自动加载我训练好的 `best-PPO.pt` 模型，在指定环境下测试 5 个回合。默认开启 `render`，如果只进行测试，可以使用 `python run.py --env_name Hopper-v4 --no_render` 来关闭渲染以节省资源。
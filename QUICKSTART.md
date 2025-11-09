# 🚀 快速开始指南

## 30秒快速测试

验证修复是否有效（仅需2个epoch，~2分钟）：

```bash
cd /home/user/c2c
./quick_test.sh configs/cross_no_skip.json 2
```

**期望看到**：
```
loss/content: 下降中...
loss/recon: 下降中...
loss/cross: 下降中...  ← 这个是关键！之前不下降
✅ Test completed successfully!
```

---

## 5分钟完整测试

运行一个完整的小实验（10 epochs）：

```bash
python train.py --config configs/cross_no_skip.json --epochs 10
```

**检查WandB**：
1. 打开wandb链接
2. 查看`loss/cross`曲线 - 应该下降
3. 查看`visuals/train`图像 - 应该去噪

---

## 1小时系统测试

测试所有Cross-domain策略，找出最优：

```bash
python run_experiments.py --cross-domain --epochs 10
```

**对比结果**：
- `cross_baseline` (原方法) - Cross loss不下降 ❌
- `cross_no_skip` (推荐) - Cross loss下降 ✅
- `cross_target_skip` (理论最优) - Cross loss下降 ✅

---

## 6小时完整实验

运行所有12个实验（Cross-domain + Loss消融）：

```bash
python run_experiments.py --all --epochs 10
```

这会生成完整的ablation study结果。

---

## 实验结果查看

### 查看实验列表
```bash
python run_experiments.py --summary
```

### WandB对比
1. 访问 https://wandb.ai
2. 查看项目：
   - `c2c-cross-*` - Cross-domain实验
   - `c2c-ablation-*` - Loss消融实验
3. 对比loss曲线和图像质量

---

## 判断成功的标准

✅ **实验成功**：
- Cross loss从~1.0降到<0.5
- 输出图像明显比输入清晰
- Loss曲线平滑，无震荡

❌ **仍有问题**：
- Cross loss平坦或上升
- 图像仍然模糊
- 训练不稳定

如果失败，检查：
1. 数据路径是否正确？`./datasets/fastmri_knee`
2. GPU是否可用？`nvidia-smi`
3. 依赖是否安装？`wandb`, `torch`, `h5py`

---

## 下一步

1. ✅ 运行快速测试验证修复
2. ✅ 运行完整实验找最佳配置
3. ✅ 用最佳配置训练50+ epochs
4. ✅ 在测试集上评估性能

详细说明见：
- `SOLUTION_SUMMARY.md` - 问题诊断和解决方案
- `EXPERIMENTS.md` - 实验设计详情
- `configs/README.md` - 配置文件说明

# 配置文件说明

本目录包含所有实验配置文件。

---

## 📂 文件组织

### Cross-Domain策略实验（5个配置）

测试不同的skip connection策略：

| 文件 | 策略 | Content | Recon | Cross | Edge |
|-----|------|---------|-------|-------|------|
| `cross_baseline.json` | use_source_skip | 1.0 | 0.5 | 1.0 | 0 |
| `cross_no_skip.json` | **no_skip** | 1.0 | 0.5 | 1.0 | 0 |
| `cross_target_skip.json` | use_target_skip | 1.0 | 0.5 | 1.0 | 0 |
| `cross_zero_skip.json` | zero_skip | 1.0 | 0.5 | 1.0 | 0 |
| `cross_mixed_skip.json` | mixed_skip (α=0.5) | 1.0 | 0.5 | 1.0 | 0 |

**推荐先测试**：`cross_no_skip.json` 或 `cross_target_skip.json`

---

### Loss消融实验（7个配置）

测试不同loss组合的效果：

| 文件 | Content | Recon | Cross | Edge | 测试目的 |
|-----|---------|-------|-------|------|----------|
| `ablation_all_losses.json` | 1.0 | 0.5 | 1.0 | **0.05** | 完整配置（推荐） |
| `ablation_no_content.json` | **0** | 0.5 | 1.0 | 0 | 无content loss |
| `ablation_no_recon.json` | 1.0 | **0** | 1.0 | 0 | 无recon loss |
| `ablation_no_cross.json` | 1.0 | 0.5 | **0** | 0 | 无cross loss |
| `ablation_only_cross.json` | **0** | **0** | 1.0 | 0 | 仅cross（原问题） |
| `ablation_content_recon.json` | 1.0 | 1.0 | **0** | 0 | 无去噪baseline |
| `ablation_strong_edge.json` | 1.0 | 0.5 | 1.0 | **0.2** | 强edge正则化 |

**推荐基线**：`ablation_all_losses.json`

---

## 🔧 共同配置

所有配置文件共享以下设置（除非特别说明）：

```json
{
  "seed": 1337,
  "device": "cuda:0",
  "data": {
    "representation": "complex",
    "crop_size": 256,
    "batch_size": 4
  },
  "model": {
    "encoder": "UNetEncoder",
    "decoder_a/b": "UNetADecoder/BDecoder",
    "latent_channels": 256,
    "identity_mapping": false  // 已禁用，避免zero初始化问题
  },
  "optimizer": {
    "type": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.01
  },
  "trainer": {
    "noise": {
      "sigma_a": 0.01,  // 降低25倍（原0.25）
      "sigma_b": 0.01
    },
    "epochs": 10,
    "grad_clip_norm": 1.0
  }
}
```

---

## 🎯 关键差异

### Cross-Domain策略参数

```json
// 在trainer配置中
"cross_domain_strategy": "no_skip" | "use_source_skip" | "use_target_skip" | "zero_skip" | "mixed_skip"
"mixed_skip_alpha": 0.5  // 仅当strategy="mixed_skip"时使用
```

### Loss权重参数

```json
// 在trainer配置中
"loss_weights": {
  "content": 0.0 ~ 1.0,  // 共享latent约束
  "recon": 0.0 ~ 1.0,    // 重建质量
  "cross": 0.0 ~ 1.0,    // 去噪能力
  "edge": 0.0 ~ 0.2      // 细节保持
}
```

---

## 🚀 使用示例

### 快速测试单个配置（2 epochs）
```bash
./quick_test.sh configs/cross_no_skip.json 2
```

### 完整训练单个配置（10 epochs）
```bash
python train.py --config configs/cross_no_skip.json --epochs 10
```

### 批量运行所有Cross-Domain实验
```bash
python run_experiments.py --cross-domain --epochs 10
```

### 批量运行所有Loss消融实验
```bash
python run_experiments.py --loss-ablation --epochs 10
```

---

## 📊 预期结果

### 成功标准

✅ **Cross loss应该下降**
- 如果不下降 → skip策略有问题
- 推荐尝试：no_skip 或 target_skip

✅ **图像应该去噪**
- 如果模糊 → 检查loss权重
- 如果失败 → 可能需要content + recon + cross

✅ **训练应该稳定**
- 如果震荡 → 降低学习率或增加grad_clip

### 失败案例诊断

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Cross loss平坦 | skip策略错误 | 用`no_skip` |
| 输出模糊 | 缺少recon loss | 设置recon≥0.5 |
| 没有去噪 | 缺少cross loss | 设置cross=1.0 |
| Loss不下降 | 缺少content loss | 设置content=1.0 |
| 过度平滑 | 缺少edge loss | 添加edge=0.05 |

---

## 📝 修改配置

如果需要自定义配置：

1. **复制现有配置**
   ```bash
   cp configs/cross_no_skip.json configs/my_custom.json
   ```

2. **修改参数**
   - 调整`loss_weights`
   - 更改`cross_domain_strategy`
   - 修改`epochs`, `batch_size`, `learning_rate`等

3. **更新wandb project名称**
   ```json
   "logging": {
     "project": "my-custom-experiment"
   }
   ```

4. **运行测试**
   ```bash
   ./quick_test.sh configs/my_custom.json 2
   ```

---

## 🔬 推荐实验流程

### 阶段1：找最佳Cross-Domain策略（~2小时）

```bash
# 测试所有策略
python run_experiments.py --cross-domain --epochs 10

# 或手动逐个测试
python train.py --config configs/cross_no_skip.json --epochs 10
python train.py --config configs/cross_target_skip.json --epochs 10
```

**观察**：哪个配置的cross loss下降最快？

---

### 阶段2：确定最佳Loss组合（~3小时）

用阶段1找到的最佳策略，测试loss组合：

```bash
# 先更新ablation配置中的cross_domain_strategy为最佳策略
# 然后运行
python run_experiments.py --loss-ablation --epochs 10
```

**观察**：哪些loss是必需的？edge loss是否改善细节？

---

### 阶段3：Fine-tuning（~1小时）

基于最佳配置，微调超参数：
- 调整loss权重比例
- 尝试不同学习率
- 调整batch size

---

## 📧 Questions?

查看主项目README和EXPERIMENTS.md获取更多信息。

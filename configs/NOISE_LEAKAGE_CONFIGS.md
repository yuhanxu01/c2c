# Noise Leakage问题专用配置

## 🎯 核心问题回顾

**Skip connections导致noise pattern leakage**：
- encoder(x_a_noisy) → z_a, skips_a（skips包含noise_a模式）
- decoder_a(z_b, skips_a) → 输出leak noise_a
- **破坏Noise2Noise的独立噪声假设！**

---

## 📋 新配置对比

| 配置 | Content | Recon | Cross | Edge | Latent | 策略 | 目的 |
|------|---------|-------|-------|------|--------|------|------|
| **pure_noise2noise.json** | 1.0 | 0.2 | 1.0 | 0 | 256 | no_skip | 标准Noise2Noise（推荐） |
| **extreme_denoising.json** | 0.5 | 0.05 | 1.0 | 0 | 512 | no_skip | 极端去噪（最大化cross loss） |
| **progressive_training.json** | 1.0 | 1.0 | 0.1 | 0 | 256 | no_skip | 渐进式训练Stage1（warm-up） |
| **progressive_training_stage2.json** | 0.5 | 0.1 | 1.0 | 0.05 | 256 | no_skip | 渐进式训练Stage2（去噪） |

---

## 🔬 pure_noise2noise.json（推荐首选）

### 设计理念
```
核心思想：平衡重建和去噪
- Weak recon (0.2): 不强迫latent保留噪声
- Strong cross (1.0): 去噪主导
- Content (1.0): 强制共享表示
- No skip: 完全避免噪声泄漏
```

### 配置要点
```json
{
  "loss_weights": {
    "content": 1.0,   // 强制z_a ≈ z_b
    "recon": 0.2,     // 弱重建（关键！）
    "cross": 1.0      // 去噪主导
  },
  "cross_domain_strategy": "no_skip",  // 避免噪声泄漏
  "latent_channels": 256,
  "epochs": 20
}
```

### 预期效果
- ✅ Cross loss稳定下降
- ✅ 去噪效果明显
- ✅ 无噪声泄漏artifacts
- ⚠️  可能比有skip的细节略少（可接受）

### 使用
```bash
python train.py --config configs/pure_noise2noise.json --epochs 20
```

---

## 🚀 extreme_denoising.json（极端去噪）

### 设计理念
```
最大化去噪能力：
- Very weak recon (0.05): 几乎不学重建
- Strong cross (1.0): 完全依赖Noise2Noise
- Weak content (0.5): 允许一定差异
- Large latent (512): 补偿无skip的信息损失
```

### 配置要点
```json
{
  "loss_weights": {
    "content": 0.5,   // 适度对齐
    "recon": 0.05,    // 极弱重建（关键！）
    "cross": 1.0      // 完全去噪
  },
  "latent_channels": 512,  // 2倍容量
  "features": [512, 256, 128]  // 更大网络
}
```

### 预期效果
- ✅✅ 最强去噪能力
- ✅ Cross loss下降最快
- ⚠️  可能需要更多epoch收敛
- ⚠️  参数量增加（512维latent）

### 适用场景
- 噪声很强的数据
- 对去噪质量要求极高
- 可以接受更大模型

### 使用
```bash
python train.py --config configs/extreme_denoising.json --epochs 30
```

---

## 📈 progressive_training.json（渐进式训练）

### 两阶段训练策略

#### Stage 1: Warm-up（10 epochs）
```json
{
  "loss_weights": {
    "content": 1.0,   // 学习共享表示
    "recon": 1.0,     // 学习重建
    "cross": 0.1      // 轻微去噪信号
  },
  "epochs": 10
}
```

**目标**：
- 快速学习基本编码-解码能力
- 建立z_a ≈ z_b的共享空间
- 避免一开始就过度去噪导致不稳定

#### Stage 2: Denoising（40 epochs）
```json
{
  "loss_weights": {
    "content": 0.5,   // 允许微调
    "recon": 0.1,     // 弱重建
    "cross": 1.0,     // 专注去噪
    "edge": 0.05      // 添加细节保持
  },
  "learning_rate": 0.0001,  // 降低LR
  "epochs": 40,
  "checkpoint": "stage1"  // 加载stage1权重
}
```

**目标**：
- 在stage1基础上fine-tune去噪
- 更稳定的训练过程
- 保留细节（edge loss）

### 使用流程
```bash
# Step 1: Warm-up training
python train.py --config configs/progressive_training.json \
                --epochs 10 \
                --run-dir runs/progressive_stage1

# Step 2: Denoising training
python train.py --config configs/progressive_training_stage2.json \
                --epochs 40 \
                --run-dir runs/progressive_stage2
```

### 预期效果
- ✅ 最稳定的训练曲线
- ✅ 避免早期崩溃
- ✅ 最终质量可能最高
- ⚠️  总训练时间最长（50 epochs）

---

## 🆚 与原配置对比

### 原配置（有问题）
```json
{
  "loss_weights": {
    "content": 0,     // ❌ 无共享表示
    "recon": 0,       // ❌ 无重建监督
    "cross": 1        // ❌ 单独cross无法优化
  },
  "cross_domain_strategy": "use_source_skip",  // ❌ 噪声泄漏
  "noise_sigma": 0.25  // ❌ 噪声过大
}
```

**问题**：
1. 无基础监督（content + recon）
2. Skip导致噪声泄漏
3. 噪声过大干扰训练

### 新配置（修复）
```json
{
  "loss_weights": {
    "content": 1.0,   // ✅ 强制共享
    "recon": 0.2,     // ✅ 适度重建
    "cross": 1.0      // ✅ 去噪主导
  },
  "cross_domain_strategy": "no_skip",  // ✅ 无泄漏
  "noise_sigma": 0.01  // ✅ 合理噪声
}
```

---

## 🎯 推荐使用顺序

### 方案A：快速验证
```bash
# 1. 先测试pure_noise2noise（20 epochs）
python train.py --config configs/pure_noise2noise.json --epochs 20

# 2. 如果效果不够好，尝试extreme_denoising（30 epochs）
python train.py --config configs/extreme_denoising.json --epochs 30
```

**总时间**：~50 epochs

---

### 方案B：最佳质量（推荐）
```bash
# 1. Stage1: Warm-up (10 epochs)
python train.py --config configs/progressive_training.json \
                --epochs 10 \
                --run-dir runs/stage1

# 2. Stage2: Fine-tune denoising (40 epochs)
python train.py --config configs/progressive_training_stage2.json \
                --epochs 40 \
                --run-dir runs/stage2
```

**总时间**：50 epochs（但质量最高）

---

## 📊 评估指标

### 成功标准

1. **Cross loss下降**
   ```
   Epoch 1:  ~1.0
   Epoch 10: <0.5  ✅
   Epoch 20: <0.3  ✅✅
   ```

2. **去噪效果**
   - WandB图像：输出明显比输入清晰
   - 无噪声模式artifact
   - 结构保持完整

3. **Loss平衡**
   ```
   Content loss: 快速降到~0.01（z_a ≈ z_b形成）
   Recon loss: 稳定但不强求接近0（不保留噪声）
   Cross loss: 持续下降（去噪生效）
   ```

4. **无噪声泄漏**
   - cross-domain重建无source domain的噪声pattern
   - 输出噪声应该是随机的，不相关的

---

## 🔧 调试指南

### 如果Cross loss还是不下降

**检查项**：
1. ✅ 确认`cross_domain_strategy: "no_skip"`
2. ✅ 确认recon权重<1.0（推荐0.1-0.2）
3. ✅ 确认content权重>0（推荐1.0）
4. ✅ 数据确实是配对的（相同clean content）

**尝试**：
- 降低recon到0.05（extreme_denoising）
- 增加latent维度到512
- 使用渐进式训练

---

### 如果图像太模糊

**原因**：Recon权重太小 or latent容量不足

**解决**：
- 增加latent_channels（256→512）
- 适度提高recon权重（0.2→0.3）
- 添加edge loss（weight=0.05）

---

### 如果训练不稳定

**原因**：Cross loss权重过大，early collapse

**解决**：
- 使用渐进式训练
- 降低初始学习率（0.0003→0.0001）
- 增加grad_clip（1.0→0.5）

---

## 💡 核心要点

1. **必须no_skip for cross-domain** → 避免噪声泄漏
2. **降低recon权重** → 不强迫保留噪声
3. **增强cross权重** → 去噪主导
4. **保持content权重** → 共享表示
5. **考虑增大latent** → 补偿无skip信息损失

---

## ✅ 立即开始

最简单的验证：
```bash
# 快速测试（2 epochs验证）
./quick_test.sh configs/pure_noise2noise.json 2

# 如果成功，完整训练
python train.py --config configs/pure_noise2noise.json --epochs 20
```

预期：Cross loss应该在前5个epoch内明显下降！

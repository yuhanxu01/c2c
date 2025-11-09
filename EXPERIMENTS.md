# Contrast2Contrast 消融实验设计

本文档描述了系统性的消融实验，用于确定最佳训练策略。

---

## 🎯 实验目标

1. **找到最佳的Cross-Domain重建策略**：解决skip connection不匹配问题
2. **确定必要的Loss组合**：找出content, recon, cross, edge loss的最优配置
3. **验证修复的有效性**：确认cross loss能够正常下降

---

## 🔬 实验组A：Cross-Domain策略对比

### 问题背景

原始实现使用源域的skip connections进行跨域重建，导致语义-结构不匹配：
```python
# 问题：用PDFS的skip来重建PD图像
x_a_from_b = decoder_a(z_b, skips=skips_b)  # ❌ 语义(PDFS) + 结构(PDFS) → 目标(PD)
```

### 实验配置

| 配置文件 | 策略 | 说明 | 预期效果 |
|---------|------|------|---------|
| `cross_baseline.json` | `use_source_skip` | 原始方法（用源域skip） | ❌ Cross loss不下降 |
| `cross_target_skip.json` | `use_target_skip` | 用目标域skip | ✅ 结构匹配，可能最优 |
| `cross_no_skip.json` | `no_skip` | 完全不用skip | ✅ 强制latent自足 |
| `cross_zero_skip.json` | `zero_skip` | 零值skip | ⚠️ 类似no_skip但保持架构 |
| `cross_mixed_skip.json` | `mixed_skip` | 源+目标skip混合(α=0.5) | ⚠️ 折中方案 |

### 评估指标

- **Cross Loss下降速度**：能否有效优化
- **重建质量**：PSNR/SSIM相对于ground truth
- **去噪效果**：噪声消除程度
- **训练稳定性**：loss曲线平滑度

### 预期结果

**最佳候选**：
1. **use_target_skip**：理论最优，skip与目标结构对齐
2. **no_skip**：次优，依赖更强的latent表示

**失败预期**：
- **use_source_skip**：原始问题，cross loss难以下降

---

## 🧪 实验组B：Loss组合消融

### 问题背景

当前配置仅使用cross loss（content=0, recon=0, cross=1），缺乏必要的监督信号。

### 实验配置

| 配置文件 | Content | Recon | Cross | Edge | 目的 |
|---------|---------|-------|-------|------|------|
| `ablation_all_losses.json` | 1.0 | 0.5 | 1.0 | 0.05 | 全部loss组合（推荐基线） |
| `ablation_no_content.json` | 0 | 0.5 | 1.0 | 0 | 测试content loss必要性 |
| `ablation_no_recon.json` | 1.0 | 0 | 1.0 | 0 | 测试recon loss必要性 |
| `ablation_no_cross.json` | 1.0 | 0.5 | 0 | 0 | 测试cross loss必要性（去噪能力） |
| `ablation_only_cross.json` | 0 | 0 | 1.0 | 0 | 复现原始问题 |
| `ablation_content_recon.json` | 1.0 | 1.0 | 0 | 0 | 无cross的baseline |
| `ablation_strong_edge.json` | 1.0 | 0.5 | 1.0 | 0.2 | 测试强edge loss效果 |

### 各Loss的理论作用

#### 1. Content Loss: `L_content = |z_a - z_b|`
**作用**：强制共享latent空间
- ✅ 确保两域编码相似
- ✅ 使cross-domain重建可行
- ❌ 过强会抑制域特异性特征

**预期**：
- 无content loss → z_a和z_b完全不同 → cross重建失败
- 有content loss → 提供共享表示的基础

#### 2. Reconstruction Loss: `L_recon = |decoder_a(z_a) - x_a| + |decoder_b(z_b) - x_b|`
**作用**：教decoder如何正确解码
- ✅ 提供直接监督
- ✅ 防止decoder退化
- ✅ 快速收敛的基础

**预期**：
- 无recon loss → decoder不知道如何重建 → 输出模糊
- 有recon loss → 清晰的同域重建

#### 3. Cross Loss: `L_cross = |decoder_a(z_b) - x_a| + |decoder_b(z_a) - x_b|`
**作用**：Noise2Noise去噪
- ✅ 利用配对数据的结构一致性
- ✅ 核心去噪机制
- ⚠️ 需要content和recon提供基础

**预期**：
- 无cross loss → 无去噪效果
- 有cross loss（配合其他loss） → 有效去噪

#### 4. Edge Loss: `L_edge = |sobel(x_from_y) - sobel(x_gt)|`
**作用**：保持高频细节
- ✅ 防止过度平滑
- ✅ 保留边缘和纹理
- ⚠️ 权重过大可能引入伪影

**预期**：
- 无edge loss → 可能过度平滑
- 适度edge loss → 更清晰的细节
- 过强edge loss → 可能不稳定

### 评估指标

- **Loss下降情况**：每个loss的收敛曲线
- **PSNR/SSIM**：量化重建质量
- **视觉质量**：wandb图像日志
- **去噪效果**：对比有/无cross loss

### 预期最佳组合

```python
"loss_weights": {
    "content": 1.0,    # 必需：共享latent
    "recon": 0.5,      # 必需：教decoder重建
    "cross": 1.0,      # 必需：去噪
    "edge": 0.05       # 可选：保持细节
}
```

---

## 🚀 运行实验

### 查看所有实验
```bash
python run_experiments.py --summary
```

### 运行所有实验
```bash
python run_experiments.py --all --epochs 10
```

### 仅运行Cross-Domain实验
```bash
python run_experiments.py --cross-domain --epochs 10
```

### 仅运行Loss消融实验
```bash
python run_experiments.py --loss-ablation --epochs 10
```

### 运行单个配置
```bash
python run_experiments.py --config configs/cross_no_skip.json --epochs 10
```

### 预演（不实际运行）
```bash
python run_experiments.py --all --dry-run
```

---

## 📊 结果分析

### 关键观察点

1. **Cross Loss趋势**
   - ✅ 应该稳定下降
   - ❌ 如果平坦或上升，说明策略失败

2. **重建质量**
   - 对比wandb图像：`[noisy_a, denoised_a, noisy_b, denoised_b]`
   - 检查cross-domain重建：`cross_abs_a`, `cross_abs_b`

3. **Loss平衡**
   - Content loss快速下降到小值 → 共享表示形成
   - Recon loss稳定下降 → decoder学习重建
   - Cross loss下降 → 去噪生效

### 判断标准

| 现象 | 诊断 | 解决方案 |
|------|------|---------|
| Cross loss不下降 | Skip策略错误 | 改用`no_skip`或`target_skip` |
| 所有loss都很大 | 权重配置问题 | 使用推荐配置 |
| 图像模糊 | Recon loss太小 | 增加recon权重 |
| 过度平滑 | 缺少edge loss | 添加edge loss (0.05-0.1) |
| Content loss不下降 | Encoder未学到共享表示 | 检查数据配对 |

---

## 📈 预期时间线

- **单个实验**（10 epochs）：~30分钟（取决于GPU）
- **Cross-Domain组**（5个配置）：~2.5小时
- **Loss消融组**（7个配置）：~3.5小时
- **全部实验**（12个配置）：~6小时

**建议**：
1. 先运行Cross-Domain组，找出最佳策略
2. 用最佳策略运行Loss消融组
3. 根据结果fine-tune超参数

---

## ✅ 成功标准

实验成功的标志：

1. ✅ **Cross loss下降**：从初始值降低至少50%
2. ✅ **去噪效果**：输出图像明显比输入清晰
3. ✅ **训练稳定**：loss曲线平滑，无震荡
4. ✅ **质量提升**：PSNR/SSIM优于baseline

如果满足以上标准，说明问题已解决！

---

## 🔄 下一步

实验完成后：

1. **分析WandB日志**：对比所有实验的loss曲线
2. **选择最佳配置**：基于量化指标和视觉质量
3. **扩展训练**：用最佳配置训练50+ epochs
4. **评估测试集**：在held-out数据上验证
5. **发布最佳模型**：保存checkpoint供后续使用

---

## 📝 Notes

- 所有配置使用相同的随机种子（1337）确保可复现
- 噪声水平统一设为0.01（比原始0.25低25倍）
- Identity mapping已禁用（避免zero初始化问题）
- 默认使用`no_skip`策略（除非在Cross-Domain实验中测试其他策略）

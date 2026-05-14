# Mysteric-Net 完整工作流程指南

本指南详细介绍如何创建环境、训练模型、生成测试集、评估模型以及提取摩擦参数。

## 1. 环境配置

### 创建Conda环境

```bash
conda env create -f environment.yml
conda activate frictionest
```

环境包含：
- Python 3.11
- PyTorch 2.0+ (CPU版本)
- NumPy
- Pytest

## 2. 数据准备

### 生成训练数据集

```bash
python scripts/generate_dataset.py
```

可选参数：
- `--T`: 轨迹长度（默认：8000）
- `--seq-len`: 序列长度（默认：30）
- `--out`: 输出路径（默认：`data/synthetic_2dof_inverse.npz`）

示例：
```bash
python scripts/generate_dataset.py --T 16000 --seq-len 30
```

### 生成测试数据集

```bash
python scripts/generate_test_dataset.py
```

可选参数：
- `--T`: 测试轨迹长度（默认：2000）
- `--seq-len`: 序列长度（默认：30）
- `--seed`: 随机种子（默认：42，确保与训练集不同）
- `--out`: 输出路径（默认：`data/test_synthetic_2dof_inverse.npz`）

## 3. 模型训练

### 基础训练（仅MSE损失）

```bash
python examples/synthetic_train.py --data data/synthetic_2dof_inverse.npz
```

### 完整训练（包含能量损失，速度较慢）

```bash
python examples/synthetic_train.py --data data/synthetic_2dof_inverse.npz --energy-loss
```

### 训练参数说明

- `--data`: 训练数据文件路径
- `--epochs`: 训练轮数（默认：25）
- `--batch`: 批大小（默认：256）
- `--energy-loss`: 启用论文式完整损失函数（l_tau + l_E）
- `--save-dir`: 模型保存目录（默认：`checkpoints/`）
- `--save-name`: 模型保存名称（默认：`mysteric_net`）

### 内存模式训练（不保存数据文件）

```bash
python examples/synthetic_train.py --T 4000
```

## 4. 模型评估

### 在测试集上评估模型

```bash
python scripts/evaluate_model.py
```

### 评估参数说明

- `--model`: 训练好的模型路径（默认：`checkpoints/mysteric_net.pt`）
- `--test-data`: 测试数据集路径（默认：`data/test_synthetic_2dof_inverse.npz`）
- `--save-predictions`: 预测结果保存路径（默认：`results/test_predictions.npz`）

### 评估输出

脚本会输出：
- 总力矩RMSE和MAE
- 各关节力矩的RMSE和MAE
- 摩擦力矩的RMSE
- 摩擦力矩的统计信息（范围、均值等）

预测结果包含：
- `tau_true`: 真实总力矩
- `tau_pred`: 预测总力矩
- `tau_core_pred`: 预测刚体力矩
- `tau_fri_true`: 真实摩擦力矩
- `tau_fri_pred`: 预测摩擦力矩
- `q`, `qd`, `qdd`: 关节位置、速度、加速度
- 各种评估指标

## 5. 提取摩擦参数

### 提取模型参数

```bash
python scripts/extract_friction_params.py
```

### 参数提取说明

- `--model`: 训练好的模型路径（默认：`checkpoints/mysteric_net.pt`）
- `--output-dir`: 参数输出目录（默认：`results/`）

### 输出内容

脚本会生成 `results/model_parameters.json` 文件，包含：

1. **检查点信息**
   - 训练轮数
   - 训练RMSE
   - 自由度和序列长度

2. **H-Net（摩擦网络）参数**
   - TCN层权重和偏置
   - Head层权重和偏置
   - 网络结构信息
   - 权重统计信息

3. **L-Net（刚体网络）参数**
   - 质量矩阵网络权重
   - 势能网络权重
   - 网络结构信息

4. **总体参数统计**
   - 总参数数量
   - H-Net参数数量
   - L-Net参数数量

## 6. 完整工作流程示例

```bash
# 1. 激活环境
conda activate frictionest

# 2. 生成训练数据
python scripts/generate_dataset.py --T 8000

# 3. 训练模型
python examples/synthetic_train.py --data data/synthetic_2dof_inverse.npz --epochs 25

# 4. 生成测试数据
python scripts/generate_test_dataset.py --T 2000 --seed 42

# 5. 评估模型
python scripts/evaluate_model.py --model checkpoints/mysteric_net.pt --test-data data/test_synthetic_2dof_inverse.npz

# 6. 提取摩擦参数
python scripts/extract_friction_params.py --model checkpoints/mysteric_net.pt
```

## 7. 网络结构说明

### Mysteric-Net 架构

```
输入 (q, qd, qdd, q_seq, qd_seq)
    ├── L-Net (刚体逆动力学)
    │   ├── Mass Network: M(q)
    │   ├── Coriolis: C(q, qd)qd
    │   ├── Gravity: g(q)
    │   └── 输出: τ_core = Mqdd + Cqd + g
    │
    └── H-Net (摩擦力矩)
        ├── TCN Layers (Conv1d + ReLU)
        │   ├── Layer 1: Conv1d(4 → 8, kernel=3)
        │   ├── ReLU
        │   ├── Layer 2: Conv1d(8 → 8, kernel=3)
        │   └── ReLU
        └── Head Layer: Linear(8 → 2)
            └── 输出: τ_fri

总输出: τ_hat = τ_core + τ_fri
```

### 物理信息集成

- **L-Net** 基于拉格朗日方程，学习刚体动力学
- **H-Net** 学习历史相关的摩擦力矩
- **能量损失** 确保物理一致性（可选启用）

## 8. 结果分析

### 预期性能指标

- 训练RMSE: < 0.01 (合成数据)
- 测试RMSE: < 0.02 (合成数据)
- 摩擦力矩RMSE: < 0.005

### 摩擦特性分析

通过提取的参数和预测结果，可以分析：
- 摩擦力矩与速度的关系
- 摩擦的迟滞特性
- 不同关节的摩擦差异

## 9. 故障排除

### 常见问题

1. **CUDA内存不足**
   - 使用CPU模式：`export CUDA_VISIBLE_DEVICES=""`
   - 减小批大小：`--batch 128`

2. **训练不收敛**
   - 增加训练轮数：`--epochs 50`
   - 启用能量损失：`--energy-loss`

3. **模型文件不存在**
   - 确保训练完成且保存成功
   - 检查 `checkpoints/` 目录

## 10. 进阶使用

### 自定义数据集

修改 `scripts/generate_dataset.py` 中的 `simulate_2dof_inverse_dynamics` 函数来生成不同的轨迹。

### 调整网络结构

修改 `examples/synthetic_train.py` 中的模型初始化参数：
- `lnet_hidden`: L-Net隐藏层维度（默认：32）
- `lnet_layers`: L-Net隐藏层数（默认：2）
- `hnet_channels`: H-Net隐藏通道数（默认：8）
- `hnet_kernel`: H-Net卷积核大小（默认：3）

### 可视化

使用 `results/test_predictions.npz` 中的数据进行可视化分析：
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('results/test_predictions.npz')
# 绘制摩擦力矩对比
plt.figure(figsize=(12, 4))
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.plot(data['tau_fri_true'][:500, i], label='True', alpha=0.7)
    plt.plot(data['tau_fri_pred'][:500, i], label='Predicted', alpha=0.7)
    plt.title(f'Joint {i+1} Friction Torque')
    plt.legend()
plt.tight_layout()
plt.savefig('friction_comparison.png')
```

## 联系与支持

如有问题，请检查：
1. 环境是否正确配置
2. 数据文件是否存在
3. 模型文件是否成功保存
4. 路径是否正确

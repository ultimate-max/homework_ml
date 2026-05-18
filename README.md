# DeLaN_Stribeck / RobotDynamics

工业机械臂动力学学习：**DeLaN**（拉格朗日刚体）+ **摩擦模块**（TCN / Stribeck SCV / PINN），Python 包名为 `RobotDynamics`。

## 项目结构

```text
DeLaN_Stribeck/
├── RobotDynamics/          # 主代码包
│   ├── DeLaN/              # L-Net、数据加载、训练与评估
│   ├── FrictionModule/     # TCN、FO 级联、Stribeck、合成数据
│   └── MystericNet/        # DeLaN + 摩擦联合模型
├── examples/
│   ├── delan_train.py      # 仅训练 DeLaN（刚体）
│   ├── delan_evaluate.py   # DeLaN 测试与绘图
│   ├── robot_train.py      # L-Net + 摩擦（机械臂 pickle）
│   └── synthetic_train.py  # 2-DoF 合成数据冒烟训练
├── scripts/
│   └── import_delan_data.py  # .mat / .npz → .pickle
├── data/                   # 数据与导入结果
├── checkpoints/            # 模型权重
├── figures/                # 评估图
├── requirements.txt
└── environment.yml
```

---

## 1. 环境安装

### 方式 A：Conda（推荐）

```bash
cd /path/to/DeLaN_Stribeck
conda env create -f environment.yml
conda activate frictionest
```

环境名默认为 `frictionest`（见 `environment.yml`）。若已存在环境，可更新：

```bash
conda env update -f environment.yml --prune
conda activate frictionest
```

### 方式 B：已有 Python 环境

```bash
cd /path/to/DeLaN_Stribeck
pip install -r requirements.txt
```

### GPU 说明

`requirements.txt` 通过 PyTorch 官方 **CUDA 12.1** 轮子安装 GPU 版 PyTorch。**不要**用 conda 安装 `pytorch` 的 `cpuonly` 包，否则 `torch.cuda.is_available()` 为 `False`。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

无 GPU 时训练脚本仍可在 CPU 上运行，将 `-c 0` 或依赖脚本默认的 CPU 回退即可。

### 依赖概览

| 包 | 用途 |
|----|------|
| torch | 网络训练 |
| numpy, dill | 数据与 pickle |
| scipy | 读取 MATLAB `.mat` |
| matplotlib | 评估绘图 |
| pytest | 单元测试（可选） |

---

## 2. 数据准备：MATLAB `.mat` → `.pickle`

训练使用 **DeLaN 官方格式** 的 `character_data.pickle`：每条轨迹含 `t, qp, qv, qa, tau`，可选 `m, c, g, p, pdot`。

### 2.1 MATLAB 中应保存的变量

**多轨迹（cell 数组，推荐）** — 每条轨迹长度 `T`，关节数为 `n_dof`，矩阵形状为 **`(T, n_dof)`**（时间在行上）：

```matlab
labels = {'e', 'v', 'q', 'a', 'b'};   % 轨迹标签（单字符或短字符串）
t  = {t1, t2, ...};                   % 各 T×1
qp = {qp1, qp2, ...};                 % 关节角
qv = {qv1, qv2, ...};                 % 角速度
qa = {qa1, qa2, ...};                 % 角加速度
tau = {tau1, tau2, ...};              % 关节力矩
m = {m1, m2, ...};                    % 惯性项力矩（可选，用于分解评估）
c = {c1, c2, ...};                    % 科氏/离心项
g = {g1, g2, ...};                    % 重力项
p = {p1, p2, ...};                    % 可置零
pdot = {pdot1, pdot2, ...};           % 可置零

save('robot.mat', 'labels','t','qp','qv','qa','tau','m','c','g','p','pdot', '-v7');
```

**单条轨迹**：`qp, qv, ...` 为 `T×n_dof` 数值矩阵，`labels` 为单个标签。

若在 MATLAB 中矩阵存为 **`n_dof×T`**，导入时加 `--transpose`。

**全部包在一个 struct 里**（例如 `character_data`）时，可用 `--root character_data`；若 `.mat` 只有一个顶层 struct，可省略 `--root`。

### 2.2 命令行转换

```bash
conda activate frictionest
cd /path/to/DeLaN_Stribeck

# 基本转换
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle

# MATLAB 存成 n_dof×T 时
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle \
  --transpose

# 指定 struct 名
python scripts/import_delan_data.py \
  -i data/character_data.mat \
  -o data/robot.pickle \
  --root character_data

# 查看 pickle 摘要（轨迹数、n_dof、是否有 m/c/g）
python scripts/import_delan_data.py --inspect data/robot.pickle

# 导入并打印推荐超参
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle \
  --transpose \
  --suggest-hyper
```

也可从 **`.npz`** 导入（键名 `qp, qv, qa, tau` 或多轨迹 `traj0_qp, traj0_qv, ...`），用法见 `scripts/import_delan_data.py` 文件头注释。

### 2.3 训练前检查数据

```bash
python examples/delan_train.py --inspect --data data/robot.pickle
```

---

## 3. 训练

以下命令均在**项目根目录**执行，且已 `conda activate frictionest`。

### 3.1 仅 DeLaN（刚体 L-Net）

适用于任意自由度机械臂；多关节、力矩量级差大时建议 **`--preset auto --tau-loss smape`**。

```bash
# 6 轴 robot 数据（推荐）
python examples/delan_train.py \
  --data data/robot.pickle \
  --preset auto \
  --tau-loss smape \
  --test-labels e v q \
  -m 1 \
  --plot

# 2-DoF 官方风格小网络
python examples/delan_train.py \
  --data /path/to/character_data.pickle.BAK \
  --preset delan_model \
  -m 1 \
  --plot
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--data` | `character_data.pickle` 路径 |
| `--preset` | `delan_model` / `example` / `auto` |
| `--tau-loss` | `mse` 或 `smape`（多轴推荐 smape） |
| `--test-labels` | 测试轨迹标签，默认 `e q v` |
| `--test-frac 0.2` | 无匹配标签时，最后 20% 轨迹作测试 |
| `-m 1` | 保存到 `checkpoints/delan_lnet.pt` |
| `--plot` | 训练后保存 `figures/delan_performance.png` |
| `-c 1` | 使用 CUDA（默认） |
| `-l 1` | 仅加载权重，不训练 |
| `--load path` | 指定 checkpoint |
| `--no-energy-loss` | 默认带能量项；加此开关则**仅**用力矩损失 \(l_\tau\) |

**损失（仅 DeLaN）**：默认 \(\mathcal{L} = l_\tau + l_E\)。\(l_\tau\) 为 `mse` 或 `smape`；\(l_E\) 为刚体功率守恒  
\(\mathbb{E}[(\mathrm{d}T/\mathrm{d}t + \mathrm{d}V/\mathrm{d}t - \tau^\top \dot q)^2]\)（由 `LNet.dynamics` 的 `dTdt`、`dVdt` 计算）。实现见 `RobotDynamics/DeLaN/train_core.py`。

权重默认路径：`checkpoints/delan_lnet.pt`。

### 3.2 DeLaN + 摩擦（MystericNet）

在 pickle 上同时训练 **L-Net** 与 **摩擦子网络**（`τ = τ_rigid + τ_fri`）。

```bash
python examples/robot_train.py \
  --data data/robot.pickle \
  --friction-backend stribeck_pinn \
  --lambda-physics 0.5 \
  --tau-loss smape \
  --test-labels e v q \
  --epochs 500 \
  -m 1
```

摩擦后端：

| `--friction-backend` | 说明 |
|----------------------|------|
| `tcn` | 时序卷积（原 Mysteric-Net 论文） |
| `fo_cascade` | TCN₁([q,q̇])→MLP→**1/s**→TCN₂，对齐 Xun 分数阶摩擦图 4（MLP 后因果积分低通） |
| `fo_cascade_pinn` | fo_cascade + SCV，`friction_pinn_loss`（Hu 等 PINN，\(\lambda\)=`--lambda-physics`） |
| `stribeck` | 可学习 Stribeck-Coulomb-Viscous 物理模型 |
| `stribeck_pinn` | MLP + SCV 物理损失（Hu 等 PINN） |

**损失（Mysteric-Net，默认不含能量项）**：

\[
\mathcal{L} = l_\tau + w_{\text{fri}}\, l_{\text{fri}} \;+\; [\;l_E\;]
\quad\text{（默认 } w_{\text{fri}}=1.0\text{）}
\]

| 项 | 何时启用 | 含义 |
|----|----------|------|
| \(l_\tau\) | 始终 | `--tau-loss smape`（默认）或 `mse` |
| \(l_{\text{fri}}\) | 有 `m/c/g` 或 PINN 后端 | **`--fri-loss smape`（默认）** 或 `mse`；与 \(l_\tau\) 相同 SMAPE 公式，利于小力矩关节。PINN：数据项 + SCV 项均用 `fri_loss` |
| \(w_{\text{fri}}\) | `--friction-loss-weight` | SMAPE 摩擦下默认 **1.0**；仅当 `fri-loss mse` 且 \(l_{\text{fri}}\) 很大时用 **0.01~0.1** |
| \(l_E\) | **仅**加 `--energy-loss` | Yeo 等 Eq. (7) 刚体能量率一致性，实现于 `RobotDynamics/FrictionModule/energy_loss.py` |

能量项（可选）：

\[
l_E = \mathbb{E}\left[\left(\frac{\mathrm{d}T}{\mathrm{d}t}+\frac{\mathrm{d}V}{\mathrm{d}t} - (\tau-\hat\tau_{\text{fri}})^\top \dot q\right)^2\right]
\]

其中 \(\mathrm{d}T/\mathrm{d}t+\mathrm{d}V/\mathrm{d}t\) 由 **L-Net** 的 `dynamics()` 给出；\((\tau-\hat\tau_{\text{fri}})^\top\dot q\) 把摩擦从总力矩中剥离后的功率残差。开启 `--energy-loss` 时训练更慢（需 L-Net 能量率前向），合成数据上同样可用：

```bash
python examples/robot_train.py \
  --data data/robot.pickle \
  --friction-backend fo_cascade_pinn \
  --lambda-physics 0.5 \
  --tau-loss smape \
  --energy-loss \
  -m 1 --friction-label none --epochs 1000

python examples/synthetic_train.py --data data/synthetic_2dof_inverse.npz --energy-loss -m 1
```

默认保存：`checkpoints/mysteric_robot.pt`（需 `-m 1`）。**Ctrl+C 中断训练**时会自动保存当前权重：若已加 `-m 1` 则写入 `--save` 路径，否则写入 `checkpoints/mysteric_robot_interrupt.pt`（checkpoint 中含 `epoch`、`interrupted` 字段）。

### 3.3 2-DoF 合成数据（快速冒烟）

```bash
python scripts/generate_dataset.py
python examples/synthetic_train.py --data data/synthetic_2dof_inverse.npz -m 1

# 带 Stribeck PINN 摩擦
python examples/synthetic_train.py \
  --data data/synthetic_2dof_inverse.npz \
  --friction-backend stribeck_pinn \
  -m 1
```

---

## 4. 测试与评估

### 4.1 DeLaN 测试集评估

```bash
python examples/delan_evaluate.py \
  --checkpoint checkpoints/delan_lnet.pt \
  --data data/robot.pickle \
  --test-labels e v q
```

- 默认保存图：`figures/delan_performance.png`
- 弹窗查看：`--show`
- 不保存图：`--no-plot`
- 自定义图路径：`--figure-out figures/my_eval.png`

也可加载官方 2-DoF 权重（若路径可用）：

```bash
python examples/delan_evaluate.py \
  --checkpoint /path/to/delan_model.torch \
  --data /path/to/character_data.pickle.BAK
```

终端会打印 **Torque / Inertial / Coriolis / Gravity / Power** 等 MSE；`n_dof > 2` 时评估图为每个关节一行、四列（τ, m, c, g）。

### 4.2 Mysteric-Net 评估（关节摩擦网络 + 总力矩曲线图）

**不要用** `delan_evaluate.py`（那只画 DeLaN 刚体四列图，默认 `figures/delan_performance.png`）。

在项目根目录、已 `conda activate frictionest` 时，生成 **各关节摩擦与总力矩对比图**：

```bash
cd /path/to/FrictionEst

python examples/robot_evaluate.py \
  --checkpoint checkpoints/mysteric_robot.pt \
  --data data/robot.pickle \
  --test-labels e v q \
  --figure-out figures/robot_friction.png
```

- **默认输出图**：`figures/robot_friction.png`（省略 `--figure-out` 时同上；含 \(\tau_{\text{fri}}\) 预测、参考分解、PINN 时还有 SCV 列）
- **自定义路径**：`--figure-out figures/my_robot_friction.png`
- **弹窗查看**：`--figure-out` 可省略，加 `--show` 不保存只显示
- **`--seq-len`**：一般省略，从 checkpoint 自动读取（与训练一致，如 30）
- 终端会打印 `RMSE τ_hat`、`RMSE τ_fri`（有 m/c/g 分解时）及 SCV 参数表（`fo_cascade_pinn` / `stribeck_pinn`）

| checkpoint 来源 | 评估脚本 | 默认图片 |
|-----------------|----------|----------|
| `delan_train.py` → `delan_lnet.pt` | `examples/delan_evaluate.py` | `figures/delan_performance.png` |
| `robot_train.py` → `mysteric_robot.pt` | `examples/robot_evaluate.py` | **`figures/robot_friction.png`** |

合成 2-DoF `.npz` 测试（无绘图或仅数值）可用：

```bash
python scripts/evaluate_model.py \
  --model checkpoints/RobotDynamics.pt \
  --test-data data/test_synthetic_2dof_inverse.npz
```

（需先用 `scripts/generate_test_dataset.py` 生成测试 npz。）

### 4.3 实现验证（可选）

```bash
python scripts/verify_delan_impl.py
```

---

## 5. 单元测试

```bash
cd /path/to/DeLaN_Stribeck
pytest tests/ -q
```

若环境中 ROS 等插件干扰 pytest，可单独运行：

```bash
python -c "from tests.test_stribeck import test_scv_zero_velocity_smooth; test_scv_zero_velocity_smooth(); print('ok')"
```

---

## 6. Python API 速查

```python
# DeLaN
from RobotDynamics.DeLaN import LNet, load_dataset, train_delan_loop, suggest_hyper

# 摩擦
from RobotDynamics.FrictionModule import HNetStribeckPINN, friction_pinn_loss, scv_torque

# 联合模型
from RobotDynamics.MystericNet import MystericNet

# 数据导入
from RobotDynamics.DeLaN import import_mat, save_pickle, inspect_dataset
```

---

## 7. 常见问题

**Q: 训练 loss 很低但图上部分关节很差？**  
多关节力矩尺度差时，请用 `--tau-loss smape`；以测试集 **RMSE_test** 和 `delan_evaluate.py` 为准。

**Q: `robot.pickle` 上摩擦网络学不好？**  
若数据中 `m+c+g ≈ τ`（仿真刚体、几乎无摩擦残差），摩擦子网只能学到接近零；需含真实摩擦或 `τ_fri = τ - τ_rigid` 的实测数据。

**Q: checkpoint 与数据 DoF 不一致？**  
加载时会报错；请用与数据相同 `n_dof` 重新训练。

**Q: 找不到 `mysteric_net`？**  
包已重命名为 **`RobotDynamics`**，请更新 import 与 IDE 工作区根目录。

---

## 参考

- DeLaN / L-Net：Lutter et al., 拉格朗日神经网络  
- Mysteric-Net / TCN 摩擦：Yeo et al.  
- Stribeck PINN 摩擦：Hu et al., *Physics-Informed Learning for the Friction Modeling*（SCV 模型 Eq. (3)(4)，PINN 损失 Eq. (6)）

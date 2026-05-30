# DeLaN_Stribeck / RobotDynamics

工业机械臂动力学学习：**DeLaN**（拉格朗日刚体）+ **摩擦模块**（TCN / FO 级联 / Stribeck SCV / PINN），Python 包名为 `RobotDynamics`。

多轴迟滞摩擦推荐 **`fo_cascade_pinn`**（文档与对比图中亦称 **LA-FC-PINN**：TCN₁→MLP→TCN₂ + SCV 物理约束）。**main 分支已移除 GMS 摩擦后端**；旧 `gms` / `gms_pinn` checkpoint 无法加载，需用 PINN 后端重新训练。

## 项目结构

```text
DeLaN_Stribeck/
├── RobotDynamics/              # 主代码包
│   ├── DeLaN/                  # L-Net、数据加载、训练与评估
│   ├── FrictionModule/         # TCN、FO 级联、Stribeck SCV、损失与合成数据
│   └── MystericNet/            # DeLaN + 摩擦联合模型
├── examples/
│   ├── delan_train.py          # 仅训练 DeLaN（刚体）
│   ├── delan_evaluate.py       # DeLaN 测试与绘图
│   ├── robot_train.py          # L-Net + 摩擦（分阶段 / warmup / 续训）
│   ├── robot_evaluate.py       # 单模型：摩擦 + 总力矩曲线
│   ├── robot_compare_evaluate.py   # 多模型同图对比（τ / τ_fri + RMSE）
│   ├── robot_compare_metrics.py    # 多模型测试集 RMSE 表 / CSV
│   ├── robot_compare_common.py     # 上述对比脚本共享逻辑
│   ├── motor_identify_train.py # 单电机惯量 J + fo_cascade_pinn
│   └── synthetic_train.py      # 2-DoF 合成数据冒烟训练
├── scripts/
│   ├── import_delan_data.py    # .mat / .npz → .pickle
│   ├── generate_dataset.py     # 2-DoF 合成 .npz
│   └── verify_delan_impl.py    # DeLaN 实现核对
├── tests/                      # pytest 单元测试
├── data/                       # 数据与导入结果（如 robot_fric.pickle）
├── checkpoints/                # 模型权重与 loss CSV
├── figures/                    # 评估与对比图
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

dt = 0.001;   % 采样周期 (s)，导入时用于低通滤波 fs=1/dt（§2.3）
save('robot.mat', 'labels','dt','t','qp','qv','qa','tau','m','c','g','p','pdot', '-v7');
```

**单条轨迹**：`qp, qv, ...` 为 `T×n_dof` 数值矩阵（单轴可为 `T×1` 或长度 `T` 的向量），`labels` 为单个标签。

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

### 2.3 导入时低通滤波（默认开启）

`import_delan_data.py` 在写入 pickle **之前**，对运动/力矩序列做 **4 阶 Butterworth 低通 + `filtfilt`（零相位）**。

| 项 | 说明 |
|----|------|
| 默认截止频率 | **200 Hz**（`--filter-cutoff`） |
| 采样率 | $f_s = 1/d_t$：优先 `.mat` 内标量 **`dt`**，否则各轨迹 `mean(diff(t))` |
| 默认滤波字段 | `qp`, `qv`, `qa`, `tau`, `p`, `pdot`（**不**滤 `m/c/g`） |
| 约束 | 需 $f_c < f_s/2$（Nyquist），否则脚本报错 |

```bash
# 默认 fc=200 Hz（电机数据 dt=0.001 → fs=1000 Hz 时合法）
python scripts/import_delan_data.py \
  -i data/motor_character_data.mat \
  -o data/motor_data.pickle

# 自定义截止频率
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle \
  --filter-cutoff 100

# 关闭滤波
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle \
  --no-filter

# 显式指定采样周期（覆盖 .mat 内 dt）
python scripts/import_delan_data.py \
  -i traj.npz \
  -o data/out.pickle \
  --dt-hint 0.001 \
  --filter-cutoff 150

# 只滤部分字段
python scripts/import_delan_data.py \
  -i data/robot.mat \
  -o data/robot.pickle \
  --filter-keys qp qv tau
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--filter-cutoff` | `200` | 截止频率 (Hz) |
| `--no-filter` | 关 | 不做低通 |
| `--filter-order` | `4` | Butterworth 阶数 |
| `--dt-hint` | 无 | 采样周期 (s)，覆盖 `.mat` 的 `dt` |
| `--filter-keys` | 见上表 | 要滤波的变量名列表 |

实现见 `RobotDynamics/DeLaN/signal_filter.py`。单轴 `.mat` 中 `qp` 等为 `(T,)` 一维向量时，导入脚本会自动 reshape 为 `(T, 1)`（见 `mat_convert.py`）。

### 2.4 导入后数据检查图（默认开启）

`import_delan_data.py` 写入 pickle 后，会按 **每条轨迹标签** 生成检查图（`RobotDynamics/DeLaN/import_plot.py`）：

| 输出 | 内容 |
|------|------|
| `figures/.../import_<label>.png` | 各标签一张：**n_dof=1** 为单列时间序列；**n_dof>1**（如 6 轴 `robot_fric`）为 **行×关节列** 网格，每关节对比 `qa` 与 `d(qv)/dt` |
| `figures/.../import_all_tau.png` | 总览：单轴为各标签 `tau` 拼接；多轴为 **每关节一张** `\|tau\|` |
| 终端统计 | 每条轨迹、**每个关节** 的 `\|qv\|max`、`\|qa\|max`、qa/梯度之比 |

多轴可只画部分关节：`--plot-joints 0 1 2`（默认画全部）。

```bash
python scripts/import_delan_data.py \
  -i data/motor_character_data.mat \
  -o data/motor_data.pickle \
  --filter-cutoff 40 \
  --figure-dir figures/motor_import_check

# 不绘图：加 --no-plot
# 仅检查已有 pickle：--inspect data/motor_data.pickle --plot
# 6 轴机械臂（mat 损坏时可直接 inspect pickle）：
python scripts/import_delan_data.py --inspect data/robot_fric.pickle --plot \
  --figure-dir figures/robot_import_check
```

看图时重点确认：**换向处 `qa` 与 `d(qv)/dt` 是否同量级**（应接近 1）；`tau` 尖峰处 `qa` 是否也有峰（否则 L-Net 学不到惯量项）。

### 2.5 训练前检查数据

```bash
python examples/delan_train.py --inspect --data data/robot.pickle
```

### 2.6 示例数据 `robot_fric` / `robot_fric1`

本仓库机械臂摩擦实验常用 **`data/robot_fric.pickle`** 或 **`data/robot_fric1.pickle`**（若本地已放置）：

| 项 | 典型值 |
|----|--------|
| 自由度 | 6 |
| 轨迹数 | 20 |
| 每条长度 | 2048 点 |
| 采样 | 200 Hz（$dt=0.005\,\mathrm{s}$） |
| 分解 | 含 `m, c, g` → $\tau_{\text{fri,ref}}=\tau-m-c-g$ |
| 阻力约定 | $\tau_{\text{fri}}$ 与 $\dot q$ **反号**（阻力矩）；SCV / fo_cascade 实现已对齐 |

无摩擦分解标签时训练用 **`--friction-label none`**，靠总力矩 $l_\tau$ 与 PINN 物理项（`fo_cascade_pinn`）联合辨识摩擦。

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
| `--no-energy-loss` | 默认带能量项；加此开关则**仅**用力矩损失 $l_\tau$ |

**损失（仅 DeLaN）**：默认 $\mathcal{L} = l_\tau + l_E$。$l_\tau$ 为 `mse` 或 `smape`；$l_E$ 为刚体功率守恒  
$\mathbb{E}[(\mathrm{d}T/\mathrm{d}t + \mathrm{d}V/\mathrm{d}t - \tau^\top \dot q)^2]$（由 `LNet.dynamics` 的 `dTdt`、`dVdt` 计算）。实现见 `RobotDynamics/DeLaN/train_core.py`。

权重默认路径：`checkpoints/delan_lnet.pt`。

### 3.2 DeLaN + 摩擦（MystericNet）

在 pickle 上同时训练 **L-Net** 与 **摩擦子网络**（`τ = τ_rigid + τ_fri`）。

**推荐（6 轴、无 τ_fri 标签、迟滞摩擦）**：

```bash
python examples/robot_train.py \
  --data data/robot_fric1.pickle \
  --friction-backend fo_cascade_pinn \
  --friction-label none \
  --lambda-physics 0.3 \
  --lr 1e-3 --lnet-lr 1e-4 \
  --scv-lr-mult 5 --warmup-lr 5e-4 \
  --friction-warmup-epochs 200 \
  --stage1-epochs 6000 \
  --stage2-epochs 4000 --stage2-lr 1e-3 \
  --stage3-epochs 4000 --stage3-lr 1e-3 \
  --energy-loss --energy-loss-weight 0.01 \
  --test-labels e v q
```

简单冒烟（默认后端 `stribeck_pinn`）：

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
| `fo_cascade` | TCN₁([q,q̇])→**两层 tanh MLP**→TCN₂（Xun 图 4 简化版） |
| **`fo_cascade_pinn`** | **推荐**：fo_cascade + SCV PINN（Hu Eq. (6)）；对比图图例显示 **LA-FC-PINN**；TCN 默认 **3 层** |
| `stribeck` | 可学习 SCV 物理模型（**仅** $\dot q$，无 MLP/TCN）；见 **§3.2.1** |
| `stribeck_pinn` | MLP + SCV 物理损失 |

> **已移除**：`gms` / `gms_pinn`（Generalized Maxwell-Slip）不在 main 分支；历史 checkpoint 续训/评估会报错。

**损失（Mysteric-Net，默认不含能量项）**：

$$
\mathcal{L} = l_\tau + w_{\text{fri}}\, l_{\text{fri}} \;+\; [\;l_E\;]
\quad\text{（默认 } w_{\text{fri}}=1.0\text{）}
$$

| 项 | 何时启用 | 含义 |
|----|----------|------|
| $l_\tau$ | 始终 | `--tau-loss smape`（默认）或 `mse` |
| $l_{\text{fri}}$ | 有 `m/c/g` 或 PINN 后端 | **`--fri-loss smape`（默认）** 或 `mse`；与 $l_\tau$ 相同 SMAPE 公式，利于小力矩关节。PINN：数据项 + SCV 项均用 `fri_loss`。**纯 `stribeck` 后端见 §3.2.1** |
| $w_{\text{fri}}$ | `--friction-loss-weight` | SMAPE 摩擦下默认 **1.0**；仅当 `fri-loss mse` 且 $l_{\text{fri}}$ 很大时用 **0.01~0.1** |
| $l_E$ | `--energy-loss` | Yeo Eq. (7)；权重 `--energy-loss-weight`（默认 1.0） |

**阻力符号（结构约束，非 loss）**：$\tau_{\text{fri}}$ 为阻力矩，与 $\dot q$ 反号；`scv_torque` / fo_cascade 的 `_resistive_torque` 已取负，与 $\tau_{\text{fri,ref}}=\tau-m-c-g$ 一致。`fo_cascade` / `fo_cascade_pinn` 的数据支路为 $\hat\tau=\mathrm{softplus}(z)\cdot\tanh(\beta\dot q)$；SCV 侧设 $k_s=k_c+\mathrm{softplus}(\Delta k_s)\ge k_c$。

能量项（可选）：

$$
l_E = \mathbb{E}\left[\left(\frac{\mathrm{d}T}{\mathrm{d}t}+\frac{\mathrm{d}V}{\mathrm{d}t} - (\tau-\hat\tau_{\text{fri}})^\top \dot q\right)^2\right]
$$

其中 $\mathrm{d}T/\mathrm{d}t+\mathrm{d}V/\mathrm{d}t$ 由 **L-Net** 的 `dynamics()` 给出；$(\tau-\hat\tau_{\text{fri}})^\top\dot q$ 把摩擦从总力矩中剥离后的功率。需显式加 `--energy-loss` 才会计入总损失。

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

**Checkpoint 与训练控制**（`robot_train.py`）：

| 文件 | 说明 |
|------|------|
| `checkpoints/{backend}_net.pt` | 训练正常结束时的最终权重（`--save` 可改；**无需** `-m` 也会保存） |
| `checkpoints/{backend}_net_interrupt.pt` | **Ctrl+C** 中断时保存（含 `epoch`、`interrupted`） |
| `checkpoints/{backend}_net_epochNNNNN.pt` | 周期性快照（默认每 **500** epoch，`--checkpoint-save-interval`；`--no-periodic-checkpoint` 关闭） |
| `checkpoints/{backend}_loss.csv` | 损失日志（默认每 **50** epoch 一行：`epoch, phase, loss, l_tau, l_fri, l_E`；`--loss-log-epoch-interval`） |

其中 `{backend}` 为 `--friction-backend`（如 `stribeck`、`fo_cascade_pinn`）。`-m` 仅保留兼容，行为与上表一致。

**分阶段训练（S0-W / S1 / S2-H / S3-L）**（推荐用于无摩擦分解标签、长训机械臂）：

| 阶段 | 标记 | 可训练模块 | 损失要点 |
|------|------|------------|----------|
| **0 warmup** | `[S0-W]` | **仅 hnet**（L-Net 冻结） | $l_\tau + w_{\text{fri}}\,l_{\text{fri}}$；SCV lr × `--scv-lr-mult` |
| **1 联合** | `[S1]` | L-Net + hnet | 同上；`--lr`→hnet，`--lnet-lr`→L-Net |
| **2 摩擦** | `[S2-H]` | **仅 hnet**（L-Net 冻结） | $l_\tau + w_{\text{fri}}\,l_{\text{fri}}$；$\lambda_{\text{phys}}$ 默认 ×0.5 |
| **3 刚体** | `[S3-L]` | **仅 L-Net**（hnet 冻结） | $l_\tau$（$\hat\tau_{\text{core}}$ 拟合 $\tau-\hat\tau_{\text{fri}}$） |

轮数由 **`--friction-warmup-epochs`**、**`--stage1-epochs`**、**`--stage2-epochs`**、**`--stage3-epochs`** 划分；未设 `stage1` 时，S1 = `epochs − warmup − stage2 − stage3`。

- **S0-W**：`--friction-warmup-epochs`（例 200）；`--warmup-lr` 可低于 `--lr`；与 `--friction-only` 互斥。  
- **S2-H**：进入时冻结 L-Net；`--stage2-lr`；`--stage2-lambda-physics-mult`（默认 0.5）降低 PINN 物理项，减轻 SCV 与 fo 抢梯度。  
- **S3-L**：进入时冻结 hnet；`--stage3-lr`；让 L-Net 在固定摩擦下拟合 $\tau_{\text{meas}}-\hat\tau_{\text{fri}}$。  
- **`--energy-loss`**：warmup **[S0-W] 仅监控** $l_E$，从 S1 起计入 loss。

```bash
# 6 轴：warmup → 联合 → 专训摩擦 → 专训刚体
python examples/robot_train.py \
  --data data/robot_fric.pickle \
  --friction-backend fo_cascade_pinn \
  --friction-label none \
  --friction-warmup-epochs 200 \
  --stage1-epochs 2000 \
  --stage2-epochs 1000 --stage2-lr 1e-3 \
  --stage3-epochs 500 --stage3-lr 5e-4 \
  --lr 1e-3 --lnet-lr 1e-4 \
  --test-labels q
```

**从 checkpoint 续训**（`--resume`）：

- 加载已有 `state_dict` 与结构；从 **checkpoint 内 `epoch` + 1** 继续（若无 `epoch` 字段，则从文件名 `*_epoch01500.pt` 推断，下一 epoch 为 1501）。  
- **`--epochs` 表示训练总目标轮数**（不是「再训多少轮」）。例：checkpoint 停在 1500、`--epochs 3000` → 训练 1501…3000。  
- 续训起点若已超过阶段边界，会自动进入对应阶段（S2-H 冻结 L-Net；S3-L 冻结 hnet）。  
- **`--friction-backend` 应与保存时一致**；**GMS 旧 checkpoint 不可用**。数据 `n_dof` 须与 checkpoint 一致。  
- 续训**不**恢复 optimizer 状态；`loss CSV` 会按本次运行重新写入（可先备份旧 CSV）。

```bash
# 从最终权重续训到 12000 epoch（fo_cascade_pinn）
python examples/robot_train.py \
  --data data/robot_fric1.pickle \
  --friction-backend fo_cascade_pinn \
  --friction-label none \
  --resume checkpoints/fo_cascade_pinn_net.pt \
  --epochs 12000 \
  --friction-warmup-epochs 200 \
  --stage1-epochs 6000 \
  --stage2-epochs 4000 \
  --stage3-epochs 2000 \
  --test-labels q

# 从周期快照续训（已完成 500 epoch → 从 501 开始）
python examples/robot_train.py \
  --data data/robot_fric1.pickle \
  --friction-backend fo_cascade_pinn \
  --friction-label none \
  --resume checkpoints/fo_cascade_pinn_net_epoch00500.pt \
  --epochs 12000 \
  --friction-warmup-epochs 200 \
  --stage1-epochs 6000 \
  --stage2-epochs 4000 \
  --stage3-epochs 2000 \
  --test-labels q
```

**常用参数（`robot_train.py`）**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `data/robot.pickle` | DeLaN pickle |
| `--friction-backend` | `stribeck_pinn` | 见上表；多轴迟滞推荐 `fo_cascade_pinn` |
| `--friction-label` | `auto` | `auto`/`none`/`decomposition`：无 m/c/g 时用 `none` |
| `--epochs` | `500` | 总训练 epoch（含 warmup + S1/S2/S3；续训为终点） |
| `--friction-warmup-epochs` | `0` | S0-W：仅训 hnet |
| `--warmup-lr` | 同 `--lr` | S0-W 学习率 |
| `--stage1-epochs` / `--stage2-epochs` / `--stage3-epochs` | 无 / `0` / `0` | S1 / S2-H / S3-L |
| `--stage2-lr` / `--stage3-lr` | 同 `--lr` | S2-H / S3-L 学习率 |
| `--stage2-lambda-physics-mult` | `0.5` | S2-H 的 $\lambda = \lambda_{\text{S1}} \times$ mult |
| `--lr` | `1e-3` | S1 **hnet** 学习率 |
| `--lnet-lr` | 同 `--lr` | S1 **L-Net** 学习率（例 `1e-4` 减轻抢残差） |
| `--scv-lr-mult` | `10` | warmup / S2-H 中 SCV 参数 lr 倍数 |
| `--grad-clip` | `1.0` | 梯度裁剪（`0`=关） |
| `--pinn-loss-mode` | `auto` | `auto`/`hu`/`tau_blend`；无 τ_fri 标签时 `auto`→`tau_blend` |
| `--lambda-physics` | `0.5` | PINN 物理项 $\lambda$（S1） |
| `--resume` | 无 | 从 checkpoint 续训 |
| `--save` | `{backend}_net.pt` | 最终权重路径 |
| `--checkpoint-save-interval` | `500` | 周期 checkpoint；`0` 关闭 |
| `--energy-loss` | 关 | 可选 $l_E$（warmup 不计入 loss） |
| `--test-labels` | `e v q` | 测试轨迹标签 |

不设 `--stage2-epochs` 且不设 `--stage3-epochs`（或都为 `0`）时为 **单阶段联合训练**（L-Net + hnet 同时更新）。

**联合阶段分学习率**（减轻 L-Net 被摩擦带偏，属启发式）：本仓库中 **摩擦 = `hnet`**，**刚体 = `lnet`**。阶段 1 可令 hnet 学得更快、L-Net 更慢：

```bash
python examples/robot_train.py \
  --data data/robot_fric.pickle \
  --friction-backend stribeck \
  --friction-label none \
  --lr 1e-3 \
  --lnet-lr 1e-4 \
  --epochs 3000 \
  --stage2-epochs 1000 \
  --test-labels q
```

仍建议配合 **S2-H / S3-L**；纯 `stribeck` 且 $l_{\text{fri}}=0$ 时，分 lr 只能缓解，不能替代专训摩擦阶段。

#### 3.2.1 纯 `stribeck` 后端与 `l_fri` 不降

`--friction-backend stribeck` 时，摩擦子网络 **只有 SCV 六个标量/关节**（$k_v,k_c,k_a,k_s,v_s,\alpha$），输出 $\tau_{\text{fri}}=\mathrm{SCV}(\dot q_t)$，**不含** MLP/TCN，也 **不使用** `--lambda-physics`（该参数仅对 `stribeck_pinn` / `fo_cascade_pinn` 有效）。

**常见现象**：`l_fri ≈ 1.996` 长期不变，而 `l_tau` 在下降。

| 原因 | 说明 |
|------|------|
| SMAPE 饱和 | 默认 SCV 初值偏小、$\hat\tau_{\text{fri}}\ll\tau_{\text{fri,ref}}$ 时 SMAPE **≈2 且梯度≈0** |
| 无摩擦标签 | `--friction-label none` 时 $l_{\text{fri}}=0$，仅靠 $l_\tau$ 更新 SCV |
| 模型能力 | 瞬时 SCV($\dot q$) 难以拟合迟滞/历史摩擦 |

**内置处理**：有 `m/c/g` 分解时 **warm-start** $k_c,k_s$；`--fri-loss smape` 且后端为 `stribeck` 时自动改用 **`fri_loss=mse`**。

**推荐**：多轴迟滞摩擦优先 **`fo_cascade_pinn`（LA-FC-PINN）**；瞬时 SCV 模型可用 `stribeck_pinn`。纯 `stribeck` 请显式 **`--fri-loss mse`**。

```bash
python examples/robot_train.py \
  --data data/robot_fric.pickle \
  --friction-backend stribeck \
  --fri-loss mse \
  --epochs 2000 \
  --test-labels e v q \
  -m 1
```

摩擦损失细节（PINN、`tau_blend`、SCV 参数化）见 [`RobotDynamics/FrictionModule/readme.md`](RobotDynamics/FrictionModule/readme.md)。

### 3.3 单电机惯量 + 摩擦辨识（`motor_identify_train.py`）

适用于 **单轴伺服 / 电机**（`n_dof=1`）：用 **DeLaN** 学等效惯量 $H \approx J$，用 **`fo_cascade_pinn`** 学摩擦；固定不监督摩擦标签（无 m/c/g 分解时与 `robot_train.py --friction-label none` + PINN 等价，但脚本已封装默认超参与辨识报告）。

**物理假设**（水平安装或重力已补偿）：

$$
\tau = J\,\ddot q + \tau_{\text{fri}}, \quad c \approx 0,\; g \approx 0
$$

竖直轴且未重力补偿时，网络可能学到非零 $g(q)$，惯量仍在 $H$ 中，需结合残余 `g_rms` 判断。

#### 数据准备

每条轨迹需 **`qp, qv, qa, tau`**，形状 **`(T, 1)`**（时间在行上）。单位建议统一为 rad、rad/s、rad/s²、N·m。

| 格式 | 说明 |
|------|------|
| `.pickle` | DeLaN 官方多轨迹格式（见 §2） |
| `.npz` | 扁平键 `qp, qv, qa, tau`（可选 `t`）；导入逻辑同 `import_delan_data.py` |

```bash
# MATLAB / 采集 → .mat 时，先转为 pickle（单轴列数为 1；默认 200 Hz 低通）
python scripts/import_delan_data.py \
  -i data/motor_character_data.mat \
  -o data/motor_data.pickle
# 不需要滤波时加 --no-filter；改截止频率用 --filter-cutoff（见 §2.3）；检查图见 §2.4

# 检查 n_dof=1
python examples/motor_identify_train.py --data data/motor.pickle --inspect
```

**激励**：轨迹应覆盖足够大的 $|\ddot q|$ 与 $|\dot q|$（正反转、加减速、扫频等），否则 $J$ 与摩擦难以分离。

#### 训练（单阶段）

联合训练 L-Net（$J$）与 `fo_cascade_pinn`（摩擦）。单轴默认刚体项为 $\tau_{\text{rigid}} = H\ddot q$（`c=g=0`，见脚本内 `zero_cg`）。

```bash
conda activate frictionest
cd /path/to/DeLaN_Stribeck

python examples/motor_identify_train.py \
  --data data/motor_data.pickle \
  --known-J 0.243 \
  --test-labels e v q \
  --epochs 800 \
  -m 1
```

#### 两阶段训练（推荐：先摩擦、再惯量）

单阶段长训时，摩擦与惯量**共用** $l_\tau$，梯度常互相牵制：摩擦曲线尚可，但 **$J$ 偏离台架值**（如 `--known-J 0.243` 即 `24.3e-2` kg·m²）。  
`motor_identify_train.py` 支持：

| 阶段 | 轮数 | 可训练模块 | 损失 | 日志标记 |
|------|------|------------|------|----------|
| **1** | `--stage1-epochs` | L-Net + hnet（fo + SCV） | $l_\tau + w_{\text{fri}}\,l_{\text{fri}}$ | `[S1]` |
| **2** | `--stage2-epochs` | **仅 L-Net**（**冻结 hnet**） | $l_\tau + w_{\text{inertia}}\,l_{\text{inertia}}$（$\hat\tau_{\text{core}}$ 拟合 $\tau-\hat\tau_{\text{fri}}$） | `[S2-J]` |

- 设 `--stage2-epochs 0`（默认）则等价于只用 `--epochs` 的单阶段训练。  
- 未指定 `--stage1-epochs` 时，阶段 1 轮数 = `epochs − stage2`（例如总 2000、阶段 2 为 800 → 阶段 1 为 1200）。  
- 进入阶段 2 时终端会打印「已冻结 hnet」，并可用更小的 `--stage2-lr`（默认与 `--lr` 相同，常取 `1e-4`）。

**示例（总 2000 epoch：1200 + 800）**：

```bash
python examples/motor_identify_train.py \
  --data data/motor_data.pickle \
  --known-J 0.243 \
  --test-labels e v q \
  --stage1-epochs 1200 \
  --stage2-epochs 800 \
  --stage2-lr 1e-4 \
  --lnet-mass-eps 1e-3 \
  -m 1
```

**调参提示**：

- 阶段 1 结束先看 `figures/motor_friction.png`（§下方评估），确认摩擦形状合理再依赖阶段 2 的 $J$。  
- 日志里 **`J_med` 长期等于 `--lnet-mass-eps`**：多半不是“没训练”，而是 $L\approx 0$、$H\approx\varepsilon I$（`net_ld` 经 ReLU 后输出≈0）。请配合 **`--known-J`**（自动把 `net_ld.bias` 初化为 $\sqrt{J-\varepsilon}$）、看 **`J_learn`/`L_med`**（$J_{\text{learn}}\approx H_{00}-\varepsilon$），并保证 $\varepsilon\ll J$（如 `1e-4`，勿把 `1e-3` 当成目标惯量）。  
- 阶段 2 默认 **`--stage2-w-inertia 1`**，显式让 $\hat\tau_{\text{core}}=H\ddot q$ 去拟合尖峰惯量项；仅用总力矩 $l_\tau$ 时摩擦已冻结，平台段梯度很弱。  
- 数据需有足够 **$|\ddot q|$** 激励，否则阶段 2 仍难以辨识惯量。

#### 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--data` | `data/motor.pickle` | `.pickle` 或 `.npz` |
| `--known-J` | 无 | 已知惯量 (kg·m²)；默认初始化 `net_ld.bias`（`--no-lnet-j-init` 关闭） |
| `--stage2-w-inertia` | `1` | 阶段 2 惯量项 $l_{\text{inertia}}$ 权重；`0`=仅 $l_\tau$ |
| `--test-frac` | `0.2` | 无匹配 `--test-labels` 时，最后 20% **轨迹**作测试 |
| `--test-labels` | `e v q` | 测试集轨迹标签 |
| `--epochs` | `800` | 单阶段总 epoch；两阶段时作 stage1 默认上限 |
| `--stage1-epochs` | 无 | 阶段 1 epoch；与 `--stage2-epochs` 联用 |
| `--stage2-epochs` | `0` | 阶段 2 epoch；`0`=关闭两阶段 |
| `--stage2-lr` | 同 `--lr` | 阶段 2 学习率（默认 `5e-4`） |
| `--lr` | `5e-4` | 阶段 1 学习率 |
| `--seq-len` | `20` | 摩擦网络滑窗长度 |
| `--fo-mlp-hidden` | 自动 | fo_cascade 两层 MLP 隐层宽度，默认 `max(4*n_dof, 16)` |
| `--fo-tcn-layers` | 自动 | TCN₁/TCN₂ 层数；`fo_cascade`=2，`fo_cascade_pinn`=3 |
| `--lnet-width` / `--lnet-depth` | `32` / `2` | L-Net 规模 |
| `--lnet-mass-eps` | `1e-2` | $H$ 对角初值/数值脊 |
| `--lambda-physics` | `0.5` | PINN 摩擦物理项权重 $\lambda$ |
| `--friction-loss-weight` | `1.0` | $w_{\text{fri}}$ |
| `-m 1` | 关 | 保存 `checkpoints/motor_identify.pt` |
| `--save` | `checkpoints/motor_identify.pt` | 自定义路径 |

**损失**（与 §3.2 一致，细节见 [`RobotDynamics/FrictionModule/readme.md`](RobotDynamics/FrictionModule/readme.md)）：

$$
\mathcal{L} = l_\tau + w_{\text{fri}}\, l_{\text{fri}}, \quad
l_{\text{fri}} = \lambda\,\text{loss}(\hat\tau_{\text{fri}},\,\tau_{\text{fri,physics}})
$$

无 $\tau_{\text{fri}}$ 真值时仅 PINN 物理项 + 总力矩 $l_\tau$。阶段 2 另加 $l_{\text{inertia}}=\text{loss}(\hat\tau_{\text{core}},\,\tau-\hat\tau_{\text{fri}})$（摩擦分支已冻结）。训练中每 50 epoch 打印 `J_med`、`J_learn`、`L_med`、`RMSE_τ` 等；结束后输出辨识表（$H_{00}$ 及 $H_{00}-\varepsilon$、SCV 参数）。checkpoint 含 `motor_identify`、`J_est`、`stage1_epochs`、`stage2_epochs`。

#### 评估

训练结束后用 `robot_evaluate.py` 查看摩擦与总力矩（脚本会从 checkpoint 自动读取 `fo_mlp_hidden_dim`）：

```bash
python examples/robot_evaluate.py \
  --checkpoint checkpoints/motor_identify.pt \
  --data data/motor_data.pickle \
  --test-labels e v q \
  --figure-out figures/motor_friction.png
```

| 看什么 | 说明 |
|--------|------|
| 终端 **辨识表** | `J` 与 `--known-J` 相对误差、`c_rms`/`g_rms`（应小）、`RMSE τ_total` |
| `motor_friction.png` | 预测 $\tau_{\text{fri}}$、SCV 物理支路、$\tau_{\text{hat}}$ vs 测量 |
| checkpoint **`J_est`** | 与表中 $H_{00}$ 中位数一致 |

仅评估刚体（不看摩擦网络）时可用 `delan_evaluate.py` + 同一 checkpoint（只加载 L-Net）。

#### 注意

- 辨识的是**数据上的等效惯量**（含减速器反射惯量等），不是 datasheet 裸转子 $J$ 的唯一值。
- **$J$ 不准、摩擦尚可**时优先试 **§两阶段训练**，并检查 `--lnet-mass-eps` 与激励是否含足够加速度。
- `m/c/g` 全零的 pickle **不能**用于监督 $\tau_{\text{fri}}$；本脚本已固定 `supervise_friction=False`。
- 多轴电机请用 §3.2 `robot_train.py`，不要用本脚本。

### 3.4 2-DoF 合成数据（快速冒烟）

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

### 4.2 Mysteric-Net 单模型评估

**不要用** `delan_evaluate.py`（那只画 DeLaN 刚体四列图）。

```bash
python examples/robot_evaluate.py \
  --checkpoint checkpoints/fo_cascade_pinn_net.pt \
  --data data/robot_fric1.pickle \
  --test-labels q \
  --figure-out figures/robot_friction.png
```

- **默认输出图**：`figures/robot_friction.png`（$\tau_{\text{fri}}$ 预测、参考分解；PINN 后端含 SCV 列）
- **`--seq-len`**：一般省略，从 checkpoint 读取
- **`--checkpoint`** 须与 **`--friction-backend` 与 n_dof** 一致；**GMS 旧权重会报错**
- 终端打印 `RMSE τ_hat`、`RMSE τ_fri`（有 m/c/g 时）及 SCV 参数表

| checkpoint 来源 | 评估脚本 |
|-----------------|----------|
| `delan_train.py` → `delan_lnet.pt` | `delan_evaluate.py` |
| `robot_train.py` → `{backend}_net.pt` | `robot_evaluate.py` |
| `motor_identify_train.py` → `motor_identify.pt` | `robot_evaluate.py`（单轴） |

### 4.3 多模型对比评估（推荐）

在同一测试轨迹上叠加多个 checkpoint 的 $\tau$ / $\tau_{\text{fri}}$，便于对比不同训练阶段或后端：

**同图对比**（测量/参考为实线；模型为红/绿/蓝虚线；`fo_cascade_pinn` 显示为 **LA-FC-PINN**）：

```bash
python examples/robot_compare_evaluate.py \
  --data data/robot_fric1.pickle \
  --test-labels q \
  --checkpoint checkpoints/fo_cascade_pinn_net_epoch00500.pt:ep500 \
  --checkpoint checkpoints/fo_cascade_pinn_net.pt:final \
  --checkpoint checkpoints/stribeck_pinn_net.pt:stribeck \
  --figure-out figures/robot_compare.png
```

- `--checkpoint path:label`：`label` 用于图例；省略时用文件名
- 图内注释各模型 **RMSE τ** / **RMSE τ_fri**

**RMSE 统计表**（终端 + 可选 CSV）：

```bash
python examples/robot_compare_metrics.py \
  --data data/robot_fric1.pickle \
  --test-labels e v q \
  --checkpoint checkpoints/fo_cascade_pinn_net_epoch00500.pt:ep500 \
  --checkpoint checkpoints/fo_cascade_pinn_net.pt:final \
  --per-joint \
  --csv-out checkpoints/compare_rmse.csv
```

共享逻辑在 `examples/robot_compare_common.py`（加载 checkpoint、滑窗推理、RMSE 计算）。

合成 2-DoF `.npz` 数值测试：

```bash
python scripts/evaluate_model.py \
  --model checkpoints/RobotDynamics.pt \
  --test-data data/test_synthetic_2dof_inverse.npz
```

（需先用 `scripts/generate_test_dataset.py` 生成测试 npz。）

### 4.4 实现验证（可选）

```bash
python scripts/verify_delan_impl.py
```

---

## 5. 单元测试

```bash
cd /path/to/DeLaN_Stribeck
conda activate frictionest   # 或 DeLaN 等含 torch 的环境
pytest tests/ -q
```

WSL / ROS 环境下若 pytest 被 `launch_testing` 等插件干扰：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

覆盖：`test_stribeck.py`、`test_fo_cascade.py`、`test_friction_loss.py`、`test_pinn_tau_blend.py`、`test_model_shapes.py`、`test_arbitrary_dof.py`。

---

## 6. Python API 速查

```python
# DeLaN
from RobotDynamics.DeLaN import LNet, load_dataset, train_delan_loop, suggest_hyper

# 摩擦（TCN / FO 级联 / SCV / PINN 损失）
from RobotDynamics.FrictionModule import (
    HNetFOCascadePINN,
    HNetStribeckPINN,
    friction_pinn_loss,
    scv_torque,
    warmstart_scv_from_samples,
)

# 联合模型
from RobotDynamics.MystericNet import MystericNet, PINN_FRICTION_BACKENDS

# 数据导入
from RobotDynamics.DeLaN import import_mat, save_pickle, inspect_dataset
```

---

## 7. 常见问题

**Q: 训练 loss 很低但图上部分关节很差？**  
多关节力矩尺度差时，请用 `--tau-loss smape`；以测试集 **RMSE_test**、`robot_evaluate.py` 或 `robot_compare_metrics.py` 为准。

**Q: 无 τ_fri 标签时摩擦学不起来？**  
用 **`fo_cascade_pinn`** + **`--friction-label none`**；配合 **S0-W warmup** 与 **S2-H 专训摩擦**；无标签时 `pinn-loss-mode` 自动为 **`tau_blend`**（见 `FrictionModule/readme.md`）。

**Q: `robot.pickle` 上摩擦网络学不好？**  
若 `m+c+g ≈ τ`（几乎无摩擦残差），摩擦子网只能接近零；需真实摩擦或含分解的 `robot_fric` 类数据。

**Q: checkpoint 与数据 DoF 不一致？**  
`robot_evaluate.py` / 续训 `--resume` 会报错；勿用 2-DoF 权重评 6 轴数据。

**Q: 旧 GMS checkpoint 还能用吗？**  
**不能**。main 分支已移除 `gms` / `gms_pinn`；请用 `fo_cascade_pinn` 重新训练。

**Q: 如何接着上次训练？**  
`robot_train.py --resume checkpoints/{backend}_net.pt`，**`--epochs` 为总目标轮数**；见 §3.2。

**Q: 找不到 `mysteric_net`？**  
包已重命名为 **`RobotDynamics`**，请更新 import 与 IDE 工作区根目录。

---

## 参考

- DeLaN / L-Net：Lutter et al., 拉格朗日神经网络  
- Mysteric-Net / TCN 摩擦：Yeo et al.  
- FO 级联摩擦结构：Xun et al.（TCN→MLP→TCN 简化实现见 `fo_cascade.py`）  
- Stribeck PINN 摩擦：Hu et al., *Physics-Informed Learning for the Friction Modeling*（SCV Eq. (3)(4)，PINN Eq. (6)）

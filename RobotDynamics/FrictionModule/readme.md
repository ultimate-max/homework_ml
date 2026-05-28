# FrictionModule 摩擦损失设计

本目录实现 Mysteric-Net 训练中的**摩擦监督**、**PINN 物理约束**（Hu 等）与可选的**刚体能量一致性**（Yeo 等）。实现入口：

| 文件 | 函数 | 作用 |
|------|------|------|
| `losses.py` | `friction_supervised_loss` | 纯数据监督（MSE / SMAPE） |
| `losses.py` | `friction_pinn_loss` | Hu 等 Eq. (6)：数据项 + SCV 物理项 |
| `energy_loss.py` | `mysteric_losses` | Yeo 等 Eq. (7)：总力矩 MSE + 能量率残差 |

基础度量 `mse` / `smape` 复用 `RobotDynamics.DeLaN.losses.torque_loss`，与总力矩损失 `--tau-loss` 使用同一套公式，便于多关节、小力矩关节与总力矩尺度对齐。

---

## 1. 训练总损失（Mysteric-Net）

在 `examples/robot_train.py` 中，每个 batch 的标量损失为：

$$
\mathcal{L} = l_\tau + w_{\text{fri}}\, l_{\text{fri}} \;+\; [\;l_E\;]
$$

| 符号 | 默认 | 含义 |
|------|------|------|
| $l_\tau$ | 始终 | 总力矩 $\hat\tau$ 对测量 $\tau$：`torque_loss(..., kind=tau_loss)`，推荐 `smape` |
| $l_{\text{fri}}$ | 有摩擦标签或 PINN 后端 | 见下文 §2–§3 |
| $w_{\text{fri}}$ | `1.0` | CLI：`--friction-loss-weight` |
| $l_E$ | 关闭 | 仅 `--energy-loss` 时加上，见 §4 |

**合成数据**（`examples/synthetic_train.py`）：无 pickle 的 m/c/g 分解时，摩擦真值取 `τ_fri_true = τ_meas - τ_rigid.detach()`；若开启 `--energy-loss`，总损失为 `mysteric_losses` 的 $l_\tau + l_E$ 再 **加上** $l_{\text{fri}}$。

---

## 2. 基础度量：`mse` 与 `smape`

由 `torque_loss(τ̂, τ, kind)` 计算，对 batch 内所有关节、样本取平均。

**MSE**

$$
l_{\text{mse}} = \mathbb{E}\left[(\hat\tau - \tau)^2\right]
$$

**SMAPE**（与 DeLaN 论文式 (20) 一致，`smape_eps` 默认 `1e-3`）

$$
l_{\text{smape}} = \mathbb{E}\left[\frac{2\,|\hat\tau - \tau|}{|\tau| + |\hat\tau| + \varepsilon}\right]
$$

**选用建议**

- 多关节 / 各关节力矩量级差异大：**`fri_loss='smape'`**（与 `--tau-loss smape` 一致），$l_{\text{fri}}$ 与 $l_\tau$ 量级通常接近，$w_{\text{fri}}=1.0$ 即可。
- 若仍用 MSE 且 $l_{\text{fri}} \gg l_\tau$：将 `--friction-loss-weight` 降到 `0.01~0.1`，或改回 SMAPE。

---

## 3. 摩擦监督损失

### 3.1 纯监督：`friction_supervised_loss`

$$
l_{\text{fri}} = \text{loss}(\hat\tau_{\text{fri}},\; \tau_{\text{fri,true}})
$$

其中 `loss` 为 `mse` 或 `smape`（参数 `kind` / CLI `--fri-loss`）。

**何时使用**

- 摩擦后端为 `tcn`、`fo_cascade`、`stribeck`，且数据中有摩擦标签（pickle 含 m/c/g 分解，或 `--friction-label decomposition`）。
- **不**用于 `stribeck_pinn` / `fo_cascade_pinn`（这两类走 PINN 损失）。

### 3.2 PINN：`friction_pinn_loss`（Hu 等 Eq. (6)）

网络输出：

- $\hat\tau_{\text{fri}}$：MLP / TCN 等**数据支路**预测；
- $\tau_{\text{fri,physics}}$：可学习 **SCV** 在瞬时速度 $\dot q_t$ 上的力矩（`stribeck.scv_torque`，Hu 等 Eq. (3)(4)）。

**有摩擦真值**（`supervise_friction=True`，默认）：

$$
l_{\text{fri}} = (1-\lambda)\,\text{loss}(\hat\tau_{\text{fri}},\; \tau_{\text{fri,true}})
+ \lambda\,\text{loss}(\hat\tau_{\text{fri}},\; \tau_{\text{fri,physics}})
$$

**无摩擦真值**（`--friction-label none` 等，`supervise_friction=False`）：

$$
l_{\text{fri}} = \lambda\,\text{loss}(\hat\tau_{\text{fri}},\; \tau_{\text{fri,physics}}),\quad l_{\text{data}} = 0
$$

数据项与物理项**共用**同一 `fri_loss`（`mse` / `smape`）。$\lambda$ 对应 CLI `--lambda-physics`，默认 `0.5`。

**何时使用**

- `--friction-backend stribeck_pinn` 或 `fo_cascade_pinn`。
- 物理支路由各后端的 `HNet*.scv` 模块提供；`tau_phys` 在 `MystericNet.forward` 中返回。

**返回值**：`(l_total, l_data, l_phys)`，便于日志拆分数据项与 SCV 项。

### 3.3 SCV 物理参数从何而来

**结论**：SCV 系数**不是推理时猜的**，而是 `nn.Parameter`，在训练中用反向传播从数据（及 PINN 约束）学出来；代码里只有**量级合理的默认初值**。

**公式结构（固定）** — 实现于 `stribeck.py` 的 `scv_torque`（Hu 等 Eq. (4)）：

$$
\tau_{\text{fri,physics}} = k_v \dot q + k_c \tanh(k_a \dot q) + (k_s - k_c)\, e^{-|\dot q / v_s|^\alpha}\, \tanh(k_a \dot q)
$$

输入仅为当前时刻关节速度 $\dot q_t$（`qd_seq` 最后一帧）。**形状由物理模型给定，六个标量每关节一组，可学习。**

**参数化方式** — `StribeckSCVParams`：`k_v,k_c,k_a,v_s,α` 经 `softplus` 保证 **> 0**；**$k_s = k_c + \mathrm{softplus}(\log\Delta k_s)$** 保证 **$k_s \ge k_c$**（Stribeck 静摩擦不低于库仑摩擦，训练过程中恒成立）：

| 参数 | 默认初值（约） | 含义 |
|------|----------------|------|
| $k_v$ | 0.1 | 粘性 |
| $k_c$ | 2.0 | 库仑（6 轴 `robot_fric` 量级；新训练自动 warm-start 会按数据覆盖） |
| $k_a$ | 10 | $\tanh$ 在零速附近的陡度 |
| $\Delta k_s$ | 0.1 → $k_s=k_c+\Delta k_s\approx 2.1$ | Stribeck 静摩擦超额 |
| $v_s$ | 0.05 | Stribeck 特征速度 |
| $\alpha$ | 1.5 | Stribeck 曲线指数 |

旧 checkpoint 若仍含 `log_k_s`，加载时自动迁移为 `log_delta_k_s = \mathrm{inv\_softplus}(k_s-k_c)`。

初值是优化起点，**不是**台架标定或曲线拟合的最终结果；本仓库**未**接入离线 Stribeck 辨识脚本。

**训练时谁更新 SCV**

| 后端 | SCV 如何得到梯度 |
|------|------------------|
| `stribeck` | 仅数据项：$\text{loss}(\tau_{\text{fri,physics}}, \tau_{\text{fri,true}})$，SCV 即唯一摩擦输出 |
| `stribeck_pinn` / `fo_cascade_pinn` | 数据项更新 MLP；物理项 $\lambda\,\text{loss}(\hat\tau_{\text{fri}}, \tau_{\text{fri,physics}})$ **同时**更新 MLP 与 SCV（二者被拉向一致） |
| 无摩擦标签（`--friction-label none`） | 仅物理项：SCV 主要靠「把 MLP 预测钉在 Stribeck 曲线上」；MLP 仍受总力矩 $l_\tau$ 约束。速度分布若不足以扫过 Stribeck 区，部分参数可辨识性较弱 |

SCV 与 L-Net、摩擦 MLP 一样由 `Adam` 联合更新；**没有**单独的最小二乘 Stribeck 拟合步骤。

**训练后查看** — `examples/robot_evaluate.py` 中 `print_scv_params` 会打印各关节学到的 $k_v, k_c, \ldots$（对 `log_*` 做 `softplus` 后的数值）。

**与经典做法对比**

| | 本仓库 | 传统台架辨识 |
|--|--------|----------------|
| 模型 | 固定 SCV 公式 | 常为同类参数化模型 |
| 系数 | 端到端梯度 + 数据/PINN 损失 | 低速/高速试验 + 最小二乘 |
| 初值 | `StribeckSCVParams.__init__` 内写死 | 文献值或预拟合 |

若需从已知台架参数热启动，可改 `stribeck.py` 中 `StribeckSCVParams` 的初值，或从 checkpoint 加载 `scv.*` 权重。

---

## 4. 可选能量损失：`mysteric_losses`（Yeo 等 Eq. (7)）

$$
l_E = \mathbb{E}\left[\left(\frac{\mathrm{d}T}{\mathrm{d}t}+\frac{\mathrm{d}V}{\mathrm{d}t} - (\tau - \hat\tau_{\text{fri}})^\top \dot q\right)^2\right]
$$

$$
\mathcal{L}_{\text{mysteric}} = l_\tau^{\text{(mse)}} + l_E
$$

其中：

- $\dfrac{\mathrm{d}T}{\mathrm{d}t}+\dfrac{\mathrm{d}V}{\mathrm{d}t}$ 由 **L-Net** 的 `dynamics(q, q̇, q̈)` 得到（与 `deep_lagrangian_networks` 能量率定义一致）；若无 `dynamics`，则用 $H(\hat q)$ 与 $\dot q$ 自动微分构造 $\mathrm{d}T/\mathrm{d}t$，再加 $g^\top \dot q$（$g=\nabla_q V$）。
- $(\tau - \hat\tau_{\text{fri}})^\top \dot q$：从总力矩中去掉摩擦预测后的功率残差。

在 `robot_train.py` 中，开启 `--energy-loss` 时：**在** $l_\tau + w_{\text{fri}} l_{\text{fri}}$ **之上再加** $l_E$（$l_\tau$ 仍用 `--tau-loss`，能量项内部总力矩项为 MSE）。

注意：能量项训练更慢（需 L-Net 能量率前向），默认关闭。

---

## 5. 后端与损失对照

| `--friction-backend` | $l_{\text{fri}}$ 实现 | 物理支路 |
|----------------------|-------------------------|----------|
| `tcn` | `friction_supervised_loss`（有标签时） | 无 |
| `fo_cascade` | 同上 | 无 |
| `stribeck` | 同上（纯 SCV 参数，无 MLP） | 无 PINN 损失 |
| `stribeck_pinn` | `friction_pinn_loss` | SCV($\dot q$) |
| `fo_cascade_pinn` | `friction_pinn_loss` | SCV($\dot q$) |

摩擦标签（`robot_train.py`）：

- `--friction-label auto`（默认）：pickle 含 m/c/g 分解则监督 $\tau_{\text{fri}}$；
- `none`：不监督摩擦真值；PINN 后端仍用 SCV 物理项；
- `decomposition`：强制用分解得到的 $\tau_{\text{fri}}$。

---

## 6. 命令行参数速查

| 参数 | 默认 | 说明 |
|------|------|------|
| `--tau-loss` | `smape` | 总力矩 $l_\tau$ |
| `--fri-loss` | `smape` | 摩擦监督及 PINN 两端的 `loss` |
| `--smape-eps` | `1e-3` | SMAPE 分母稳定项 $\varepsilon$ |
| `--friction-loss-weight` | `1.0` | $w_{\text{fri}}$ |
| `--lambda-physics` | `0.5` | PINN 中 $\lambda$ |
| `--energy-loss` | 关 | 是否加 $l_E$ |
| `--friction-label` | `auto` | 是否提供 $\tau_{\text{fri,true}}$ |

示例（机器人数据 + Stribeck PINN）：

```bash
python examples/robot_train.py \
  --data data/robot.pickle \
  --friction-backend stribeck_pinn \
  --lambda-physics 0.5 \
  --tau-loss smape \
  --fri-loss smape \
  --friction-loss-weight 1.0 \
  -m 1
```

无摩擦标签、仅物理约束：

```bash
python examples/robot_train.py \
  --data data/robot.pickle \
  --friction-backend fo_cascade_pinn \
  --friction-label none \
  --lambda-physics 0.5 \
  -m 1
```

---

## 7. Python API

```python
from RobotDynamics.FrictionModule import (
    friction_supervised_loss,
    friction_pinn_loss,
    mysteric_losses,
)

# 纯监督
l_fri = friction_supervised_loss(
    tau_fri_pred, tau_fri_target, kind="smape", smape_eps=1e-3
)

# PINN（返回 total, data, physics）
l_total, l_data, l_phys = friction_pinn_loss(
    tau_fri_pred,
    tau_fri_target,
    tau_fri_physics,
    lambda_physics=0.5,
    supervise_friction=True,
    fri_loss="smape",
    smape_eps=1e-3,
)

# 能量项（需 lnet、状态与 g_hat）
l_tot, l_tau, l_E = mysteric_losses(
    lnet, tau_hat, tau_target, tau_fri_hat, q, qd, qdd, g
)
```

单元测试：`tests/test_friction_loss.py`（SMAPE 有界、PINN 标量输出形状）。

---

## 8. 单电机惯量 + 摩擦辨识（应用）

单轴伺服（`n_dof=1`）端到端流程见项目根目录 **README §3.3** 与 `examples/motor_identify_train.py`：DeLaN 学 $H \approx J$，`fo_cascade_pinn` + 本节 PINN 损失学摩擦；无摩擦标签时仅用 $\lambda\,\text{loss}(\hat\tau_{\text{fri}}, \tau_{\text{fri,physics}})$ 与总力矩 $l_\tau$。

---

## 9. 参考文献

- **SCV 摩擦模型**：Hu et al., *Physics-Informed Learning for the Friction Modeling* — Eq. (3)(4) 实现于 `stribeck.py`。
- **PINN 摩擦损失**：同上 — Eq. (6) 实现于 `losses.friction_pinn_loss`。
- **能量一致性**：Yeo et al.（Mysteric-Net）— Eq. (7) 实现于 `energy_loss.mysteric_losses`。
- **总力矩 SMAPE**：DeLaN 论文式 (20)，实现于 `RobotDynamics/DeLaN/losses.py`。

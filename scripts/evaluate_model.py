#!/usr/bin/env python3
"""
在测试集上评估训练好的Mysteric-Net模型，输出预测效果和摩擦力矩分析。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.model import MystericNet
from mysteric_net.synthetic_plant import build_windows


def load_dataset_npz(path: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    """加载测试数据集"""
    z = np.load(path, allow_pickle=False)
    seq_len = int(z["seq_len"])
    dof = int(z["dof"])
    qi = torch.from_numpy(z["qi"]).to(device=device, dtype=torch.float32)
    qdi = torch.from_numpy(z["qdi"]).to(device=device, dtype=torch.float32)
    qddi = torch.from_numpy(z["qddi"]).to(device=device, dtype=torch.float32)
    taui = torch.from_numpy(z["taui"]).to(device=device, dtype=torch.float32)
    q_seq = torch.from_numpy(z["q_seq"]).to(device=device, dtype=torch.float32)
    qd_seq = torch.from_numpy(z["qd_seq"]).to(device=device, dtype=torch.float32)
    
    # 对摩擦力矩也进行滑窗处理，使其与预测维度匹配
    tau_fri_full = torch.from_numpy(z["tau_fri"]).to(device=device, dtype=torch.float32)
    _, _, _, tau_fri_i, _, _ = build_windows(tau_fri_full, torch.zeros_like(tau_fri_full), 
                                            torch.zeros_like(tau_fri_full), torch.zeros_like(tau_fri_full), 
                                            seq_len)
    
    return qi, qdi, qddi, taui, tau_fri_i, q_seq, qd_seq, seq_len, dof


def load_model(model_path: Path, device: torch.device) -> MystericNet:
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    dof = checkpoint.get('dof', 2)
    seq_len = checkpoint.get('seq_len', 30)
    
    backend = checkpoint.get("friction_backend", "tcn")
    model = MystericNet(dof=dof, seq_len=seq_len, friction_backend=backend).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints" / "mysteric_net.pt", 
                   help="训练好的模型路径")
    p.add_argument("--test-data", type=Path, default=ROOT / "data" / "test_synthetic_2dof_inverse.npz",
                   help="测试数据集路径")
    p.add_argument("--save-predictions", type=Path, default=ROOT / "results" / "test_predictions.npz",
                   help="保存预测结果的路径")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model}")
    model, checkpoint = load_model(args.model, device)
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"训练RMSE: {checkpoint.get('rmse', 'N/A')}")

    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    if not args.test_data.is_file():
        raise SystemExit(f"测试数据文件不存在: {args.test_data}")
    
    qi, qdi, qddi, taui, tau_fri_true, q_seq, qd_seq, seq_len, dof = load_dataset_npz(args.test_data, device)
    print(f"测试样本数: {qi.shape[0]}")
    print(f"自由度: {dof}")
    print(f"序列长度: {seq_len}")

    # 评估模型
    print("\n开始评估...")
    with torch.no_grad():
        tau_hat, tau_core_hat, tau_fri_hat, H_hat, g_hat, _ = model(
            qi, qdi, qddi, q_seq, qd_seq
        )

    # 计算各项指标
    tau_error = tau_hat - taui
    rmse_total = torch.sqrt(torch.mean(tau_error ** 2)).item()
    rmse_per_joint = torch.sqrt(torch.mean(tau_error ** 2, dim=0)).cpu().numpy()
    mae_total = torch.mean(torch.abs(tau_error)).item()
    mae_per_joint = torch.mean(torch.abs(tau_error), dim=0).cpu().numpy()

    # 摩擦力矩误差
    tau_fri_error = tau_fri_hat - tau_fri_true
    rmse_fri = torch.sqrt(torch.mean(tau_fri_error ** 2)).item()
    rmse_fri_per_joint = torch.sqrt(torch.mean(tau_fri_error ** 2, dim=0)).cpu().numpy()

    # 输出结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    print(f"总力矩 RMSE: {rmse_total:.6f}")
    print(f"总力矩 MAE:  {mae_total:.6f}")
    print(f"\n各关节总力矩 RMSE:")
    for i, rmse in enumerate(rmse_per_joint):
        print(f"  关节 {i+1}: {rmse:.6f}")
    print(f"\n各关节总力矩 MAE:")
    for i, mae in enumerate(mae_per_joint):
        print(f"  关节 {i+1}: {mae:.6f}")
    
    print(f"\n摩擦力矩 RMSE: {rmse_fri:.6f}")
    print(f"各关节摩擦力矩 RMSE:")
    for i, rmse in enumerate(rmse_fri_per_joint):
        print(f"  关节 {i+1}: {rmse:.6f}")

    # 统计信息
    print(f"\n摩擦力矩统计:")
    tau_fri_true_np = tau_fri_true.cpu().numpy()
    tau_fri_hat_np = tau_fri_hat.cpu().numpy()
    for i in range(dof):
        print(f"  关节 {i+1}:")
        print(f"    真实摩擦力矩范围: [{tau_fri_true_np[:, i].min():.4f}, {tau_fri_true_np[:, i].max():.4f}]")
        print(f"    预测摩擦力矩范围: [{tau_fri_hat_np[:, i].min():.4f}, {tau_fri_hat_np[:, i].max():.4f}]")
        print(f"    真实摩擦力矩均值: {tau_fri_true_np[:, i].mean():.4f}")
        print(f"    预测摩擦力矩均值: {tau_fri_hat_np[:, i].mean():.4f}")

    # 保存预测结果
    args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.save_predictions,
        tau_true=taui.cpu().numpy(),
        tau_pred=tau_hat.cpu().numpy(),
        tau_core_pred=tau_core_hat.cpu().numpy(),
        tau_fri_true=tau_fri_true.cpu().numpy(),
        tau_fri_pred=tau_fri_hat.cpu().numpy(),
        q=qi.cpu().numpy(),
        qd=qdi.cpu().numpy(),
        qdd=qddi.cpu().numpy(),
        rmse_total=rmse_total,
        mae_total=mae_total,
        rmse_per_joint=rmse_per_joint,
        mae_per_joint=mae_per_joint,
        rmse_fri=rmse_fri,
        rmse_fri_per_joint=rmse_fri_per_joint,
    )
    print(f"\n预测结果已保存: {args.save_predictions}")


if __name__ == "__main__":
    main()

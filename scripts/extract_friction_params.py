#!/usr/bin/env python3
"""
提取训练好的Mysteric-Net模型中的摩擦网络(H-Net)参数和权重。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mysteric_net.model import MystericNet


def extract_hnet_params(model: MystericNet, device: torch.device) -> dict:
    """提取H-Net的所有参数"""
    hnet_params = {}
    
    # TCN层参数
    hnet_params['tcn'] = []
    for i, layer in enumerate(model.hnet.tcn):
        if isinstance(layer, torch.nn.Conv1d):
            layer_params = {
                'type': 'Conv1d',
                'index': i,
                'in_channels': layer.in_channels,
                'out_channels': layer.out_channels,
                'kernel_size': layer.kernel_size[0],
                'padding': layer.padding[0],
                'weight': layer.weight.detach().cpu().numpy().tolist(),
                'bias': layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else None
            }
            hnet_params['tcn'].append(layer_params)
        elif isinstance(layer, torch.nn.ReLU):
            hnet_params['tcn'].append({'type': 'ReLU', 'index': i, 'inplace': layer.inplace})
    
    # Head层参数
    hnet_params['head'] = {
        'type': 'Linear',
        'in_features': model.hnet.head.in_features,
        'out_features': model.hnet.head.out_features,
        'weight': model.hnet.head.weight.detach().cpu().numpy().tolist(),
        'bias': model.hnet.head.bias.detach().cpu().numpy().tolist()
    }
    
    # 网络结构信息
    hnet_params['structure'] = {
        'dof': model.dof,
        'seq_len': model.seq_len,
        'input_dim': 2 * model.dof,  # q_seq 和 qd_seq 拼接
        'hidden_channels': model.hnet.tcn[0].out_channels,
        'kernel_size': model.hnet.tcn[0].kernel_size[0],
        'total_parameters': sum(p.numel() for p in model.hnet.parameters())
    }
    
    return hnet_params


def extract_lnet_params(model: MystericNet, device: torch.device) -> dict:
    """提取L-Net的参数（刚体部分）"""
    lnet_params = {}
    
    # Mass network参数
    lnet_params['mass_network'] = {
        'hidden_dim': model.lnet._mass_head[0].in_features if len(model.lnet._mass_head) > 0 else 32,
        'output_dim': model.lnet._mass_head[-1].out_features,
        'layers': []
    }
    
    for i, layer in enumerate(model.lnet._mass_head):
        if isinstance(layer, torch.nn.Linear):
            layer_params = {
                'index': i,
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'weight': layer.weight.detach().cpu().numpy().tolist(),
                'bias': layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else None
            }
            lnet_params['mass_network']['layers'].append(layer_params)
        elif isinstance(layer, torch.nn.Tanh):
            lnet_params['mass_network']['layers'].append({'index': i, 'type': 'Tanh'})
    
    # Potential network参数
    lnet_params['potential_network'] = {
        'hidden_dim': model.lnet._pot_head[0].in_features if len(model.lnet._pot_head) > 0 else 32,
        'output_dim': model.lnet._pot_head[-1].out_features,
        'layers': []
    }
    
    for i, layer in enumerate(model.lnet._pot_head):
        if isinstance(layer, torch.nn.Linear):
            layer_params = {
                'index': i,
                'in_features': layer.in_features,
                'out_features': layer.out_features,
                'weight': layer.weight.detach().cpu().numpy().tolist(),
                'bias': layer.bias.detach().cpu().numpy().tolist() if layer.bias is not None else None
            }
            lnet_params['potential_network']['layers'].append(layer_params)
        elif isinstance(layer, torch.nn.Tanh):
            lnet_params['potential_network']['layers'].append({'index': i, 'type': 'Tanh'})
    
    # 网络结构信息
    lnet_params['structure'] = {
        'dof': model.dof,
        'mass_diag_eps': model.lnet.mass_diag_eps,
        'total_parameters': sum(p.numel() for p in model.lnet.parameters())
    }
    
    return lnet_params


def load_model(model_path: Path, device: torch.device) -> tuple[MystericNet, dict]:
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    dof = checkpoint.get('dof', 2)
    seq_len = checkpoint.get('seq_len', 30)
    
    model = MystericNet(dof=dof, seq_len=seq_len).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "checkpoints" / "mysteric_net.pt",
                   help="训练好的模型路径")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results",
                   help="参数输出目录")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model}")
    if not args.model.is_file():
        raise SystemExit(f"模型文件不存在: {args.model}")
    
    model, checkpoint = load_model(args.model, device)
    print(f"训练轮数: {checkpoint.get('epoch', 'N/A')}")
    print(f"训练RMSE: {checkpoint.get('rmse', 'N/A')}")

    # 提取参数
    print("\n提取模型参数...")
    
    # 提取H-Net参数
    hnet_params = extract_hnet_params(model, device)
    
    # 提取L-Net参数
    lnet_params = extract_lnet_params(model, device)
    
    # 合并所有参数
    all_params = {
        'checkpoint_info': {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'rmse': checkpoint.get('rmse', 'N/A'),
            'dof': checkpoint.get('dof', 2),
            'seq_len': checkpoint.get('seq_len', 30)
        },
        'hnet': hnet_params,
        'lnet': lnet_params,
        'total_model_parameters': sum(p.numel() for p in model.parameters())
    }

    # 保存为JSON文件
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "model_parameters.json"
    
    with open(output_json, 'w') as f:
        json.dump(all_params, f, indent=2)
    
    print(f"\n参数已保存: {output_json}")

    # 打印摘要信息
    print("\n" + "="*60)
    print("模型参数摘要")
    print("="*60)
    print(f"总参数数量: {all_params['total_model_parameters']:,}")
    print(f"H-Net参数数量: {all_params['hnet']['structure']['total_parameters']:,}")
    print(f"L-Net参数数量: {all_params['lnet']['structure']['total_parameters']:,}")
    
    print(f"\nH-Net结构:")
    print(f"  自由度: {all_params['hnet']['structure']['dof']}")
    print(f"  序列长度: {all_params['hnet']['structure']['seq_len']}")
    print(f"  隐藏通道数: {all_params['hnet']['structure']['hidden_channels']}")
    print(f"  卷积核大小: {all_params['hnet']['structure']['kernel_size']}")
    
    print(f"\nH-Net TCN层数: {len([l for l in all_params['hnet']['tcn'] if l['type'] == 'Conv1d'])}")
    print(f"H-Net输出层: {all_params['hnet']['head']['in_features']} -> {all_params['hnet']['head']['out_features']}")
    
    print(f"\nL-Net结构:")
    print(f"  自由度: {all_params['lnet']['structure']['dof']}")
    print(f"  质量矩阵对角线epsilon: {all_params['lnet']['structure']['mass_diag_eps']}")
    
    print("\n" + "="*60)
    print("摩擦网络(H-Net)权重统计")
    print("="*60)
    
    # 统计H-Net权重
    for i, layer in enumerate(all_params['hnet']['tcn']):
        if layer['type'] == 'Conv1d':
            weights = np.array(layer['weight'])
            print(f"\nTCN Conv1d Layer {i}:")
            print(f"  权重形状: {weights.shape}")
            print(f"  权重范围: [{weights.min():.6f}, {weights.max():.6f}]")
            print(f"  权重均值: {weights.mean():.6f}")
            print(f"  权重标准差: {weights.std():.6f}")
            if layer['bias'] is not None:
                bias = np.array(layer['bias'])
                print(f"  偏置形状: {bias.shape}")
                print(f"  偏置范围: [{bias.min():.6f}, {bias.max():.6f}]")
    
    # 统计Head层权重
    head_weights = np.array(all_params['hnet']['head']['weight'])
    print(f"\nH-Net Head Layer:")
    print(f"  权重形状: {head_weights.shape}")
    print(f"  权重范围: [{head_weights.min():.6f}, {head_weights.max():.6f}]")
    print(f"  权重均值: {head_weights.mean():.6f}")
    print(f"  权重标准差: {head_weights.std():.6f}")
    head_bias = np.array(all_params['hnet']['head']['bias'])
    print(f"  偏置范围: [{head_bias.min():.6f}, {head_bias.max():.6f}]")


if __name__ == "__main__":
    main()

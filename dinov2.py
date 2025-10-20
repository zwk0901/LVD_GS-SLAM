#!/usr/bin/env python3
"""
DINOv2 高级特征热力图可视化工具
可视化DINOv2模型的不同层级特征，生成语义热力图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os
from typing import Dict, List, Tuple, Optional
import requests
from PIL import Image
from torchvision import transforms


class DINOv2HeatmapVisualizer:
    """DINOv2特征热力图可视化器"""

    def __init__(self, model_name='dinov2_vitb14', device='cuda'):
        """
        初始化DINOv2模型
        model_name: 可选 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        # 加载DINOv2模型
        print(f"加载{model_name}模型...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 特征存储
        self.features = {}
        self.hooks = []

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 获取patch大小
        self.patch_size = self.model.patch_size

    def register_hooks(self):
        """注册特征提取钩子"""

        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output.detach()

            return hook

        # 注册到不同的transformer块
        for i, block in enumerate(self.model.blocks):
            # 注册注意力输出
            self.hooks.append(
                block.attn.register_forward_hook(get_activation(f'block_{i}_attn'))
            )
            # 注册块输出
            self.hooks.append(
                block.register_forward_hook(get_activation(f'block_{i}_output'))
            )

        # 注册最终的norm层输出
        self.hooks.append(
            self.model.norm.register_forward_hook(get_activation('final_norm'))
        )

        print(f"✅ 注册了 {len(self.hooks)} 个特征提取钩子")

    def extract_features(self, image_path: str) -> Dict[str, torch.Tensor]:
        """提取图像特征"""
        self.features.clear()

        # 加载并预处理图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        # 保存原始图像大小
        self.original_size = image.size

        # 预处理
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 前向传播
        with torch.no_grad():
            _ = self.model(image_tensor)

        return self.features.copy()

    def create_heatmap(self, features: torch.Tensor, target_size: Tuple[int, int]) -> np.ndarray:
        """
        将特征转换为热力图
        features: [B, N, D] 或 [B, D, H, W]
        target_size: (H, W) 目标大小
        """
        if len(features.shape) == 3:  # Transformer输出 [B, N, D]
            B, N, D = features.shape
            # 计算特征图大小
            h = w = int(np.sqrt(N - 1))  # 减1是因为有CLS token

            # 移除CLS token并重塑
            features_no_cls = features[:, 1:, :]  # [B, h*w, D]
            features_reshaped = features_no_cls.reshape(B, h, w, D)

            # 计算特征范数作为激活强度
            activation = features_reshaped.norm(dim=-1)[0].cpu().numpy()  # [h, w]

        elif len(features.shape) == 4:  # CNN风格 [B, C, H, W]
            # 计算通道维度的范数
            activation = features.norm(dim=1)[0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        # 上采样到目标大小
        heatmap = cv2.resize(activation, target_size[::-1], interpolation=cv2.INTER_CUBIC)

        # 归一化
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

        return heatmap

    def visualize_layer_heatmaps(self, image_path: str, output_dir: str,
                                 selected_layers: Optional[List[int]] = None):
        """可视化不同层的特征热力图"""
        os.makedirs(output_dir, exist_ok=True)

        # 提取特征
        features = self.extract_features(image_path)

        # 加载原图
        original_img = cv2.imread(image_path) if isinstance(image_path, str) else np.array(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]

        # 选择要可视化的层
        if selected_layers is None:
            # 默认选择：早期、中期、后期层
            n_blocks = len([k for k in features.keys() if 'block_' in k and '_output' in k])
            selected_layers = [0, n_blocks // 4, n_blocks // 2, 3 * n_blocks // 4, n_blocks - 1]

        # 创建可视化
        n_layers = len(selected_layers)
        fig, axes = plt.subplots(3, n_layers, figsize=(4 * n_layers, 12))

        for idx, layer_idx in enumerate(selected_layers):
            # 获取该层的输出特征
            layer_features = features.get(f'block_{layer_idx}_output')
            if layer_features is None:
                continue

            # 生成热力图
            heatmap = self.create_heatmap(layer_features, (h, w))

            # 1. 原始热力图
            ax1 = axes[0, idx] if n_layers > 1 else axes[0]
            im1 = ax1.imshow(heatmap, cmap='hot')
            ax1.set_title(f'Layer {layer_idx} Heatmap')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046)

            # 2. 带原图的叠加
            ax2 = axes[1, idx] if n_layers > 1 else axes[1]
            ax2.imshow(original_img)
            ax2.imshow(heatmap, alpha=0.5, cmap='hot')
            ax2.set_title(f'Layer {layer_idx} Overlay')
            ax2.axis('off')

            # 3. 使用不同的colormap展示
            ax3 = axes[2, idx] if n_layers > 1 else axes[2]
            im3 = ax3.imshow(heatmap, cmap='viridis')
            ax3.set_title(f'Layer {layer_idx} (Viridis)')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)

        plt.suptitle(f'DINOv2 Feature Heatmaps - {self.model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_heatmaps.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 层级热力图已保存到: {output_dir}/layer_heatmaps.png")

    def generate_attention_maps(self, image_path: str, output_dir: str):
        """生成注意力图可视化"""
        os.makedirs(output_dir, exist_ok=True)

        # 提取特征
        features = self.extract_features(image_path)

        # 加载原图
        original_img = cv2.imread(image_path) if isinstance(image_path, str) else np.array(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]

        # 选择几个关键层的注意力
        attention_layers = [0, 5, 10, -1]  # 早期、中期、后期

        fig, axes = plt.subplots(2, len(attention_layers), figsize=(16, 8))

        for idx, layer_idx in enumerate(attention_layers):
            # 获取注意力特征
            if layer_idx == -1:
                layer_idx = len([k for k in features.keys() if '_attn' in k]) - 1

            attn_features = features.get(f'block_{layer_idx}_attn')
            if attn_features is None:
                continue

            # DINOv2的注意力输出可能需要特殊处理
            # 这里简化处理，使用输出特征的范数
            if isinstance(attn_features, torch.Tensor):
                heatmap = self.create_heatmap(attn_features, (h, w))
            else:
                continue

            # 上图：纯注意力热力图
            ax1 = axes[0, idx]
            im1 = ax1.imshow(heatmap, cmap='plasma')
            ax1.set_title(f'Attention Layer {layer_idx}')
            ax1.axis('off')

            # 下图：叠加在原图上
            ax2 = axes[1, idx]
            ax2.imshow(original_img)
            ax2.imshow(heatmap, alpha=0.6, cmap='plasma')
            ax2.set_title(f'Attention Overlay')
            ax2.axis('off')

        plt.suptitle('DINOv2 Attention Maps', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'attention_maps.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 注意力图已保存到: {output_dir}/attention_maps.png")

    def create_semantic_heatmap(self, image_path: str, output_dir: str):
        """创建语义理解热力图 - 使用最后几层的特征"""
        os.makedirs(output_dir, exist_ok=True)

        # 提取特征
        features = self.extract_features(image_path)

        # 加载原图
        original_img = cv2.imread(image_path) if isinstance(image_path, str) else np.array(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        h, w = original_img.shape[:2]

        # 使用最后的norm层特征（最高级的语义特征）
        final_features = features.get('final_norm')
        if final_features is None:
            print("未找到final_norm特征")
            return

        # 创建热力图
        heatmap = self.create_heatmap(final_features, (h, w))

        # 创建多种可视化
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. 原图
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # 2. 语义热力图
        im2 = axes[0, 1].imshow(heatmap, cmap='hot')
        axes[0, 1].set_title('Semantic Heatmap (Hot)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])

        # 3. 叠加效果
        axes[0, 2].imshow(original_img)
        axes[0, 2].imshow(heatmap, alpha=0.5, cmap='hot')
        axes[0, 2].set_title('Overlay')
        axes[0, 2].axis('off')

        # 4. 不同的colormap
        im4 = axes[1, 0].imshow(heatmap, cmap='viridis')
        axes[1, 0].set_title('Semantic Heatmap (Viridis)')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0])

        # 5. 阈值化显示
        threshold = np.percentile(heatmap, 75)
        heatmap_thresholded = heatmap.copy()
        heatmap_thresholded[heatmap < threshold] = 0

        axes[1, 1].imshow(original_img)
        axes[1, 1].imshow(heatmap_thresholded, alpha=0.7, cmap='Reds')
        axes[1, 1].set_title('High Activation Regions (Top 25%)')
        axes[1, 1].axis('off')

        # 6. 轮廓图
        axes[1, 2].imshow(original_img)
        contours = axes[1, 2].contour(heatmap, levels=10, colors='white', linewidths=2)
        axes[1, 2].set_title('Activation Contours')
        axes[1, 2].axis('off')

        plt.suptitle(f'DINOv2 Semantic Feature Analysis - {self.model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'semantic_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 语义热力图已保存到: {output_dir}/semantic_heatmap.png")

    def analyze_feature_evolution(self, image_path: str, output_dir: str):
        """分析特征演化 - 从低级到高级"""
        os.makedirs(output_dir, exist_ok=True)

        # 提取特征
        features = self.extract_features(image_path)

        # 获取所有层的输出
        layer_outputs = []
        layer_indices = []

        for key in sorted(features.keys()):
            if '_output' in key and 'block_' in key:
                layer_idx = int(key.split('_')[1])
                layer_outputs.append(features[key])
                layer_indices.append(layer_idx)

        # 计算特征统计
        feature_stats = {
            'mean': [],
            'std': [],
            'sparsity': [],
            'entropy': []
        }

        for feat in layer_outputs:
            # 计算统计量
            feat_flat = feat.flatten()
            feature_stats['mean'].append(feat_flat.mean().item())
            feature_stats['std'].append(feat_flat.std().item())

            # 稀疏度（接近0的比例）
            sparsity = (feat_flat.abs() < 0.1).float().mean().item()
            feature_stats['sparsity'].append(sparsity)

            # 特征熵（归一化后）
            feat_norm = F.softmax(feat_flat.abs(), dim=0)
            entropy = -(feat_norm * (feat_norm + 1e-8).log()).sum().item()
            feature_stats['entropy'].append(entropy)

        # 可视化特征演化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. 均值演化
        axes[0, 0].plot(layer_indices, feature_stats['mean'], 'b-o')
        axes[0, 0].set_title('Feature Mean Evolution')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Mean Activation')
        axes[0, 0].grid(True)

        # 2. 标准差演化
        axes[0, 1].plot(layer_indices, feature_stats['std'], 'r-o')
        axes[0, 1].set_title('Feature Std Evolution')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Std Activation')
        axes[0, 1].grid(True)

        # 3. 稀疏度演化
        axes[1, 0].plot(layer_indices, feature_stats['sparsity'], 'g-o')
        axes[1, 0].set_title('Feature Sparsity Evolution')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Sparsity')
        axes[1, 0].grid(True)

        # 4. 熵演化
        axes[1, 1].plot(layer_indices, feature_stats['entropy'], 'm-o')
        axes[1, 1].set_title('Feature Entropy Evolution')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True)

        plt.suptitle('DINOv2 Feature Evolution Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_evolution.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✅ 特征演化分析已保存到: {output_dir}/feature_evolution.png")

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def create_comprehensive_analysis(self, image_path: str, output_dir: str):
        """创建综合分析报告"""
        print(f"\n=== DINOv2 特征分析 ===")
        print(f"模型: {self.model_name}")
        print(f"图像: {image_path}")
        print(f"输出目录: {output_dir}\n")

        # 1. 层级热力图
        print("1. 生成层级热力图...")
        self.visualize_layer_heatmaps(image_path, output_dir)

        # 2. 注意力图
        print("2. 生成注意力图...")
        self.generate_attention_maps(image_path, output_dir)

        # 3. 语义热力图
        print("3. 生成语义热力图...")
        self.create_semantic_heatmap(image_path, output_dir)

        # 4. 特征演化分析
        print("4. 分析特征演化...")
        self.analyze_feature_evolution(image_path, output_dir)

        print(f"\n✅ 所有分析完成！结果保存在: {output_dir}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='DINOv2 高级特征热力图可视化')
    parser.add_argument('--image', required=True, help='输入图像路径')
    parser.add_argument('--output', default='./dinov2_analysis', help='输出目录')
    parser.add_argument('--model', default='dinov2_vitb14',
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DINOv2模型选择')

    args = parser.parse_args()

    # 创建可视化器
    visualizer = DINOv2HeatmapVisualizer(model_name=args.model)

    # 注册钩子
    visualizer.register_hooks()

    try:
        # 执行综合分析
        visualizer.create_comprehensive_analysis(args.image, args.output)
    finally:
        # 清理钩子
        visualizer.remove_hooks()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
简化版VGGT训练脚本 - 基于原始TRAM训练脚本
"""

import os
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from lib.core.config import parse_args
from lib.core.losses import compile_criterion
from lib.utils.utils import prepare_output_dir, create_logger
from lib.trainer_vggt import VGGTTrainer  # 使用我们的VGGT Trainer

from lib.get_videoloader import get_dataloaders
from lib.models.tram_with_vggt_transfer import VGGTTransferLearning, TRAMWithVGGTTransfer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    # create logger
    logger = create_logger(cfg.LOGDIR, phase='train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # Dataloaders - 使用原始方式
    data_loaders = get_dataloaders(cfg)

    # Compile Loss - 使用原始方式
    criterion = compile_criterion(cfg)

    # VGGT模型 - 替换原始HMR_VIMO
    logger.info("初始化VGGT迁移学习模型...")

    checkpoint_path = cfg.MODEL.CHECKPOINT
    print(f"检查CHECKPOINT文件:")
    print(f"  路径: {checkpoint_path}")
    print(f"  存在: {os.path.exists(checkpoint_path)}")

    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"  文件可读取: ✅")
            print(f"  包含keys: {list(checkpoint.keys())}")
            if 'state_dict' in checkpoint:
                print(f"  state_dict keys数量: {len(checkpoint['state_dict'])}")
        except Exception as e:
            print(f"  文件读取失败: {e}")



    # 1. 提取VGGT权重
    vggt_checkpoint = getattr(cfg.MODEL, 'VGGT_CHECKPOINT', '~/.cache/torch/hub/checkpoints/model.pt')
    vggt_tool = VGGTTransferLearning(os.path.expanduser(vggt_checkpoint))
    backbone_path, camera_path, _ = vggt_tool.extract_and_save_weights()
    
    # 2. 创建VGGT模型
    freeze_backbone = getattr(cfg.MODEL, 'FREEZE_TRAM_BACKBONE', True)
    camera_fine_tune = getattr(cfg.MODEL, 'VGGT_CAMERA_FINE_TUNE', True)
    
    model = TRAMWithVGGTTransfer(
        backbone_path=backbone_path,
        camera_path=camera_path,
        freeze_backbone=freeze_backbone,
        camera_fine_tune=camera_fine_tune,
        cfg=cfg
    )

    print("\n=== 模型参数统计 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")
    print(f"模型显存估算: {total_params * 4 / (1024**3):.2f} GB")
    print(f"优化器显存估算: {trainable_params * 8 / (1024**3):.2f} GB")

    # 检查各个模块的参数量
    if hasattr(model, 'tram_backbone'):
        backbone_params = sum(p.numel() for p in model.tram_backbone.parameters())
        print(f"TRAM Backbone: {backbone_params:,} 参数")

    if hasattr(model, 'tram_smpl_head'):
        smpl_params = sum(p.numel() for p in model.tram_smpl_head.parameters())
        print(f"TRAM SMPL Head: {smpl_params:,} 参数")

    if hasattr(model, 'vggt_camera_head'):
        camera_params = sum(p.numel() for p in model.vggt_camera_head.parameters())
        print(f"VGGT Camera Head: {camera_params:,} 参数")

    if hasattr(model, 'motion_module') and model.motion_module is not None:
        motion_params = sum(p.numel() for p in model.motion_module.parameters())
        print(f"Motion Module: {motion_params:,} 参数")

    if hasattr(model, 'st_module') and model.st_module is not None:
        st_params = sum(p.numel() for p in model.st_module.parameters())
        print(f"ST Module: {st_params:,} 参数")

    if hasattr(model, 'feature_adapter'):
        adapter_params = sum(p.numel() for p in model.feature_adapter.parameters())
        print(f"Feature Adapter: {adapter_params:,} 参数")

    from lib.models.hmr_vimo import HMR_VIMO

    # 直接调用HMR_VIMO的方法来设置
    def freeze_modules(self):
        return HMR_VIMO.freeze_modules(self)

    def unfreeze_modules(self):
        return HMR_VIMO.unfreeze_modules(self)

    import types
    model.freeze_modules = types.MethodType(freeze_modules, model)
    model.unfreeze_modules = types.MethodType(unfreeze_modules, model)
 
    model = model.to(cfg.DEVICE)
    
    # 设置冻结模块（适配原始接口）
    model.frozen_modules = [model.tram_backbone] if freeze_backbone else []
    if hasattr(model, 'freeze_modules'):
        model.freeze_modules()

    logger.info(f'混合TRAM-VGGT模型初始化完成')

    # === 在这里替换原来的验证代码 ===
    print("\n=== 验证预训练权重加载 ===")

    # 检查backbone的第一层权重是否合理
    first_layer_weight = None
    first_layer_name = None
    for name, param in model.tram_backbone.named_parameters():
        if 'weight' in name:
            first_layer_weight = param
            first_layer_name = name
            break

    if first_layer_weight is not None:
        weight_mean = first_layer_weight.mean().item()
        weight_std = first_layer_weight.std().item()
        print(f"第一层权重名称: {first_layer_name}")
        print(f"VGGT版本 Backbone第一层: mean={weight_mean:.6f}, std={weight_std:.6f}")
        
        if abs(weight_mean) < 0.01 and weight_std < 0.1:
            print("⚠️ 权重分布像随机初始化")
        else:
            print("✅ 权重分布像预训练权重")

    # 检查权重加载的详细信息
    print("VGGT版本权重加载调试:")
    checkpoint_path = cfg.MODEL.CHECKPOINT
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone.')]
            print(f"Checkpoint中backbone keys数量: {len(backbone_keys)}")
            if len(backbone_keys) > 0:
                print("前5个backbone keys及其权重统计:")
                for key in backbone_keys[:5]:
                    print(f"  {key}")
                    if key in state_dict:
                        weight = state_dict[key]
                        if weight.dim() >= 2:  # 只看权重矩阵
                            print(f"    shape: {weight.shape}, mean: {weight.mean().item():.6f}, std: {weight.std().item():.6f}")

    # 检查SMPL head
    smpl_weight_count = sum(p.numel() for p in model.tram_smpl_head.parameters())
    print(f"SMPL head参数数量: {smpl_weight_count:,}")

    # 检查backbone是否真的冻结了
    backbone_trainable = sum(p.numel() for p in model.tram_backbone.parameters() if p.requires_grad)
    backbone_total = sum(p.numel() for p in model.tram_backbone.parameters())
    print(f"Backbone冻结状态: {backbone_trainable}/{backbone_total} 参数可训练")

    # 检查各组件的权重统计对比
    print(f"\n=== 各组件权重分布对比 ===")

    # TRAM backbone sample
    backbone_sample = None
    for name, param in model.tram_backbone.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            backbone_sample = param
            print(f"TRAM Backbone样本 ({name}): mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            break

    # TRAM SMPL head sample  
    smpl_sample = None
    for name, param in model.tram_smpl_head.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            smpl_sample = param
            print(f"TRAM SMPL Head样本 ({name}): mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            break

    # VGGT Camera head sample
    camera_sample = None
    for name, param in model.vggt_camera_head.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            camera_sample = param
            print(f"VGGT Camera Head样本 ({name}): mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            break

    # Feature adapter sample
    adapter_sample = None
    for name, param in model.feature_adapter.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            adapter_sample = param
            print(f"Feature Adapter样本 ({name}): mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            break

    print("=== 权重验证完成 ===\n")

    print(f"\n=== 优化器创建调试 ===")
    print(f"cfg.TRAIN.MULTI_LR: {cfg.TRAIN.MULTI_LR}")
    print(f"freeze_backbone: {freeze_backbone}")
    print(f"camera_fine_tune: {camera_fine_tune}")

    if cfg.TRAIN.MULTI_LR:
        print("进入MULTI_LR分支")
        params = []  # ← 确保这行在最开始
        print(f"params初始化完成: {params}")
        
        # VGGT Camera head（微调）
        if camera_fine_tune:
            camera_lr = getattr(cfg.TRAIN, 'LR_CAMERA_FINE_TUNE', 5e-6)
            params.append({
                'params': [p for p in model.vggt_camera_head.parameters() if p.requires_grad],
                'lr': camera_lr
            })
            print(f'VGGT Camera head lr: {camera_lr}')
        
        # TRAM SMPL head（主要训练目标）
        smpl_lr = cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR
        params.append({
            'params': [p for p in model.tram_smpl_head.parameters() if p.requires_grad],
            'lr': smpl_lr
        })
        print(f'TRAM SMPL head lr: {smpl_lr}')

        # Motion Module
        if hasattr(model, 'motion_module') and model.motion_module is not None:
            motion_params = [p for p in model.motion_module.parameters() if p.requires_grad]
            if motion_params:
                params.append({
                    'params': motion_params,
                    'lr': cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR
                })
                motion_param_count = sum(p.numel() for p in motion_params)
                print(f'Motion Module lr: {cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR} (参数数量: {motion_param_count:,})')

        # ST Module  
        if hasattr(model, 'st_module') and model.st_module is not None:
            st_params = [p for p in model.st_module.parameters() if p.requires_grad]
            if st_params:
                params.append({
                    'params': st_params,
                    'lr': cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR
                })
                st_param_count = sum(p.numel() for p in st_params)
                print(f'ST Module lr: {cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR} (参数数量: {st_param_count:,})')

        # Feature Adapter
        if hasattr(model, 'feature_adapter') and model.feature_adapter is not None:
            adapter_params = [p for p in model.feature_adapter.parameters() if p.requires_grad]
            if adapter_params:
                params.append({
                    'params': adapter_params,
                    'lr': cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR
                })
                adapter_param_count = sum(p.numel() for p in adapter_params)
                print(f'Feature Adapter lr: {cfg.TRAIN.LR2 if cfg.TRAIN.LR2 else cfg.TRAIN.LR} (参数数量: {adapter_param_count:,})')

        # TRAM Backbone（如果不冻结的话）
        if not freeze_backbone:
            backbone_lr = cfg.TRAIN.LR
            params.append({
                'params': [p for p in model.tram_backbone.parameters() if p.requires_grad],
                'lr': backbone_lr
            })
            print(f'TRAM Backbone lr: {backbone_lr}')

        optimizer = torch.optim.AdamW(params, weight_decay=cfg.TRAIN.WD)
        print(f'使用多学习率优化器，组件数: {len(params)}')
        
        # === 添加优化器验证 ===
        print("\n=== VGGT优化器参数组详细验证 ===")
        total_optimizer_params = 0
        for i, param_group in enumerate(optimizer.param_groups):
            group_params = sum(p.numel() for p in param_group['params'])
            total_optimizer_params += group_params
            print(f"参数组 {i}: lr={param_group['lr']}, 参数数量={group_params:,}")

        print(f"优化器总参数: {total_optimizer_params:,}")
        model_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数: {model_trainable:,}")
        print(f"参数数量匹配: {total_optimizer_params == model_trainable}")
        print(f"优化器创建后显存: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        
    else:
        # 单一学习率的fallback
        optimizer = torch.optim.AdamW(
            params=[p for p in model.parameters() if p.requires_grad],
            lr=cfg.TRAIN.LR, 
            weight_decay=cfg.TRAIN.WD
        )
        print(f'使用单一学习率: {cfg.TRAIN.LR}')

    # ========= Start Training ========= #
    # 使用VGGT Trainer，保持原始接口
    VGGTTrainer(
        cfg=cfg,
        data_loaders=data_loaders,  # 原始数据加载器
        model=model,               # VGGT模型
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        lr_scheduler=None,
    ).train()


if __name__ == '__main__':
    cfg = parse_args()
    cfg = prepare_output_dir(cfg)
    
    main(cfg)
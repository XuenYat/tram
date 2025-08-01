#!/usr/bin/env python3
"""
VGGT Trainer - 继承原始Trainer，添加VGGT支持
"""

import torch
import logging
import torch.nn.functional as F
from tqdm import tqdm
from lib.trainer import Trainer
from lib.core.dataset_vggt_adapter import create_vggt_dataloader_wrapper

logger = logging.getLogger(__name__)


class VGGTTrainer(Trainer):
    """
    VGGT训练器 - 继承原始Trainer，只添加VGGT相机参数支持
    """
    
    def __init__(self, cfg, data_loaders, model, criterion, optimizer, writer=None, lr_scheduler=None):
        # 应用VGGT数据适配器
        vggt_data_loaders = self.apply_vggt_adapter(data_loaders)
        
        # 调用原始Trainer的初始化
        super().__init__(
            cfg=cfg,
            data_loaders=vggt_data_loaders,
            model=model, 
            criterion=criterion,
            optimizer=optimizer,
            writer=writer,
            lr_scheduler=lr_scheduler
        )
        
        logger.info("✅ VGGT Trainer初始化完成")
    
    def apply_vggt_adapter(self, data_loaders):
        """
        对数据加载器应用VGGT适配器
        
        Args:
            data_loaders: 原始数据加载器 (可能是dict或list)
            
        Returns:
            包装后的数据加载器
        """
        logger.info("应用VGGT数据适配器...")
        
        if isinstance(data_loaders, dict):
            # 字典格式: {'train': train_loader, 'val': val_loader}
            wrapped_loaders = {}
            for key, loader in data_loaders.items():
                wrapped_loaders[key] = create_vggt_dataloader_wrapper(
                    loader, 
                    enable_synthesis=True
                )
            logger.info(f"包装了 {len(wrapped_loaders)} 个数据加载器")
            return wrapped_loaders
            
        elif isinstance(data_loaders, (list, tuple)):
            # 列表格式: [train_loader, val_loader]
            wrapped_loaders = []
            for i, loader in enumerate(data_loaders):
                wrapped_loader = create_vggt_dataloader_wrapper(
                    loader,
                    enable_synthesis=True
                )
                wrapped_loaders.append(wrapped_loader)
            logger.info(f"包装了 {len(wrapped_loaders)} 个数据加载器")
            return wrapped_loaders
            
        else:
            # 单个加载器
            wrapped_loader = create_vggt_dataloader_wrapper(
                data_loaders,
                enable_synthesis=True
            )
            logger.info("包装了单个数据加载器")
            return wrapped_loader
    
    def train_one_epoch(self):
        """
        基于原始Trainer，只添加VGGT相机参数处理
        """
        self.model.train()
        self.model.freeze_modules()
        update_iter = self.cfg.TRAIN.UPDATE_ITER
        crop_size = self.model.crop_size

        for i, batch in enumerate(tqdm(self.train_loader, desc="Computing batch")):
            # === 原始batch处理（完全保持不变）===
            batch = {k: v.to(self.device).flatten(0, 1) for k, v in batch.items() if type(v)==torch.Tensor}
            batch['beta_weight'] = self.cfg.TRAIN.SMPL_BETA
            batch['smpl'] = self.model.smpl

            # === 原始前向传播（完全保持不变）===
            out, iter_preds = self.model(batch, iters=update_iter)
            
            # === VGGT扩展：添加相机参数（新增）===
            if 'pred_camera' in out:
                batch['pred_camera_params'] = out['pred_camera']
            
            # === 原始损失计算（完全保持不变）===
            try:
                batch['pred_rotmat_0'] = out['pred_rotmat_0']
            except Exception:
                batch['pred_rotmat_0'] = None

            # Loss on full sequence - 完全按照原始逻辑
            rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds = iter_preds
            N = len(rotmat_preds)
            gamma = self.cfg.TRAIN.GAMMA
            loss = 0
            
            for j in range(N):
                batch['pred_rotmat'] = rotmat_preds[j]
                batch['pred_betas'] = shape_preds[j]
                batch['pred_cam'] = cam_preds[j]
                batch['pred_keypoints_3d'] = j3d_preds[j]
                batch['pred_keypoints_2d'] = (j2d_preds[j]-crop_size/2.) / (crop_size/2.)

                loss_j, losses = self.criterion(batch)
                loss += gamma**(N-j-1) * loss_j
                
            loss *= self.cfg.TRAIN.LOSS_SCALE

            # === 原始反向传播（完全保持不变）===
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.cfg.TRAIN.CLIP_GRADIENT == True:
                self.clip_gradient_norm(self.model, max_norm=self.cfg.TRAIN.CLIP_NORM)

            self.optimizer.step()
            
            # === 原始状态更新（完全保持不变）===
            self.global_step += 1
            self.loss_meter.update(losses)
            self.lr_scheduler.step()

            # === VGGT扩展：相机验证（新增）===
            if self.global_step % 500 == 0:
                try:
                    self._validate_vggt_camera_effect(batch, crop_size)
                except Exception as e:
                    logger.warning(f"VGGT相机验证失败: {e}")

            # === 原始验证调用（完全保持不变）===
            self.check_and_validate(i)

            # === 原始退出条件（完全保持不变）===
            if self.should_break():
                break

        return
    
    def _validate_vggt_camera_effect(self, batch, crop_size):
        """
        验证VGGT相机效果的辅助方法
        """
        with torch.no_grad():
            # 检查必要的数据是否存在
            if 'pred_keypoints_2d' not in batch or 'pred_keypoints_3d' not in batch:
                logger.warning("缺少关键点数据，跳过VGGT相机验证")
                return
            
            if 'pred_cam' not in batch or 'keypoints' not in batch:
                logger.warning("缺少相机或GT数据，跳过VGGT相机验证")
                return
            
            # 准备投影参数
            batch_size = batch['img'].shape[0]
            center = torch.tensor([[crop_size/2, crop_size/2]]).repeat(batch_size, 1).to(self.device)
            scale = torch.ones(batch_size).to(self.device) 
            img_focal = torch.ones(batch_size).to(self.device) * 1000.0
            img_center = torch.tensor([[crop_size/2, crop_size/2]]).repeat(batch_size, 1).to(self.device)
            
            # 使用VGGT相机的投影
            pred_2d_vggt = batch['pred_keypoints_2d']
            
            # 使用随机相机的投影（作为对比）
            random_cam = torch.randn_like(batch['pred_cam'])
            try:
                pred_2d_random = self.model.project(
                    batch['pred_keypoints_3d'], 
                    random_cam,
                    center, scale, img_focal, img_center
                )
            except Exception as e:
                logger.warning(f"随机相机投影失败: {e}")
                return
            
            # 计算投影误差
            gt_2d = batch['keypoints'][:, :, :2]
            
            # 确保维度匹配
            if pred_2d_vggt.shape != gt_2d.shape:
                logger.warning(f"维度不匹配: pred={pred_2d_vggt.shape}, gt={gt_2d.shape}")
                return
            
            error_vggt = F.mse_loss(pred_2d_vggt, gt_2d)
            error_random = F.mse_loss(pred_2d_random, gt_2d)
            
            logger.info(f"2D投影误差 - VGGT相机: {error_vggt:.4f}, 随机相机: {error_random:.4f}")
            
            if error_vggt < error_random:
                improvement = ((error_random - error_vggt) / error_random * 100).item()
                logger.info(f"✅ VGGT相机指导生效！改善了 {improvement:.1f}%")
            else:
                degradation = ((error_vggt - error_random) / error_random * 100).item()
                logger.info(f"⚠️ VGGT相机效果待提升，比随机相机差 {degradation:.1f}%")
            
            # 记录到tensorboard
            if hasattr(self, 'writer'):
                self.writer.add_scalar('VGGT/camera_error_vggt', error_vggt, self.global_step)
                self.writer.add_scalar('VGGT/camera_error_random', error_random, self.global_step)
                self.writer.add_scalar('VGGT/camera_improvement', 
                                     (error_random - error_vggt) / error_random, self.global_step)
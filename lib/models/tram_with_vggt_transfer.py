import torch
import torch.nn as nn
import sys
import os
import einops
from typing import Dict, Tuple, Optional

# 添加VGGT路径
sys.path.append('thirdparty/vggt')

# 导入VGGT
from vggt.models.vggt import VGGT
from lib.models.hmr_vimo import HMR_VIMO

class VGGTTransferLearning:
    """VGGT迁移学习工具类（基于真实结构）"""
    
    def __init__(self, model_path: str = "~/.cache/torch/hub/checkpoints/model.pt"):
        self.model_path = os.path.expanduser(model_path)
        self.output_dir = "/workspace/tram/data/pretrain"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def extract_and_save_weights(self, save_dir="./vggt_weights") -> Tuple[str, str, str]:
        """提取VGGT的aggregator和camera_head权重"""
        
        print("正在加载VGGT预训练模型...")
        model = VGGT()
        
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print(f"✅ 成功加载预训练权重: {self.model_path}")
        else:
            print(f"⚠️  预训练文件不存在: {self.model_path}，使用随机初始化权重")
            state_dict = model.state_dict()
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 权重分类 - 基于实际的VGGT结构
        aggregator_weights = {}  # backbone特征提取器
        camera_weights = {}      # 相机参数预测器
        other_weights = {}       # 其他组件
        
        print("\n分析VGGT模型权重...")
        
        for key, value in state_dict.items():
            if key.startswith('aggregator.'):
                # aggregator是主要的特征提取器（backbone）
                aggregator_weights[key] = value
                #print(f"Aggregator: {key} {list(value.shape)}")
                
            elif key.startswith('camera_head.'):
                # camera_head是相机参数预测器
                camera_weights[key] = value
                #print(f"Camera: {key} {list(value.shape)}")
                
            else:
                # 其他组件（depth_head, point_head, track_head等）
                other_weights[key] = value
                #print(f"Other: {key} {list(value.shape)}")
        
        # 保存权重文件
        aggregator_path = os.path.join(save_dir, "vggt_aggregator.pth")
        camera_path = os.path.join(save_dir, "vggt_camera_head.pth")
        full_path = os.path.join(save_dir, "vggt_full_model.pth")
        
        torch.save(aggregator_weights, aggregator_path)
        torch.save(camera_weights, camera_path)
        torch.save(state_dict, full_path)
        
        # 统计信息
        self._print_weight_statistics(aggregator_weights, camera_weights, other_weights)
        
        return aggregator_path, camera_path, full_path
    
    def _print_weight_statistics(self, aggregator_weights: Dict, camera_weights: Dict, other_weights: Dict):
        """打印权重统计信息"""
        aggregator_params = sum(p.numel() for p in aggregator_weights.values())
        camera_params = sum(p.numel() for p in camera_weights.values())
        other_params = sum(p.numel() for p in other_weights.values())
        total_params = aggregator_params + camera_params + other_params
        
        print(f"\n=== 权重提取完成 ===")
        print(f"Aggregator权重: {aggregator_params:,} ({aggregator_params/1e6:.1f}M) 参数")
        print(f"相机头权重: {camera_params:,} ({camera_params/1e6:.1f}M) 参数")
        print(f"其他权重: {other_params:,} ({other_params/1e6:.1f}M) 参数")
        print(f"总参数: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"权重比例: Aggregator {aggregator_params/total_params*100:.1f}%, Camera {camera_params/total_params*100:.1f}%")

class TokenAdapter(nn.Module):
    """Token数量和特征维度适配层：[B, 192, 1280] → [B, 196, 2048]"""
    def __init__(self, input_dim=1280, output_dim=2048):
        super().__init__()
        # 4个可学习的补充token
        self.additional_tokens = nn.Parameter(torch.randn(1, 4, input_dim) * 0.02)
        
        # 特征维度映射：1280 → 2048
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(self, x): 
        # x: [B, 192, 1280]
        B, T, D = x.shape
        assert T == 192, f"期望192个tokens，但得到{T}个"
        assert D == 1280, f"期望1280维特征，但得到{D}维"
        
        # 1. 添加4个tokens：192 → 196
        additional_tokens = self.additional_tokens.expand(B, -1, -1)  # [B, 4, 1280]
        x_expanded = torch.cat([x, additional_tokens], dim=1)  # [B, 196, 1280]
        
        # 2. 特征维度映射：1280 → 2048
        output = self.feature_projection(x_expanded)  # [B, 196, 2048]
        
        return output

class TRAMWithVGGTTransfer(nn.Module):
    """集成VGGT迁移学习的TRAM模型（基于真实结构）"""
    
    def __init__(self,
                backbone_path: str,
                camera_path: str,
                cfg,  # 需要添加cfg参数
                freeze_backbone: bool = True,
                camera_fine_tune: bool = True):
        super().__init__()
        
        from lib.models.smpl import SMPL
        self.smpl = SMPL()
        self.crop_size = 256
        self.seq_len = 16
        
        self.freeze_backbone = freeze_backbone
        self.camera_fine_tune = camera_fine_tune
        
        # === 核心修改：使用TRAM backbone替代VGGT aggregator ===
        # 1. 加载原始TRAM模型来获取强大的backbone
        
        try:
            # 1. 创建TRAM模型
            tram_model = HMR_VIMO(cfg=cfg)
            print("✅ TRAM模型创建成功")

            # === 添加检查2: 配置检查 ===
            print("=== 检查配置 ===")
            print(f"cfg.MODEL.ST_MODULE: {cfg.MODEL.ST_MODULE}")
            print(f"cfg.MODEL.MOTION_MODULE: {cfg.MODEL.MOTION_MODULE}")
            print(f"cfg.MODEL.ST_HDIM: {cfg.MODEL.ST_HDIM}")
            print(f"cfg.MODEL.MOTION_HDIM: {cfg.MODEL.MOTION_HDIM}")
            print(f"cfg.MODEL.ST_NLAYER: {cfg.MODEL.ST_NLAYER}")
            print(f"cfg.MODEL.MOTION_NLAYER: {cfg.MODEL.MOTION_NLAYER}")

            # 2. 强制加载预训练权重
            if hasattr(cfg.MODEL, 'CHECKPOINT') and cfg.MODEL.CHECKPOINT:
                checkpoint_path = cfg.MODEL.CHECKPOINT
                print(f"🔄 强制加载TRAM权重: {checkpoint_path}")

                # === 添加检查3: checkpoint内容检查 ===
                print("=== 检查checkpoint内容 ===")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    
                    st_keys = [k for k in state_dict.keys() if 'st_module' in k]
                    motion_keys = [k for k in state_dict.keys() if 'motion_module' in k]
                    
                    print(f"ST module keys: {len(st_keys)}")
                    print(f"Motion module keys: {len(motion_keys)}")
                    
                    if len(st_keys) > 0:
                        print("前几个ST keys:", st_keys[:5])
                    if len(motion_keys) > 0:
                        print("前几个Motion keys:", motion_keys[:5])

                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    
                    # 加载权重到TRAM模型
                    missing_keys, unexpected_keys = tram_model.load_state_dict(
                        state_dict, strict=False
                    )
                    print(f"✅ TRAM权重加载完成:")
                    print(f"   - Missing keys: {len(missing_keys)}")
                    print(f"   - Unexpected keys: {len(unexpected_keys)}")

                    # === 添加检查1: 模块继承检查 ===
                    print("=== 检查原始TRAM模型结构 ===")
                    print(f"原始TRAM总参数: {sum(p.numel() for p in tram_model.parameters()):,}")

                    # 详细检查每个组件
                    components_to_check = ['backbone', 'st_module', 'motion_module', 'smpl_head']
                    for comp_name in components_to_check:
                        if hasattr(tram_model, comp_name):
                            comp = getattr(tram_model, comp_name)
                            if comp is not None:
                                params = sum(p.numel() for p in comp.parameters())
                                print(f"{comp_name}: {params:,} 参数")
                            else:
                                print(f"{comp_name}: None")
                        else:
                            print(f"{comp_name}: 不存在")

                    if len(missing_keys) > 0:
                        print(f"   - 部分missing keys: {missing_keys[:5]}...")
            
            # 3. 现在测试backbone（应该可以工作了）
            test_input = torch.randn(1, 3, 256, 192)
            with torch.no_grad():
                test_output = tram_model.backbone(test_input)
                print(f"✅ 加载权重后backbone测试成功: {test_output.shape}")
            
            self.tram_backbone = tram_model.backbone
            self.tram_model = tram_model
            
        except Exception as e:
            print(f"❌ TRAM backbone仍然失败: {e}")
            print("使用备用ResNet backbone")
        

        print("=== 调试权重加载 ===")
        print("Missing keys详细信息:")
        for i, key in enumerate(missing_keys):
            if i < 20:  # 打印前20个
                print(f"  {key}")
            elif i == 20:
                print("  ...")
                break

        # 检查backbone权重是否真的加载了
        print("检查backbone第一层权重:")
        first_weight = None
        for name, param in tram_model.backbone.named_parameters():
            if 'weight' in name:
                first_weight = param
                print(f"  {name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                break

        # 检查checkpoint中是否有backbone权重
        backbone_keys_in_checkpoint = [k for k in state_dict.keys() if k.startswith('backbone.')]
        print(f"Checkpoint中backbone相关keys数量: {len(backbone_keys_in_checkpoint)}")
        if len(backbone_keys_in_checkpoint) > 0:
            print("前5个backbone keys:")
            for key in backbone_keys_in_checkpoint[:5]:
                print(f"  {key}")

        # 2. 只加载VGGT的相机头（保持专业相机预测能力）
        self.vggt_camera_head = self._load_camera_head(camera_path)

        #特征适配：TRAM输出(192 tokens) → VGGT期望格式(196 tokens)
        self.feature_adapter = self._init_feature_adapter()

        if hasattr(tram_model, 'smpl_head'):
            self.tram_smpl_head = tram_model.smpl_head
            print("✅ 使用原始TRAM SMPL head")
        else:
            raise ValueError("TRAM模型没有smpl_head组件")
        
        # 在模型初始化中添加检查：
        if hasattr(tram_model, 'st_module') and tram_model.st_module is not None:
            self.st_module = tram_model.st_module
            print("✅ ST Module已继承")
        else:
            self.st_module = None
            print("⚠️ ST Module未找到")

        if hasattr(tram_model, 'motion_module') and tram_model.motion_module is not None:
            self.motion_module = tram_model.motion_module  
            print("✅ Motion Module已继承")
        else:
            self.motion_module = None
            print("⚠️ Motion Module未找到")

        # 4. 设置混合架构的训练策略
        self._setup_hybrid_training_strategy()
        
        print("✅ 混合TRAM-VGGT模型初始化完成")

    def _setup_hybrid_training_strategy(self):
        """设置混合架构的训练策略"""
        # 设置TRAM backbone
        if self.freeze_backbone:
            for param in self.tram_backbone.parameters():
                param.requires_grad = False
            print("✓ TRAM Backbone已冻结")
        else:
            for param in self.tram_backbone.parameters():
                param.requires_grad = True
            print("✓ TRAM Backbone允许训练")
        
        # 设置VGGT相机头微调
        if self.camera_fine_tune:
            for param in self.vggt_camera_head.parameters():
                param.requires_grad = True
            print("✓ VGGT Camera Head允许微调")
        else:
            for param in self.vggt_camera_head.parameters():
                param.requires_grad = False
            print("✓ VGGT Camera Head已冻结")
        
        # TRAM SMPL头默认可训练
        for param in self.tram_smpl_head.parameters():
            param.requires_grad = True
        print("✓ TRAM SMPL Head允许训练")

        # 添加Motion和ST模块的训练设置
        if hasattr(self, 'motion_module') and self.motion_module is not None:
            for param in self.motion_module.parameters():
                param.requires_grad = True
            print("✓ Motion Module允许训练")
        
        if hasattr(self, 'st_module') and self.st_module is not None:
            for param in self.st_module.parameters():
                param.requires_grad = True  
            print("✓ ST Module允许训练")

        for param in self.feature_adapter.parameters():
            param.requires_grad = True
        print("✓ 特征适配层允许训练")

    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):
        # 直接复制HMR_VIMO的project方法
        return HMR_VIMO.project(self, points, pred_cam, center, scale, img_focal, img_center, return_full)
    
    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        return HMR_VIMO.get_trans(self, pred_cam, center, scale, img_focal, img_center)
  
    def _load_camera_head(self, camera_path: str) -> nn.Module:
        """加载VGGT camera_head"""
        print(f"加载VGGT camera_head from {camera_path}")
        
        # 创建完整的VGGT模型来获取camera_head
        vggt_model = VGGT()
        camera_head = vggt_model.camera_head
        
        try:
            # 加载camera_head权重
            camera_weights = torch.load(camera_path, map_location='cpu')
            
            # 只加载camera_head相关的权重
            camera_state_dict = {}
            for key, value in camera_weights.items():
                if key.startswith('camera_head.'):
                    # 移除'camera_head.'前缀
                    new_key = key[12:]  # len('camera_head.') = 12
                    camera_state_dict[new_key] = value
            
            missing_keys, unexpected_keys = camera_head.load_state_dict(camera_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in camera_head: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in camera_head: {len(unexpected_keys)} keys")
            
            print(f"✅ Camera head权重加载完成")
            
        except Exception as e:
            print(f"⚠️  Camera head权重加载失败: {e}")
            print("使用随机初始化的权重")
        
        return camera_head

    def _init_feature_adapter(self) -> nn.Module:
        """初始化特征适配层：[B, 192, 1280] → [B, 196, 2048]"""
        return nn.Sequential(
            TokenAdapter(input_dim=1280, output_dim=2048),  # 核心适配
            nn.Dropout(0.1)                                 # 正则化
        )

    def _setup_training_strategy(self):
        """设置训练策略"""
        # 冻结aggregator
        if self.freeze_backbone:
            for param in self.vggt_aggregator.parameters():
                param.requires_grad = False
            print("✓ VGGT Aggregator已冻结")
        else:
            for param in self.vggt_aggregator.parameters():
                param.requires_grad = True
            print("✓ VGGT Aggregator允许训练")
        
        # 设置camera head微调
        if self.camera_fine_tune:
            for param in self.vggt_camera_head.parameters():
                param.requires_grad = True
            print("✓ VGGT Camera Head允许微调")
        else:
            for param in self.vggt_camera_head.parameters():
                param.requires_grad = False
            print("✓ VGGT Camera Head已冻结")
        
        # TRAM组件默认可训练
        for param in self.tram_smpl_head.parameters():
            param.requires_grad = True
        print("✓ TRAM SMPL Head允许训练")
    
    def forward(self, batch=None, images=None, iters=1, **kwargs):
        # 处理不同的输入格式
        if batch is not None and 'img' in batch:
            images = batch['img']
            center = batch['center']
            scale = batch['scale']
            img_focal = batch['img_focal']
            img_center = batch['img_center']
        elif images is not None:
            # 如果只提供images，创建默认的相机参数
            batch_size = images.size(0)
            device = images.device
            center = torch.tensor([[self.crop_size/2, self.crop_size/2]]).repeat(batch_size, 1).to(device)
            scale = torch.ones(batch_size).to(device) 
            img_focal = torch.ones(batch_size).to(device) * 1000.0
            img_center = torch.tensor([[self.crop_size/2, self.crop_size/2]]).repeat(batch_size, 1).to(device)
        else:
            raise ValueError("需要提供 batch 或 images 参数")

        # === 添加调试信息 ===
        # print(f"DEBUG: 输入图像shape: {images.shape}")

        # 如果输入是256×256，立即裁剪为256×192
        if images.shape[-2:] == (256, 256):
            # print("DEBUG: 检测到256×256输入，立即裁剪为256×192")
            images = images[:, :, :, 32:-32]  # 裁剪为256×192
            # print(f"DEBUG: 裁剪后尺寸: {images.shape}")

        batch_size = images.size(0)

        # === 现在使用正确尺寸的图像调用backbone ===
        tram_features = self.tram_backbone(images)  # 现在应该是256×192的输入

        if self.st_module is not None:
            # 检查是否满足时序条件（16的倍数）
            if batch_size % 16 == 0:
                #print(f"🔄 调用ST Module处理 {batch_size} 帧")
                
                # 使用保存的tram_model调用bbox_est
                bbox_info = self.tram_model.bbox_est(center, scale, img_focal, img_center)  # [B, 3]
                
                # 按照原始TRAM的方式处理ST模块
                bb = einops.repeat(bbox_info, 'b c -> b c h w', h=16, w=12)  # [B, 3, 16, 12]
                st_input = torch.cat([tram_features, bb], dim=1)  # [B, 1283, 16, 12]

                # 重塑为时序格式并调用ST模块
                st_input = einops.rearrange(st_input, '(b t) c h w -> (b h w) t c', t=16)
                st_output = self.st_module(st_input)  # [(B*H*W), 16, 1280]
                tram_features = einops.rearrange(st_output, '(b h w) t c -> (b t) c h w', h=16, w=12)
                
                #print("✅ ST Module处理完成")
            else:
                print(f"⚠️ 批次大小 {batch_size} 不是16的倍数，跳过ST Module")

        # print(f"DEBUG: TRAM特征shape: {tram_features.shape}")
        
        # 1.5 特征适配：TRAM输出(192 tokens) → VGGT期望格式(196 tokens)
        assert tram_features.dim() == 4, f"期望TRAM backbone输出4维特征，实际得到{tram_features.dim()}维"
        B, C, H, W = tram_features.shape
        assert H == 16 and W == 12, f"期望16×12空间布局，实际得到{H}×{W}"
        assert C == 1280, f"期望ViT-huge特征维度1280，实际得到{C}"

        # 转换为序列格式 [B, 192, 1280]
        tram_features_seq = tram_features.reshape(B, C, -1).transpose(1, 2)

        # 特征适配：192 tokens → 196 tokens, 1280 dim → 2048 dim
        adapted_features = self.feature_adapter(tram_features_seq)  # [B, 196, 2048]

        # print(f"DEBUG: 适配后特征shape: {adapted_features.shape}")

        # === 关键修改：创建专用相机token ===
        # 创建全局感知的相机token（所有图像patch的全局表示）
        global_camera_token = adapted_features.mean(dim=1, keepdim=True)  # [B, 1, 2048]

        # 重新组织为VGGT期望格式：[相机token, 图像tokens]
        vggt_tokens = torch.cat([global_camera_token, adapted_features], dim=1)  # [B, 197, 2048]

        # 添加序列维度（VGGT期望 [B, S, N, 2048] 格式）
        vggt_tokens = vggt_tokens.unsqueeze(1)  # [B, 1, 197, 2048]

        # print(f"DEBUG: VGGT tokens shape: {vggt_tokens.shape}")
        # print(f"DEBUG: 相机token测试: {vggt_tokens[:, :, 0].shape}")  # 应该是[B, 1, 2048]

        # 3. VGGT相机头处理
        # 创建24层aggregator输出（VGGT期望transformer的24层输出）
        aggregated_tokens_list = [vggt_tokens] * 24
        camera_predictions = self.vggt_camera_head(aggregated_tokens_list)

        # === 添加详细调试信息 ===
        # print(f"DEBUG: aggregated_tokens_list长度: {len(aggregated_tokens_list)}")
        # print(f"DEBUG: aggregated_tokens_list[0]的shape: {aggregated_tokens_list[0].shape}")
        # print(f"DEBUG: aggregated_tokens_list[0]的数据类型: {type(aggregated_tokens_list[0])}")

        camera_params = camera_predictions[-1]  # [B, 1, 9] 或 [B, 9]

        # print(f"DEBUG: 原始相机预测shape: {camera_params.shape}")

        # 确保维度正确
        if camera_params.dim() == 3 and camera_params.size(1) == 1:
            camera_params = camera_params.squeeze(1)  # [B, 9]
        elif camera_params.dim() != 2:
            raise ValueError(f"意外的相机参数维度: {camera_params.shape}")

        # print(f"DEBUG: 最终相机参数shape: {camera_params.shape}")
        
        # 4. TRAM SMPL处理 - 使用原始TRAM方式
        # print(f"DEBUG: TRAM特征原始shape: {tram_features.shape}")  # [B, 1280, 16, 12]

        # 直接使用4维特征，就像原始TRAM一样
        pred_pose, pred_shape, pred_cam_tram_original = self.tram_smpl_head(tram_features)

        # === 相机参数对比调试 ===
        pred_cam_tram = self.convert_vggt_to_tram_camera(camera_params)

        # print(f"🔍 相机参数对比:")
        # print(f"  原始TRAM相机: mean={pred_cam_tram_original.mean(dim=0)}, std={pred_cam_tram_original.std(dim=0)}")
        # print(f"  VGGT转换相机: mean={pred_cam_tram.mean(dim=0)}, std={pred_cam_tram.std(dim=0)}")

        # === 关键修复：添加Motion模块调用 ===
        if self.motion_module is not None:
            if batch_size % 16 == 0:
                #print(f"🔄 调用Motion Module处理 {batch_size} 帧的pose")
                
                # 使用保存的tram_model调用bbox_est
                bbox_info = self.tram_model.bbox_est(center, scale, img_focal, img_center)  # [B, 3]
                bb = einops.rearrange(bbox_info, '(b t) c -> b t c', t=16)  # [B/16, 16, 3]
                pred_pose_seq = einops.rearrange(pred_pose, '(b t) c -> b t c', t=16)  # [B/16, 16, 144]
                
                motion_input = torch.cat([pred_pose_seq, bb], dim=2)  # [B/16, 16, 147]
                motion_output = self.motion_module(motion_input)  # [B/16, 16, 144]
                pred_pose = einops.rearrange(motion_output, 'b t c -> (b t) c')  # [B, 144]
                
                #print("✅ Motion Module处理完成")
            else:
                print(f"⚠️ 批次大小 {batch_size} 不是16的倍数，跳过Motion Module")

        # print(f"DEBUG: 原始TRAM SMPL输出:")
        # print(f"  - pred_pose shape: {pred_pose.shape}")
        # print(f"  - pred_shape shape: {pred_shape.shape}")
        # print(f"  - pred_cam_tram shape: {pred_cam_tram_original.shape}")

        # 注意：我们忽略pred_cam_tram_original，使用VGGT的相机参数

        # 6. 从SMPL参数生成3D关键点
        from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
        pred_rotmat = rot6d_to_rotmat(pred_pose).reshape(batch_size, 24, 3, 3)

        smpl_output = self.smpl(
            global_orient=pred_rotmat[:, [0]],
            body_pose=pred_rotmat[:, 1:],
            betas=pred_shape,
            pose2rot=False
        )
        pred_keypoints_3d = smpl_output.joints  # [B, 49, 3]
        
        # 7. 关键步骤：将VGGT相机参数转换为TRAM格式用于投影
        pred_cam_tram = self.convert_vggt_to_tram_camera(camera_params)  # [B, 3]
        
        # 8. 使用VGGT指导的相机参数进行2D投影
        center = torch.tensor([[self.crop_size/2, self.crop_size/2]]).repeat(batch_size, 1).to(images.device)
        scale = torch.ones(batch_size).to(images.device) 
        img_focal = torch.ones(batch_size).to(images.device) * 1000.0
        img_center = torch.tensor([[self.crop_size/2, self.crop_size/2]]).repeat(batch_size, 1).to(images.device)
        
        pred_keypoints_2d = self.project(
            pred_keypoints_3d,    # 3D关键点
            pred_cam_tram,        # 使用VGGT指导的相机参数！
            center, scale, img_focal, img_center
        )

        # 9. 输出结果
        output = {
            'pred_pose': pred_pose,
            'pred_shape': pred_shape,
            'pred_cam': pred_cam_tram,        # 输出VGGT指导的相机参数（TRAM格式）
            'pred_camera': camera_params,     # 原始VGGT相机参数（9维格式）
            'pred_keypoints_2d': pred_keypoints_2d,  # 基于VGGT相机的2D投影！
            'pred_keypoints_3d': pred_keypoints_3d,
            'pred_rotmat': pred_rotmat,
            'pred_rotmat_0': pred_rotmat,
            'features': tram_features,    
        }

        # 10. 兼容原始TRAM的迭代训练接口
        iter_preds = [
            [pred_rotmat] * iters,         # rotmat_preds
            [pred_shape] * iters,          # shape_preds  
            [pred_cam_tram] * iters,       # cam_preds - 使用VGGT指导的相机参数
            [pred_keypoints_3d] * iters,   # j3d_preds
            [pred_keypoints_2d] * iters    # j2d_preds - 基于VGGT相机的投影
        ]
        
        return output, iter_preds

    def analyze_camera_parameters(self, vggt_camera, tram_camera_original, batch):
        """分析两种相机参数的真实物理含义"""
        
        # 获取图像信息
        center = batch['center']
        scale = batch['scale'] 
        img_focal = batch['img_focal']
        img_center = batch['img_center']
        
        print(f"\n🔍 === 相机参数物理含义分析 ===")
        print(f"批次大小: {len(center)}")
        print(f"图像信息 (第一个样本):")
        print(f"  center (crop中心): [{center[0][0]:.1f}, {center[0][1]:.1f}]")
        print(f"  scale (crop缩放): {scale[0]:.3f}")   
        print(f"  img_focal (焦距): {img_focal[0]:.1f}")  
        print(f"  img_center (图像中心): [{img_center[0][0]:.1f}, {img_center[0][1]:.1f}]")
        print(f"  crop_size: {self.crop_size}")
        
        # 分析VGGT参数 (取前3个样本)
        print(f"\nVGGT相机参数分析:")
        for i in range(min(3, len(vggt_camera))):
            vggt_trans = vggt_camera[i, :3]
            vggt_quat = vggt_camera[i, 3:7]
            vggt_fov = vggt_camera[i, 7:9]
            
            print(f"  样本{i}: translation=[{vggt_trans[0]:.6f}, {vggt_trans[1]:.6f}, {vggt_trans[2]:.3f}]")
            print(f"         quaternion=[{vggt_quat[0]:.3f}, {vggt_quat[1]:.3f}, {vggt_quat[2]:.3f}, {vggt_quat[3]:.3f}]")
            print(f"         fov=[{vggt_fov[0]:.1f}°, {vggt_fov[1]:.1f}°]")
        
        # 分析TRAM参数
        print(f"\nTRAM相机参数分析:")
        for i in range(min(3, len(tram_camera_original))):
            tram_params = tram_camera_original[i]
            print(f"  样本{i}: [s={tram_params[0]:.3f}, tx={tram_params[1]:.3f}, ty={tram_params[2]:.3f}]")
        
        # === 关键分析：计算TRAM的实际3D translation ===
        print(f"\n🎯 3D坐标对比分析:")
        for i in range(min(3, len(vggt_camera))):
            # TRAM计算的3D translation
            tram_trans_3d = self.tram_model.get_trans(
                tram_camera_original[[i]], center[[i]], scale[[i]], 
                img_focal[[i]], img_center[[i]]
            )
            
            vggt_trans = vggt_camera[i, :3]
            tram_3d = tram_trans_3d[0, 0]  # [tx, ty, tz]
            
            print(f"  样本{i}:")
            print(f"    VGGT 3D translation: [{vggt_trans[0]:.6f}, {vggt_trans[1]:.6f}, {vggt_trans[2]:.3f}]")
            print(f"    TRAM 3D translation: [{tram_3d[0]:.6f}, {tram_3d[1]:.6f}, {tram_3d[2]:.3f}]")
            print(f"    深度比值 (VGGT/TRAM): {vggt_trans[2]/tram_3d[2]:.3f}")
            print(f"    XY差异: dx={abs(vggt_trans[0]-tram_3d[0]):.6f}, dy={abs(vggt_trans[1]-tram_3d[1]):.6f}")
        
        # === 统计分析 ===
        print(f"\n📊 统计对比:")
        
        # VGGT统计
        vggt_trans_all = vggt_camera[:, :3]
        print(f"VGGT translation统计:")
        print(f"  X: mean={vggt_trans_all[:, 0].mean():.6f}, std={vggt_trans_all[:, 0].std():.6f}")
        print(f"  Y: mean={vggt_trans_all[:, 1].mean():.6f}, std={vggt_trans_all[:, 1].std():.6f}")
        print(f"  Z: mean={vggt_trans_all[:, 2].mean():.3f}, std={vggt_trans_all[:, 2].std():.3f}")
        
        # TRAM统计
        print(f"TRAM参数统计:")
        print(f"  s: mean={tram_camera_original[:, 0].mean():.3f}, std={tram_camera_original[:, 0].std():.3f}")
        print(f"  tx: mean={tram_camera_original[:, 1].mean():.3f}, std={tram_camera_original[:, 1].std():.3f}")
        print(f"  ty: mean={tram_camera_original[:, 2].mean():.3f}, std={tram_camera_original[:, 2].std():.3f}")
        
        # TRAM 3D translation统计
        all_tram_3d = []
        for i in range(len(tram_camera_original)):
            tram_3d = self.tram_model.get_trans(
                tram_camera_original[[i]], center[[i]], scale[[i]], 
                img_focal[[i]], img_center[[i]]
            )
            all_tram_3d.append(tram_3d[0, 0])
        
        all_tram_3d = torch.stack(all_tram_3d)
        print(f"TRAM 3D translation统计:")
        print(f"  X: mean={all_tram_3d[:, 0].mean():.6f}, std={all_tram_3d[:, 0].std():.6f}")
        print(f"  Y: mean={all_tram_3d[:, 1].mean():.6f}, std={all_tram_3d[:, 1].std():.6f}")
        print(f"  Z: mean={all_tram_3d[:, 2].mean():.3f}, std={all_tram_3d[:, 2].std():.3f}")
        
        print(f"=== 分析完成 ===\n")

    def convert_vggt_to_tram_camera(self, vggt_camera):
        """增强版固定分布 - 带详细调试信息"""
        batch_size = vggt_camera.shape[0]
        device = vggt_camera.device
        
        # print(f"🔍 === 增强版固定分布转换调试 ===")
        # print(f"批次大小: {batch_size}")
        
        # === 1. 分析输入VGGT参数 ===
        vggt_translation = vggt_camera[:, :3]
        vggt_quaternion = vggt_camera[:, 3:7]
        vggt_fov = vggt_camera[:, 7:9]
        
        vggt_tx, vggt_ty, vggt_tz = vggt_translation.unbind(-1)
        
        # print(f"📊 输入VGGT参数统计:")
        # print(f"  translation:")
        # print(f"    tx: [{vggt_tx.min():.6f}, {vggt_tx.max():.6f}], mean={vggt_tx.mean():.6f}±{vggt_tx.std():.6f}")
        # print(f"    ty: [{vggt_ty.min():.6f}, {vggt_ty.max():.6f}], mean={vggt_ty.mean():.6f}±{vggt_ty.std():.6f}")
        # print(f"    tz: [{vggt_tz.min():.3f}, {vggt_tz.max():.3f}], mean={vggt_tz.mean():.3f}±{vggt_tz.std():.3f}")
        # print(f"  quaternion: mean={vggt_quaternion.mean(dim=0)}")
        # print(f"  fov: [{torch.rad2deg(vggt_fov).min():.1f}°, {torch.rad2deg(vggt_fov).max():.1f}°], mean={torch.rad2deg(vggt_fov).mean():.1f}°")
        
        # === 2. 生成基础分布 ===
        s_base = torch.normal(0.86, 0.17, (batch_size,), device=device)
        tx_base = torch.normal(0.19, 0.13, (batch_size,), device=device)
        ty_base = torch.normal(0.39, 0.33, (batch_size,), device=device)
        
        # print(f"📊 基础分布生成:")
        # print(f"  s_base: [{s_base.min():.3f}, {s_base.max():.3f}], mean={s_base.mean():.3f}±{s_base.std():.3f}")
        # print(f"  tx_base: [{tx_base.min():.3f}, {tx_base.max():.3f}], mean={tx_base.mean():.3f}±{tx_base.std():.3f}")
        # print(f"  ty_base: [{ty_base.min():.3f}, {ty_base.max():.3f}], mean={ty_base.mean():.3f}±{ty_base.std():.3f}")
        
        # === 3. 计算VGGT影响 ===
        depth_influence_weight = 0.05
        xy_influence_weight = 0.02
        
        vggt_tz_centered = vggt_tz - vggt_tz.mean()
        vggt_tx_centered = vggt_tx - vggt_tx.mean()
        vggt_ty_centered = vggt_ty - vggt_ty.mean()
        
        depth_influence = vggt_tz_centered * depth_influence_weight
        tx_influence = vggt_tx_centered * xy_influence_weight
        ty_influence = vggt_ty_centered * xy_influence_weight
        
        # print(f"📊 VGGT影响计算:")
        # print(f"  中心化后的VGGT变化:")
        # print(f"    tz_centered: [{vggt_tz_centered.min():.3f}, {vggt_tz_centered.max():.3f}], std={vggt_tz_centered.std():.3f}")
        # print(f"    tx_centered: [{vggt_tx_centered.min():.6f}, {vggt_tx_centered.max():.6f}], std={vggt_tx_centered.std():.6f}")
        # print(f"    ty_centered: [{vggt_ty_centered.min():.6f}, {vggt_ty_centered.max():.6f}], std={vggt_ty_centered.std():.6f}")
        # print(f"  影响量:")
        # print(f"    depth_influence: [{depth_influence.min():.4f}, {depth_influence.max():.4f}], mean={depth_influence.mean():.6f}")
        # print(f"    tx_influence: [{tx_influence.min():.6f}, {tx_influence.max():.6f}], mean={tx_influence.mean():.6f}")
        # print(f"    ty_influence: [{ty_influence.min():.6f}, {ty_influence.max():.6f}], mean={ty_influence.mean():.6f}")
        
        # === 4. 合成最终参数 ===
        s_final = s_base + depth_influence
        tx_final = tx_base + tx_influence
        ty_final = ty_base + ty_influence
        
        # print(f"📊 合成前参数:")
        # print(f"  s_final: [{s_final.min():.3f}, {s_final.max():.3f}], mean={s_final.mean():.3f}±{s_final.std():.3f}")
        # print(f"  tx_final: [{tx_final.min():.3f}, {tx_final.max():.3f}], mean={tx_final.mean():.3f}±{tx_final.std():.3f}")
        # print(f"  ty_final: [{ty_final.min():.3f}, {ty_final.max():.3f}], mean={ty_final.mean():.3f}±{ty_final.std():.3f}")
        
        # === 5. Clamp操作 ===
        s_clamped = s_final.clamp(0.3, 1.5)
        tx_clamped = tx_final.clamp(-0.5, 0.8)
        ty_clamped = ty_final.clamp(0.0, 0.8)
        
        # 统计clamp影响
        s_clamp_count = (s_final != s_clamped).sum().item()
        tx_clamp_count = (tx_final != tx_clamped).sum().item()
        ty_clamp_count = (ty_final != ty_clamped).sum().item()
        
        # print(f"📊 Clamp操作:")
        # print(f"  s clamp: {s_clamp_count}/{batch_size} 个值被clamp")
        # print(f"  tx clamp: {tx_clamp_count}/{batch_size} 个值被clamp")
        # print(f"  ty clamp: {ty_clamp_count}/{batch_size} 个值被clamp")
        
        # === 6. 最终结果 ===
        s, tx, ty = s_clamped, tx_clamped, ty_clamped
        
        # print(f"🎯 最终转换结果:")
        # print(f"  s: [{s.min():.3f}, {s.max():.3f}], mean={s.mean():.3f}±{s.std():.3f}")
        # print(f"  tx: [{tx.min():.3f}, {tx.max():.3f}], mean={tx.mean():.3f}±{tx.std():.3f}")
        # print(f"  ty: [{ty.min():.3f}, {ty.max():.3f}], mean={ty.mean():.3f}±{ty.std():.3f}")
        
        # === 7. 与目标对比 ===
        target_s, target_tx, target_ty = 0.86, 0.19, 0.39
        target_s_std, target_tx_std, target_ty_std = 0.17, 0.13, 0.33
        
        # print(f"📈 与目标TRAM分布对比:")
        # print(f"  目标: s={target_s}±{target_s_std}, tx={target_tx}±{target_tx_std}, ty={target_ty}±{target_ty_std}")
        # print(f"  实际: s={s.mean():.2f}±{s.std():.2f}, tx={tx.mean():.2f}±{tx.std():.2f}, ty={ty.mean():.2f}±{ty.std():.2f}")
        # print(f"  均值误差: Δs={abs(s.mean().item()-target_s):.3f}, Δtx={abs(tx.mean().item()-target_tx):.3f}, Δty={abs(ty.mean().item()-target_ty):.3f}")
        # print(f"  标准差比较: s_std比例={s.std().item()/target_s_std:.2f}, tx_std比例={tx.std().item()/target_tx_std:.2f}, ty_std比例={ty.std().item()/target_ty_std:.2f}")
        
        # === 8. VGGT信息利用率 ===
        vggt_contribution_s = depth_influence.abs().mean() / s.abs().mean() * 100
        vggt_contribution_tx = tx_influence.abs().mean() / tx.abs().mean() * 100 if tx.abs().mean() > 1e-6 else 0
        vggt_contribution_ty = ty_influence.abs().mean() / ty.abs().mean() * 100 if ty.abs().mean() > 1e-6 else 0
        
        # print(f"📊 VGGT信息利用率:")
        # print(f"  s参数中VGGT深度的贡献: {vggt_contribution_s:.2f}%")
        # print(f"  tx参数中VGGT的贡献: {vggt_contribution_tx:.2f}%")
        # print(f"  ty参数中VGGT的贡献: {vggt_contribution_ty:.2f}%")
        
        # print(f"=== 增强版转换调试完成 ===\n")
        
        return torch.stack([s, tx, ty], dim=1)

    def get_trainable_parameters(self):
        """获取可训练参数统计"""
        trainable_params = 0
        total_params = 0
        
        components = {
            'TRAM Backbone': self.tram_backbone,
            'Feature Adapter': self.feature_adapter,  # 添加这行
            'VGGT Camera Head': self.vggt_camera_head,
            'TRAM SMPL Head': self.tram_smpl_head
        }
        
        print(f"\n=== 混合架构参数统计 ===")
        for name, module in components.items():
            module_total = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            total_params += module_total
            trainable_params += module_trainable
            
            print(f"{name}: {module_total:,} ({module_total/1e6:.1f}M) 总参数, "
                f"{module_trainable:,} ({module_trainable/1e6:.1f}M) 可训练")
        
        print(f"\n总计:")
        print(f"总参数: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"冻结参数: {total_params-trainable_params:,} ({(total_params-trainable_params)/1e6:.1f}M)")
        print(f"可训练比例: {trainable_params/total_params*100:.1f}%")
        
        return trainable_params, total_params
    
    def set_camera_fine_tune(self, enable: bool):
        """动态设置相机头微调状态"""
        for param in self.vggt_camera_head.parameters():
            param.requires_grad = enable
        self.camera_fine_tune = enable
        print(f"相机头微调状态更新为: {enable}")
    
    def set_backbone_freeze(self, freeze: bool):
        """动态设置骨干网络冻结状态"""
        for param in self.vggt_aggregator.parameters():
            param.requires_grad = not freeze
        self.freeze_backbone = freeze
        print(f"骨干网络冻结状态更新为: {freeze}")


# 测试代码
if __name__ == "__main__":
    print("测试基于真实结构的VGGT迁移学习模型...")
    
    # 测试权重提取
    tool = VGGTTransferLearning()
    backbone_path, camera_path, _ = tool.extract_and_save_weights()
    
    # 测试模型
    model = TRAMWithVGGTTransfer(backbone_path, camera_path)
    
    # 测试前向传播
    test_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(test_input)
    
    print("输出键:", list(output.keys()))
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
    
    # 参数统计
    model.get_trainable_parameters()
    
    print("模块测试完成！")
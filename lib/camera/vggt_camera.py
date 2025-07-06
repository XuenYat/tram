import torch
import numpy as np
import sys
import os
from glob import glob
from pathlib import Path
import cv2
try:
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, 6DoF pose conversion will be limited")

# 直接使用相对导入path_utils
from ..utils.path_utils import ensure_vggt_in_path
ensure_vggt_in_path()

# VGGT相关导入
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# 尝试导入几何工具函数
try:
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    HAS_VGGT_GEOMETRY = True
except ImportError:
    try:
        from vggt.utils.misc import unproject_depth_map_to_point_map
        HAS_VGGT_GEOMETRY = True
    except ImportError:
        HAS_VGGT_GEOMETRY = False
        print("Warning: VGGT geometry utils not found, will use simple unprojection")


class VGGTCameraEstimator:
    """VGGT相机轨迹估计封装类"""
    
    def __init__(self, model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.model = None
        self.model_path = model_path
        
    def load_model(self):
        """加载VGGT模型 - 使用本地文件"""
        if self.model is None:
            print("🚀 Loading VGGT model from local checkpoint...")
            self.model = VGGT()
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✅ VGGT model loaded from {self.model_path} on {self.device}")
    
    def predict_poses_with_intrinsics(self, imgfiles, batch_size=4):
        """预测相机poses和内参矩阵（用于Bundle Adjustment）"""
        self.load_model()
        
        print(f"📸 Processing {len(imgfiles)} images with VGGT for poses+intrinsics (batch_size={batch_size})...")
        
        all_predictions = []
        
        # 分批处理图像
        for i in range(0, len(imgfiles), batch_size):
            batch_files = imgfiles[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
            
            # 加载和预处理当前批次的图像
            images = load_and_preprocess_images(batch_files).to(self.device)
            
            # VGGT推理 - 预测相机头并获取pose encoding
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    # 添加batch维度
                    images_batch = images[None] if images.dim() == 4 else images
                    
                    # 运行aggregator和camera_head
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                    pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                    
                    # 同时解码得到外参和内参
                    H, W = images.shape[-2:]
                    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W), build_intrinsics=True)
                    
                    # 将结果包装成字典格式
                    batch_predictions = {
                        'pose_enc': pose_enc,
                        'extrinsics': extrinsics,
                        'intrinsics': intrinsics
                    }
            
            # 收集预测结果
            all_predictions.append(batch_predictions)
            
            # 清理当前批次的显存
            del images
            torch.cuda.empty_cache()
        
        # 合并所有批次的预测结果
        combined_predictions = self._combine_batch_predictions(all_predictions)
        
        # 提取相机poses和内参
        cam_R, cam_T = self._extract_camera_poses(combined_predictions, imgfiles)
        intrinsics = combined_predictions.get('intrinsics', None)
        
        if intrinsics is not None:
            # 确保内参矩阵在CPU上并且形状正确
            if len(intrinsics.shape) == 4:  # [batch, seq, 3, 3]
                intrinsics = intrinsics.squeeze(0)  # [seq, 3, 3]
            intrinsics = intrinsics.cpu()
            print(f"✅ Extracted per-frame intrinsics: {intrinsics.shape}")
        else:
            print("⚠️ Could not extract intrinsics from VGGT output")
        
        return cam_R, cam_T, intrinsics
    
    def _extract_camera_poses(self, predictions, imgfiles):
        """从VGGT预测中提取相机poses"""
        print("🔍 Extracting camera poses from VGGT predictions...")
        
        # 直接使用已解码的extrinsics，如果不存在则使用pose_enc
        if 'extrinsics' in predictions:
            print("✅ Using pre-decoded extrinsics from predictions")
            extrinsics = predictions['extrinsics']
            
            # 确保移除batch维度
            if len(extrinsics.shape) == 4 and extrinsics.shape[0] == 1:
                extrinsics = extrinsics.squeeze(0)  # [S, 3, 4]
            
            # 提取旋转矩阵和平移向量
            cam_R = extrinsics[:, :3, :3].float()  # (S, 3, 3)
            cam_T = extrinsics[:, :3, 3].float()   # (S, 3)
            
        elif 'pose_enc' in predictions:
            print("⚠️ Extrinsics not found, converting from pose_enc")
            pose_data = predictions['pose_enc']
            cam_R, cam_T = self._convert_pose_encoding(pose_data, imgfiles[0])
            
        else:
            raise ValueError("Neither 'extrinsics' nor 'pose_enc' found in predictions")
        
        print(f"✅ Extracted {cam_R.shape[0]} camera poses")
        return cam_R, cam_T
    
    def _convert_pose_encoding(self, pose_data, first_img_path):
        """将VGGT的pose encoding转换为旋转矩阵和平移向量"""
        
        # 确保为float32类型
        pose_data = pose_data.float()
        
        # 获取图像尺寸
        first_img = cv2.imread(first_img_path)
        H, W = first_img.shape[:2]
        image_shape = (H, W)
        
        print(f"Converting pose encoding: {pose_data.shape} -> extrinsics")
        
        # 处理VGGT输出的标准格式：[B, S, 9] -> [S, 9]
        if len(pose_data.shape) == 3 and pose_data.shape[0] == 1:
            processed_pose_data = pose_data.squeeze(0)  # [S, 9]
        elif len(pose_data.shape) == 3:
            processed_pose_data = pose_data[0]  # 取第一个batch
        elif len(pose_data.shape) == 2:
            processed_pose_data = pose_data  # 已经是 [S, 9] 格式
        else:
            raise ValueError(f"Unexpected pose data shape: {pose_data.shape}")
        
        # 验证是9D pose encoding
        if processed_pose_data.shape[-1] != 9:
            raise ValueError(f"Expected 9D pose encoding, got {processed_pose_data.shape[-1]}D")
        
        # 使用官方VGGT函数转换
        pose_encoding_with_batch = processed_pose_data.unsqueeze(0)  # [1, S, 9]
        extrinsic, _ = pose_encoding_to_extri_intri(pose_encoding_with_batch, image_shape)
        
        # 提取旋转和平移
        extrinsic = extrinsic.squeeze(0)  # [S, 3, 4] 
        cam_R = extrinsic[:, :3, :3].float()  # (S, 3, 3)
        cam_T = extrinsic[:, :3, 3].float()   # (S, 3)
        
        print(f"✅ Converted to cam_R: {cam_R.shape}, cam_T: {cam_T.shape}")
        return cam_R, cam_T
    
    def _combine_batch_predictions(self, all_predictions):
        """合并多个批次的预测结果"""
        if len(all_predictions) == 1:
            return all_predictions[0]
        
        # 合并字典类型的预测结果
        if isinstance(all_predictions[0], dict):
            combined = {}
            for key in all_predictions[0].keys():
                # 收集所有批次中该key对应的tensor
                tensors = []
                for pred in all_predictions:
                    if key in pred and isinstance(pred[key], torch.Tensor):
                        tensor = pred[key]
                        print(f"  Batch tensor {key}: {tensor.shape}")
                        
                        # 🔧 修复：更智能的batch维度处理
                        # VGGT输出格式: [batch_size, num_frames, feature_dim]
                        # 我们需要合并所有批次的帧，所以要移除batch维度
                        
                        if tensor.dim() == 3:
                            # 3维tensor，检查第一维是否为batch维度
                            if tensor.shape[0] == 1:
                                # 如果batch_size=1，移除batch维度进行帧级别拼接
                                tensor = tensor.squeeze(0)
                                print(f"    Squeezed batch dim for {key}: {tensor.shape}")
                            else:
                                # 如果batch_size>1，说明这一批有多帧，保持原样
                                # 但这种情况在当前VGGT实现中不应该出现
                                print(f"    Warning: {key} has batch_size > 1: {tensor.shape}")
                        elif tensor.dim() == 4:
                            # 4维tensor，需要根据具体情况处理
                            if tensor.shape[0] == 1:
                                # 移除batch维度: [1, N, H, W] -> [N, H, W]
                                tensor = tensor.squeeze(0)
                                print(f"    Squeezed batch dim for {key}: {tensor.shape}")
                            else:
                                print(f"    Warning: {key} has batch_size > 1: {tensor.shape}")
                        # 对于其他维度的tensor，不做修改
                        
                        tensors.append(tensor)
                
                if tensors:
                    # 沿第0维（帧维度）拼接
                    try:
                        print(f"  Concatenating {key} tensors with shapes: {[t.shape for t in tensors]}")
                        combined[key] = torch.cat(tensors, dim=0)
                        print(f"  Result {key}: {combined[key].shape}")
                    except RuntimeError as e:
                        print(f"❌ Failed to concatenate {key} tensors: {e}")
                        print(f"  Tensor shapes: {[t.shape for t in tensors]}")
                        # 如果拼接失败，只保留第一个tensor
                        combined[key] = tensors[0]
                else:
                    # 非tensor类型，取第一个
                    combined[key] = all_predictions[0][key]
            return combined
        
        # 合并tensor类型的预测结果
        elif isinstance(all_predictions[0], torch.Tensor):
            # 处理tensor列表
            print(f"Concatenating direct tensors with shapes: {[t.shape for t in all_predictions]}")
            try:
                result = torch.cat(all_predictions, dim=0)
                print(f"Result shape: {result.shape}")
                return result
            except RuntimeError as e:
                print(f"❌ Failed to concatenate tensors: {e}")
                print(f"Tensor shapes: {[t.shape for t in all_predictions]}")
                # 返回第一个tensor
                return all_predictions[0]
        
        else:
            raise ValueError(f"Unsupported prediction type: {type(all_predictions[0])}")
    
# 全局VGGT实例
_vggt_camera_estimator = None

def get_vggt_camera_estimator(model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
    """获取全局VGGT相机估计实例"""
    global _vggt_camera_estimator
    if _vggt_camera_estimator is None:
        _vggt_camera_estimator = VGGTCameraEstimator(model_path)
    return _vggt_camera_estimator

def run_vggt_camera_estimation(img_folder, masks=None, calib=None, is_static=False, debug=False, debug_dir=None, batch_size=4):
    """
    VGGT版本的相机轨迹估计，替换传统SLAM方法
    
    Args:
        img_folder: 图像文件夹路径
        masks: 人体mask (VGGT不需要，忽略)
        calib: 相机内参 (VGGT可能不需要，忽略)
        is_static: 是否静态相机 (VGGT自己判断，忽略)
        debug: 调试模式
        debug_dir: 调试输出目录
        batch_size: 批处理大小，用于控制显存使用
    
    Returns:
        cam_R: torch.Tensor, shape (N, 3, 3) - 旋转矩阵
        cam_T: torch.Tensor, shape (N, 3) - 平移向量
        intrinsics: torch.Tensor, shape (N, 3, 3) - 每帧的内参矩阵
    """
    
    print("🎥 Running VGGT camera trajectory estimation...")
    
    # 获取图像文件列表
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No jpg images found in {img_folder}")
    
    print(f"Found {len(imgfiles)} images")
    
    # 使用VGGT预测相机poses和内参 (分批处理)
    vggt = get_vggt_camera_estimator()
    cam_R, cam_T, intrinsics = vggt.predict_poses_with_intrinsics(imgfiles, batch_size=batch_size)
    
    # 调试输出
    if debug and debug_dir is not None:
        _save_debug_info(cam_R, cam_T, imgfiles, debug_dir)
    
    return cam_R, cam_T, intrinsics

def _save_debug_info(cam_R, cam_T, imgfiles, debug_dir):
    """保存调试信息"""
    debug_path = Path(debug_dir)
    debug_path.mkdir(exist_ok=True, parents=True)
    
    print(f"💾 Saving VGGT debug info to {debug_path}")
    
    # 保存poses
    np.save(debug_path / 'vggt_cam_R.npy', cam_R.cpu().numpy())
    np.save(debug_path / 'vggt_cam_T.npy', cam_T.cpu().numpy())
    
    # 分析轨迹
    positions = cam_T.cpu().numpy()
    if len(positions) > 1:
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        debug_info = {
            'total_frames': len(imgfiles),
            'predicted_poses': len(cam_R),
            'mean_speed': float(np.mean(speeds)) if len(speeds) > 0 else 0.0,
            'std_speed': float(np.std(speeds)) if len(speeds) > 0 else 0.0,
            'max_speed': float(np.max(speeds)) if len(speeds) > 0 else 0.0,
            'trajectory_length': float(np.sum(speeds)) if len(speeds) > 0 else 0.0,
        }
        
        print(f"📊 VGGT Trajectory stats:")
        print(f"   Frames: {debug_info['predicted_poses']}/{debug_info['total_frames']}")
        print(f"   Mean speed: {debug_info['mean_speed']:.4f}")
        print(f"   Speed variation: {debug_info['std_speed']:.4f}")
        print(f"   Total length: {debug_info['trajectory_length']:.4f}")
        
        np.save(debug_path / 'vggt_analysis.npy', debug_info)

def run_vggt_depth_estimation(img_folder, batch_size=4, model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
    """
    VGGT深度图估计
    
    Args:
        img_folder: 图像文件夹路径
        batch_size: 批处理大小
        model_path: VGGT模型路径
        
    Returns:
        depth_maps: torch.Tensor, shape (N, H, W) - 深度图
        depth_conf: torch.Tensor, shape (N, H, W) - 深度置信度
    """
    
    print("🎯 Running VGGT depth estimation...")
    
    # 设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 加载模型
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}")
    
    print(f"Processing {len(imgfiles)} images for depth estimation...")
    
    all_depth_maps = []
    all_depth_conf = []
    
    # 分批处理
    for i in range(0, len(imgfiles), batch_size):
        batch_files = imgfiles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
        
        try:
            # 加载图像
            images = load_and_preprocess_images(batch_files).to(device)
            images = images[None]  # 添加batch维度
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 获取聚合特征
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
                    # 预测深度图
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                    
                    # 移除batch维度并转到CPU
                    if depth_map.shape[0] == 1:
                        depth_map = depth_map.squeeze(0)
                        depth_conf = depth_conf.squeeze(0)
                    
                    all_depth_maps.append(depth_map.cpu())
                    all_depth_conf.append(depth_conf.cpu())
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 合并所有批次结果
    if all_depth_maps:
        depth_maps = torch.cat(all_depth_maps, dim=0)
        depth_conf = torch.cat(all_depth_conf, dim=0)
        print(f"✅ Depth estimation completed: {depth_maps.shape}")
        return depth_maps, depth_conf
    else:
        raise RuntimeError("No depth maps were successfully generated")

def run_vggt_point_estimation(img_folder, batch_size=4, model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
    """
    VGGT点云估计
    
    Args:
        img_folder: 图像文件夹路径
        batch_size: 批处理大小
        model_path: VGGT模型路径
        
    Returns:
        point_maps: torch.Tensor, shape (N, H, W, 3) - 点云图
        point_conf: torch.Tensor, shape (N, H, W) - 点云置信度
    """
    
    print("📍 Running VGGT point cloud estimation...")
    
    # 设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 加载模型
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}")
    
    print(f"Processing {len(imgfiles)} images for point cloud estimation...")
    
    all_point_maps = []
    all_point_conf = []
    
    # 分批处理
    for i in range(0, len(imgfiles), batch_size):
        batch_files = imgfiles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
        
        try:
            # 加载图像
            images = load_and_preprocess_images(batch_files).to(device)
            images = images[None]  # 添加batch维度
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 获取聚合特征
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
                    # 预测点云图
                    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
                    
                    # 移除batch维度并转到CPU
                    if point_map.shape[0] == 1:
                        point_map = point_map.squeeze(0)
                        point_conf = point_conf.squeeze(0)
                    
                    all_point_maps.append(point_map.cpu())
                    all_point_conf.append(point_conf.cpu())
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 合并所有批次结果
    if all_point_maps:
        point_maps = torch.cat(all_point_maps, dim=0)
        point_conf = torch.cat(all_point_conf, dim=0)
        print(f"✅ Point cloud estimation completed: {point_maps.shape}")
        return point_maps, point_conf
    else:
        raise RuntimeError("No point maps were successfully generated")

def run_vggt_track_estimation(img_folder, query_points, batch_size=4, model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
    """
    VGGT轨迹跟踪
    
    Args:
        img_folder: 图像文件夹路径
        query_points: torch.Tensor, shape (N, 2) - 要跟踪的查询点
        batch_size: 批处理大小
        model_path: VGGT模型路径
        
    Returns:
        track_list: 轨迹列表
        vis_score: 可见性分数
        conf_score: 置信度分数
    """
    
    print("🎯 Running VGGT track estimation...")
    
    # 设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 加载模型
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}")
    
    print(f"Processing {len(imgfiles)} images for tracking {query_points.shape[0]} points...")
    
    # 确保query_points在正确设备上
    query_points = query_points.to(device)
    
    all_track_lists = []
    all_vis_scores = []
    all_conf_scores = []
    
    # 分批处理
    for i in range(0, len(imgfiles), batch_size):
        batch_files = imgfiles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
        
        try:
            # 加载图像
            images = load_and_preprocess_images(batch_files).to(device)
            images = images[None]  # 添加batch维度
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 获取聚合特征
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
                    # 预测轨迹
                    track_list, vis_score, conf_score = model.track_head(
                        aggregated_tokens_list, images, ps_idx, query_points=query_points[None]
                    )
                    
                    # 转到CPU并收集结果
                    all_track_lists.append(_move_tracks_to_cpu(track_list))
                    all_vis_scores.append(vis_score.cpu() if isinstance(vis_score, torch.Tensor) else vis_score)
                    all_conf_scores.append(conf_score.cpu() if isinstance(conf_score, torch.Tensor) else conf_score)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
        
        # 清理显存
        torch.cuda.empty_cache()
    
    if all_track_lists:
        print(f"✅ Track estimation completed for {len(all_track_lists)} batches")
        return all_track_lists, all_vis_scores, all_conf_scores
    else:
        raise RuntimeError("No tracks were successfully generated")

def run_vggt_depth_unprojection(img_folder, batch_size=4, model_path="/root/.cache/torch/hub/checkpoints/model.pt"):
    """
    VGGT深度图反投影到3D点云（使用深度+相机参数）
    这通常比直接的点云预测更准确
    
    Args:
        img_folder: 图像文件夹路径
        batch_size: 批处理大小
        model_path: VGGT模型路径
        
    Returns:
        point_maps_unprojected: torch.Tensor, shape (N, H, W, 3) - 反投影的3D点云
    """
    
    print("🔄 Running VGGT depth unprojection to 3D points...")
    
    # 设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 加载模型
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}")
    
    print(f"Processing {len(imgfiles)} images for depth unprojection...")
    
    all_unprojected_points = []
    
    # 分批处理
    for i in range(0, len(imgfiles), batch_size):
        batch_files = imgfiles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
        
        try:
            # 加载图像
            images = load_and_preprocess_images(batch_files).to(device)
            images = images[None]  # 添加batch维度
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 获取聚合特征
                    aggregated_tokens_list, ps_idx = model.aggregator(images)
                    
                    # 预测相机参数
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                    
                    # 预测深度图
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                    
                    # 从深度图反投影得到3D点云
                    if not HAS_VGGT_GEOMETRY:
                        raise RuntimeError("VGGT geometry utils not available. Please install proper VGGT version.")
                    
                    point_map_unprojected = unproject_depth_map_to_point_map(
                        depth_map.squeeze(0), 
                        extrinsic.squeeze(0), 
                        intrinsic.squeeze(0)
                    )
                    
                    all_unprojected_points.append(point_map_unprojected.cpu())
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
        
        # 清理显存
        torch.cuda.empty_cache()
    
    # 合并所有批次结果
    if all_unprojected_points:
        point_maps_unprojected = torch.cat(all_unprojected_points, dim=0)
        print(f"✅ Depth unprojection completed: {point_maps_unprojected.shape}")
        return point_maps_unprojected
    else:
        raise RuntimeError("No unprojected points were successfully generated")

def _move_tracks_to_cpu(tracks):
    """将tracks字典中的tensor移动到CPU"""
    if isinstance(tracks, dict):
        result = {}
        for key, value in tracks.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu()
            elif isinstance(value, list):
                result[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result
    elif isinstance(tracks, torch.Tensor):
        return tracks.cpu()
    else:
        return tracks

def run_vggt_unified_estimation(img_folder, 
                                batch_size=4, 
                                model_path="/root/.cache/torch/hub/checkpoints/model.pt",
                                outputs=['camera', 'depth', 'pointmap'],
                                query_points=None,
                                debug=False,
                                debug_dir=None):
    """
    VGGT统一估计：一次性获取所有需要的数据
    
    Args:
        img_folder: 图像文件夹路径
        batch_size: 批处理大小
        model_path: VGGT模型路径
        outputs: 需要的输出列表，可选: ['camera', 'depth', 'pointmap', 'unprojection', 'tracking']
        query_points: torch.Tensor, shape (N, 2) - 跟踪点（仅当'tracking'在outputs中时需要）
        debug: 是否输出调试信息
        debug_dir: 调试输出目录
        
    Returns:
        results: dict，包含请求的所有输出
            - 'camera': (cam_R, cam_T, intrinsics) - 相机姿态和内参
            - 'depth': (depth_maps, depth_conf) - 深度图和置信度
            - 'pointmap': (point_maps, point_conf) - 点云图和置信度  
            - 'unprojection': point_maps_unprojected - 深度反投影点云
            - 'tracking': (track_lists, vis_scores, conf_scores) - 轨迹跟踪结果
    """
    
    print(f"🚀 Running VGGT unified estimation for outputs: {outputs}")
    
    # 设备和数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    # 加载模型
    model = VGGT()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # 获取图像文件
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}")
    
    print(f"Processing {len(imgfiles)} images with unified estimation...")
    
    # 检查跟踪参数
    if 'tracking' in outputs:
        if query_points is None:
            raise ValueError("query_points must be provided when 'tracking' is in outputs")
        query_points = query_points.to(device)
        print(f"Tracking {query_points.shape[0]} query points")
    
    # 初始化结果收集器
    results = {}
    if 'camera' in outputs:
        all_pose_enc = []
        all_extrinsics = []
        all_intrinsics = []
    if 'depth' in outputs:
        all_depth_maps = []
        all_depth_conf = []
    if 'pointmap' in outputs:
        all_point_maps = []
        all_point_conf = []
    if 'unprojection' in outputs:
        all_unprojected_points = []
    if 'tracking' in outputs:
        all_track_lists = []
        all_vis_scores = []
        all_conf_scores = []
    
    # 分批处理
    for i in range(0, len(imgfiles), batch_size):
        batch_files = imgfiles[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(imgfiles) + batch_size - 1)//batch_size}: frames {i}-{i+len(batch_files)-1}")
        
        try:
            # 加载图像 - 只加载一次！
            images = load_and_preprocess_images(batch_files).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    # 添加batch维度
                    images_batch = images[None]
                    
                    # 共享的特征聚合 - 只计算一次！
                    aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                    
                    # 根据需要预测各种输出
                    if 'camera' in outputs:
                        # 预测相机参数
                        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                        # 解码外参和内参
                        H, W = images.shape[-2:]
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))
                        
                        all_pose_enc.append(pose_enc.cpu())
                        all_extrinsics.append(extrinsic.cpu())
                        all_intrinsics.append(intrinsic.cpu())
                    
                    if 'depth' in outputs:
                        # 预测深度图
                        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
                        
                        # 移除batch维度
                        if depth_map.shape[0] == 1:
                            depth_map = depth_map.squeeze(0)
                            depth_conf = depth_conf.squeeze(0)
                        
                        all_depth_maps.append(depth_map.cpu())
                        all_depth_conf.append(depth_conf.cpu())
                    
                    if 'pointmap' in outputs:
                        # 预测点云图
                        point_map, point_conf = model.point_head(aggregated_tokens_list, images_batch, ps_idx)
                        
                        # 移除batch维度
                        if point_map.shape[0] == 1:
                            point_map = point_map.squeeze(0)
                            point_conf = point_conf.squeeze(0)
                        
                        all_point_maps.append(point_map.cpu())
                        all_point_conf.append(point_conf.cpu())
                    
                    if 'unprojection' in outputs:
                        # 需要先获取相机参数和深度图
                        if 'camera' not in outputs or 'depth' not in outputs:
                            # 如果没有在outputs中，临时计算
                            if 'camera' not in outputs:
                                pose_enc_temp = model.camera_head(aggregated_tokens_list)[-1]
                                H, W = images.shape[-2:]
                                extrinsic_temp, intrinsic_temp = pose_encoding_to_extri_intri(pose_enc_temp, (H, W))
                            else:
                                extrinsic_temp, intrinsic_temp = extrinsic, intrinsic
                                
                            if 'depth' not in outputs:
                                depth_map_temp, _ = model.depth_head(aggregated_tokens_list, images_batch, ps_idx)
                            else:
                                depth_map_temp = depth_map
                        else:
                            extrinsic_temp, intrinsic_temp = extrinsic, intrinsic
                            depth_map_temp = depth_map
                        
                        # 深度图反投影到3D点云
                        if not HAS_VGGT_GEOMETRY:
                            print("Warning: VGGT geometry utils not available, skipping unprojection")
                        else:
                            point_map_unprojected = unproject_depth_map_to_point_map(
                                depth_map_temp.squeeze(0), 
                                extrinsic_temp.squeeze(0), 
                                intrinsic_temp.squeeze(0)
                            )
                            all_unprojected_points.append(point_map_unprojected.cpu())
                    
                    if 'tracking' in outputs:
                        # 预测轨迹跟踪
                        track_list, vis_score, conf_score = model.track_head(
                            aggregated_tokens_list, images_batch, ps_idx, query_points=query_points[None]
                        )
                        
                        # 转到CPU并收集结果
                        all_track_lists.append(_move_tracks_to_cpu(track_list))
                        all_vis_scores.append(vis_score.cpu() if isinstance(vis_score, torch.Tensor) else vis_score)
                        all_conf_scores.append(conf_score.cpu() if isinstance(conf_score, torch.Tensor) else conf_score)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
        
        # 清理显存
        del images
        torch.cuda.empty_cache()
    
    # 合并所有批次结果
    print("📦 Combining batch results...")
    
    if 'camera' in outputs and all_pose_enc:
        # 合并相机数据
        pose_enc_combined = torch.cat(all_pose_enc, dim=1) if len(all_pose_enc) > 1 else all_pose_enc[0]
        extrinsics_combined = torch.cat([e.squeeze(0) if e.shape[0] == 1 else e for e in all_extrinsics], dim=0)
        intrinsics_combined = torch.cat([i.squeeze(0) if i.shape[0] == 1 else i for i in all_intrinsics], dim=0)
        
        # 提取相机poses
        cam_R = extrinsics_combined[:, :3, :3].float()
        cam_T = extrinsics_combined[:, :3, 3].float()
        
        results['camera'] = (cam_R, cam_T, intrinsics_combined)
        print(f"✅ Camera estimation completed: {cam_R.shape[0]} poses")
        
        # 调试输出
        if debug and debug_dir is not None:
            _save_debug_info(cam_R, cam_T, imgfiles, debug_dir)
    
    if 'depth' in outputs and all_depth_maps:
        depth_maps = torch.cat(all_depth_maps, dim=0)
        depth_conf = torch.cat(all_depth_conf, dim=0)
        results['depth'] = (depth_maps, depth_conf)
        print(f"✅ Depth estimation completed: {depth_maps.shape}")
    
    if 'pointmap' in outputs and all_point_maps:
        point_maps = torch.cat(all_point_maps, dim=0)
        point_conf = torch.cat(all_point_conf, dim=0)
        results['pointmap'] = (point_maps, point_conf)
        print(f"✅ Point cloud estimation completed: {point_maps.shape}")
    
    if 'unprojection' in outputs and all_unprojected_points:
        point_maps_unprojected = torch.cat(all_unprojected_points, dim=0)
        results['unprojection'] = point_maps_unprojected
        print(f"✅ Depth unprojection completed: {point_maps_unprojected.shape}")
    
    if 'tracking' in outputs and all_track_lists:
        results['tracking'] = (all_track_lists, all_vis_scores, all_conf_scores)
        print(f"✅ Track estimation completed for {len(all_track_lists)} batches")
    
    print(f"🎉 Unified VGGT estimation completed with outputs: {list(results.keys())}")
    return results
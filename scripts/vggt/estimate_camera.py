import sys
import os
import torch
import argparse
import numpy as np
import cv2
from glob import glob

# 使用path_utils确保项目路径正确
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from lib.utils.path_utils import find_tram_root
from lib.pipeline import video2frames, detect_segment_track
from lib.camera.vggt_camera import run_vggt_camera_estimation, run_vggt_depth_estimation, run_vggt_track_estimation, run_vggt_unified_estimation
from lib.camera import align_cam_to_world
from pycocotools import mask as masktool

# 添加VGGT路径
from lib.utils.path_utils import ensure_vggt_in_path
ensure_vggt_in_path()

# VGGT相关导入
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

parser = argparse.ArgumentParser()
parser.add_argument("--video", type=str, default='example_video.mov', help='input video')
parser.add_argument("--static_camera", action='store_true', help='whether the camera is static')
parser.add_argument("--visualize_mask", action='store_true', help='save deva vos for visualization')
parser.add_argument('--output_dir', type=str, default=None, help='output directory')
parser.add_argument('--debug', action='store_true', help='enable debug mode for camera analysis')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for VGGT processing (reduce if out of memory, max ~50 frames needs 11GB)')
parser.add_argument('--vggt_model_path', type=str, default="/root/.cache/torch/hub/checkpoints/model.pt", 
                    help='path to VGGT model checkpoint')
parser.add_argument('--use_vggt_tracking', action='store_true', 
                    help='Use VGGT tracking for Bundle Adjustment (better quality but slower)')
parser.add_argument('--num_query_points', type=int, default=500,
                    help='Number of query points for VGGT tracking')
args = parser.parse_args()

# Bundle Adjustment 默认参数（暂时关闭）
BA_STEPS = 0  # 关闭BA优化
BA_ROBUST_THRESHOLD = 1.0  # 降低鲁棒阈值，更严格
USE_HUMAN_MASK = False  # 暂时不使用人体遮罩约束 (设为True可启用)

# File and folders - 处理视频文件路径
if os.path.isabs(args.video):
    # 如果是绝对路径，直接使用
    file = args.video
else:
    # 如果是相对路径，基于TRAM项目根目录
    tram_root = find_tram_root()
    file = os.path.join(tram_root, args.video)

root = os.path.dirname(file)
seq = os.path.basename(file).split('.')[0]

# 确保使用项目根目录下的结果路径
tram_root = find_tram_root()

if args.output_dir is not None:
    if os.path.isabs(args.output_dir):
        seq_folder = args.output_dir
    else:
        # 相对路径基于项目根目录
        seq_folder = os.path.join(tram_root, args.output_dir)
else:
    seq_folder = os.path.join(tram_root, 'results', seq)

img_folder = os.path.join(seq_folder, 'images')
print(f"Using output directory: {seq_folder}")
os.makedirs(seq_folder, exist_ok=True)
os.makedirs(img_folder, exist_ok=True)

##### Extract Frames #####
print('Extracting frames ...')
print(f'Input video: {file}')
print(f'Output image folder: {img_folder}')

# 检查视频文件是否存在
if not os.path.exists(file):
    raise FileNotFoundError(f"Video file not found: {file}")

nframes = video2frames(file, img_folder)
print(f'Extracted {nframes} frames')

# 检查是否成功提取了图像
extracted_images = glob(f'{img_folder}/*.jpg')
if len(extracted_images) == 0:
    print(f"Warning: No .jpg images found, checking for other formats...")
    all_images = glob(f'{img_folder}/*')
    print(f"Files in {img_folder}: {all_images}")
    if len(all_images) == 0:
        raise ValueError(f"No frames were extracted to {img_folder}")
else:
    print(f'Found {len(extracted_images)} extracted images')

##### Human Detection, Segmentation, and Tracking #####
print('Detect, Segment, and Track ...')
imgfiles = sorted(glob(f'{img_folder}/*.jpg'))

# 直接调用TRAM的检测/分割/跟踪管道
boxes_, masks_, tracks_ = detect_segment_track(
    imgfiles, 
    seq_folder, 
    thresh=0.25,
    min_size=100, 
    save_vos=args.visualize_mask
)

print(f'✅ Tracking completed:')
print(f'   Total tracks: {len(tracks_.item()) if hasattr(tracks_, "item") else len(tracks_)}')
if hasattr(tracks_, 'item'):
    tracks_dict = tracks_.item()
    total_detections = sum(len(track_data) for track_data in tracks_dict.values())
    print(f'   Total detections: {total_detections}')
elif hasattr(tracks_, 'values'):
    total_detections = sum(len(track_data) for track_data in tracks_.values())
    print(f'   Total detections: {total_detections}')

# 将masks转换为torch tensor用于相机估计（可选使用）
if len(masks_) > 0:
    decoded_masks = np.array([masktool.decode(m) for m in masks_])
    human_masks_for_camera = torch.from_numpy(decoded_masks)
    print(f'   Human masks available for camera estimation: {human_masks_for_camera.shape}')
else:
    human_masks_for_camera = None
    print('   No human masks generated')

##### VGGT Unified Estimation #####
print('Running VGGT unified estimation...')

# 确定需要的输出类型
vggt_outputs = ['camera', 'depth']
query_points = None

# 如果需要VGGT跟踪，准备查询点
if args.use_vggt_tracking:
    print('Preparing VGGT tracking for Bundle Adjustment...')
    from lib.camera.vggt_ba import generate_query_points_for_ba
    
    # 生成适合BA的查询点
    query_points = generate_query_points_for_ba(imgfiles, args.num_query_points, avoid_center=True)
    vggt_outputs.append('tracking')
    print(f'Added tracking with {query_points.shape[0]} query points')

# 一次性运行所有VGGT估计
try:
    vggt_results = run_vggt_unified_estimation(
        img_folder,
        batch_size=args.batch_size,
        model_path=args.vggt_model_path,
        outputs=vggt_outputs,
        query_points=query_points,
        debug=args.debug,
        debug_dir=f'{seq_folder}/vggt_debug' if args.debug else None
    )
    
    # 提取相机数据
    cam_R, cam_T, vggt_intrinsics = vggt_results['camera']
    print(f'✅ Camera estimation completed: {cam_R.shape[0]} poses')
    
    # 提取深度数据
    depth_maps, depth_conf = vggt_results['depth']
    print(f'✅ Depth estimation completed: {depth_maps.shape}')
    
    # 保存深度数据
    depth_dir = os.path.join(seq_folder, 'depths')
    os.makedirs(depth_dir, exist_ok=True)
    
    np.save(f'{depth_dir}/depth_maps.npy', depth_maps.numpy())
    np.save(f'{depth_dir}/depth_conf.npy', depth_conf.numpy())
    print(f'💾 Depth data saved to: {depth_dir}/')
    
    # 处理跟踪数据（如果有）
    vggt_tracks = None
    if 'tracking' in vggt_results:
        track_lists, vis_scores, conf_scores = vggt_results['tracking']
        
        # 合并跟踪结果为单个字典
        vggt_tracks = {
            'track': torch.cat(track_lists, dim=0) if track_lists else None,
            'vis': torch.cat(vis_scores, dim=0) if vis_scores else None,
            'conf': torch.cat(conf_scores, dim=0) if conf_scores else None
        }
        print(f"✅ VGGT tracking completed: {vggt_tracks['track'].shape if vggt_tracks['track'] is not None else 'None'}")
    
except Exception as e:
    print(f'❌ VGGT unified estimation failed: {e}')
    print('Falling back to separate function calls...')
    
    # 回退到原来的分开调用方式
    print('Running VGGT camera pose estimation...')
    cam_R, cam_T, vggt_intrinsics = run_vggt_camera_estimation(
        img_folder,
        debug=args.debug,
        debug_dir=f'{seq_folder}/vggt_debug' if args.debug else None,
        batch_size=args.batch_size
    )
    
    print('Running VGGT depth estimation...')
    try:
        depth_maps, depth_conf = run_vggt_depth_estimation(
            img_folder,
            batch_size=args.batch_size,
            model_path=args.vggt_model_path
        )
        print(f'✅ Depth estimation completed: {depth_maps.shape}')
        
        # 保存深度数据
        depth_dir = os.path.join(seq_folder, 'depths')
        os.makedirs(depth_dir, exist_ok=True)
        
        np.save(f'{depth_dir}/depth_maps.npy', depth_maps.numpy())
        np.save(f'{depth_dir}/depth_conf.npy', depth_conf.numpy())
        print(f'💾 Depth data saved to: {depth_dir}/')
        
    except Exception as e:
        print(f'❌ Depth estimation failed: {e}')
        depth_maps, depth_conf = None, None
    
    # 处理跟踪（如果需要）
    vggt_tracks = None
    if args.use_vggt_tracking and query_points is not None:
        try:
            track_results = run_vggt_track_estimation(
                img_folder, query_points, args.batch_size, args.vggt_model_path
            )
            track_lists, vis_scores, conf_scores = track_results
            
            # 合并跟踪结果为单个字典
            vggt_tracks = {
                'track': torch.cat(track_lists, dim=0) if track_lists else None,
                'vis': torch.cat(vis_scores, dim=0) if vis_scores else None,
                'conf': torch.cat(conf_scores, dim=0) if conf_scores else None
            }
            print(f"✅ VGGT tracking completed: {vggt_tracks['track'].shape if vggt_tracks['track'] is not None else 'None'}")
        except Exception as e:
            print(f"   Warning: VGGT tracking failed: {e}")
            vggt_tracks = None

# 世界坐标对齐
print('Aligning camera poses to world coordinates...')
# 确保tensor在CPU上并转换为float32进行世界坐标对齐
cam_R_cpu = cam_R.cpu().float()
cam_T_cpu = cam_T.cpu().float()
wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R_cpu, cam_T_cpu)

##### Bundle Adjustment Refinement (可选) #####
if BA_STEPS > 0:
    print(f'Applying Bundle Adjustment refinement with {BA_STEPS} steps...')
    
    # 导入BA模块
    from lib.camera.vggt_ba import (
        extract_vggt_feature_matches, 
        apply_lightweight_ba
    )
    
    # 保存原始结果用于对比
    cam_R_original = cam_R_cpu.clone()
    cam_T_original = cam_T_cpu.clone()
    
    try:
        # 提取特征匹配 - 优先使用VGGT跟踪，然后是光流
        feature_matches = extract_vggt_feature_matches(vggt_tracks, imgfiles)
        
        # 使用VGGT的per-frame内参进行Bundle Adjustment
        if vggt_intrinsics is not None:
            print('   Using VGGT per-frame intrinsics for Bundle Adjustment')
            ba_intrinsics = vggt_intrinsics
            # 从第一帧内参提取focal和center用于兼容性
            first_intrinsic = vggt_intrinsics[0]  # [3, 3]
            focal = first_intrinsic[0, 0].item()  # fx
            center = [first_intrinsic[0, 2].item(), first_intrinsic[1, 2].item()]  # cx, cy
        else:
            print('   Warning: No VGGT intrinsics available, using default intrinsics')
            # 默认内参估计
            imgfiles_for_calib = sorted(glob(f'{img_folder}/*.jpg'))
            first_img = cv2.imread(imgfiles_for_calib[0])
            H, W = first_img.shape[:2]
            focal = max(H, W) * 1.2
            center = [W / 2, H / 2]
            ba_intrinsics = None
        
        # 生成人体遮罩（如果需要）
        human_masks = None
        if USE_HUMAN_MASK:
            print('   Using existing human masks for BA constraint...')
            # 使用已生成的人体遮罩
            if human_masks_for_camera is not None:
                human_masks = human_masks_for_camera
                print(f'     Using {human_masks.shape[0]} human masks for BA constraint')
            else:
                print('     Warning: No human masks available, skipping mask constraint')
        else:
            print('   Human mask constraint disabled (USE_HUMAN_MASK=False)')
        
        # 应用轻量级BA
        cam_R_refined, cam_T_refined = apply_lightweight_ba(
            cam_R_cpu, cam_T_cpu, feature_matches,
            intrinsics=ba_intrinsics,
            masks=human_masks,
            steps=BA_STEPS,
            robust_threshold=BA_ROBUST_THRESHOLD
        )
        
        # 计算改进度量
        rot_diff = torch.norm(cam_R_refined - cam_R_original, dim=(1,2)).mean()
        trans_diff = torch.norm(cam_T_refined - cam_T_original, dim=1).mean()
        
        print(f'   BA completed:')
        print(f'     Average rotation change: {rot_diff:.6f}')
        print(f'     Average translation change: {trans_diff:.6f}')
        
        # 更新相机参数
        cam_R_cpu = cam_R_refined
        cam_T_cpu = cam_T_refined
        
        # 重新对齐世界坐标
        wd_cam_R, wd_cam_T, spec_f = align_cam_to_world(imgfiles[0], cam_R_cpu, cam_T_cpu)
        
        ba_success = True
        
    except Exception as e:
        print(f'   BA refinement failed: {e}')
        print('   Using original VGGT results')
        cam_R_cpu = cam_R_original
        cam_T_cpu = cam_T_original
        ba_success = False

else:
    print('Bundle Adjustment disabled (BA_STEPS=0)')
    ba_success = False
    
    # 提取focal和center用于保存
    if vggt_intrinsics is not None:
        first_intrinsic = vggt_intrinsics[0]  # [3, 3]
        focal = first_intrinsic[0, 0].item()  # fx
        center = [first_intrinsic[0, 2].item(), first_intrinsic[1, 2].item()]  # cx, cy
    else:
        # 默认内参估计
        imgfiles_for_calib = sorted(glob(f'{img_folder}/*.jpg'))
        first_img = cv2.imread(imgfiles_for_calib[0])
        H, W = first_img.shape[:2]
        focal = max(H, W) * 1.2
        center = [W / 2, H / 2]

##### Process VGGT Intrinsics Scaling #####
# Scale VGGT intrinsics from 518x518 to original image resolution (if needed)
scaled_vggt_intrinsics = None
if vggt_intrinsics is not None:
    print(f'Processing VGGT intrinsics scaling...')
    
    # Get original image dimensions
    first_img = cv2.imread(imgfiles[0])
    original_h, original_w = first_img.shape[:2]
    vggt_resolution = 518
    
    scale_x = original_w / vggt_resolution
    scale_y = original_h / vggt_resolution
    
    # Scale the intrinsics to match original resolution
    scaled_vggt_intrinsics = vggt_intrinsics.clone()
    scaled_vggt_intrinsics[:, 0, 0] *= scale_x  # fx
    scaled_vggt_intrinsics[:, 1, 1] *= scale_y  # fy
    scaled_vggt_intrinsics[:, 0, 2] *= scale_x  # cx
    scaled_vggt_intrinsics[:, 1, 2] *= scale_y  # cy
    
    # Update focal and center with scaled values
    first_intrinsic_scaled = scaled_vggt_intrinsics[0]  # [3, 3]
    focal = first_intrinsic_scaled[0, 0].item()  # fx (scaled)
    center = [first_intrinsic_scaled[0, 2].item(), first_intrinsic_scaled[1, 2].item()]  # cx, cy (scaled)
    
    print(f'   📐 Scaled VGGT intrinsics from {vggt_resolution}x{vggt_resolution} to {original_w}x{original_h}')
    print(f'   📐 Original focal: {vggt_intrinsics[0, 0, 0].item():.1f} -> Scaled focal: {focal:.1f}')
    print(f'   📐 Scaled center: ({center[0]:.1f}, {center[1]:.1f})')

##### Save Camera Results #####
print('Saving VGGT camera estimation results ...')

# 确保所有tensor都在CPU上再转换为numpy
camera = {
    'pred_cam_R': cam_R_cpu.numpy(), 
    'pred_cam_T': cam_T_cpu.numpy(),
    'world_cam_R': wd_cam_R.cpu().numpy() if hasattr(wd_cam_R, 'cpu') else wd_cam_R, 
    'world_cam_T': wd_cam_T.cpu().numpy() if hasattr(wd_cam_T, 'cpu') else wd_cam_T,
    # 使用缩放后的内参值（如果有VGGT内参的话）
    'img_focal': np.array(focal.item() if hasattr(focal, 'item') else focal), 
    'img_center': [float(center[0].item() if hasattr(center[0], 'item') else center[0]), 
                   float(center[1].item() if hasattr(center[1], 'item') else center[1])], 
    'spec_focal': np.array(spec_f.item() if hasattr(spec_f, 'item') else spec_f),
    'method': 'VGGT' if BA_STEPS == 0 else 'VGGT+BA',  # 根据BA状态标记方法
    'ba_refined': ba_success,  # BA是否成功执行
    'ba_steps': BA_STEPS,
    'ba_success': ba_success,
    'total_frames': len(imgfiles),
    'estimated_poses': len(cam_R_cpu),
    # 保存内参信息
    'has_per_frame_intrinsics': vggt_intrinsics is not None,
    'intrinsics_type': 'vggt_per_frame' if vggt_intrinsics is not None else 'fixed_calibrated'
}

# 保存VGGT内参矩阵（原始和缩放后的版本）
if vggt_intrinsics is not None:
    camera['vggt_intrinsics'] = vggt_intrinsics.cpu().numpy()  # [N, 3, 3] 原始518x518内参矩阵
    camera['vggt_intrinsics_scaled'] = scaled_vggt_intrinsics.cpu().numpy()  # [N, 3, 3] 缩放后内参矩阵
    print(f'   � Saved VGGT per-frame intrinsics: {vggt_intrinsics.shape}')
    print(f'   � Saved scaled intrinsics for original resolution')

# 保存对比信息（如果BA成功执行）
if ba_success and 'cam_R_original' in locals():
    camera['pred_cam_R_original'] = cam_R_original.numpy()
    camera['pred_cam_T_original'] = cam_T_original.numpy()
    camera['ba_rot_improvement'] = np.array(rot_diff.item() if hasattr(rot_diff, 'item') else rot_diff) if 'rot_diff' in locals() else 0.0
    camera['ba_trans_improvement'] = np.array(trans_diff.item() if hasattr(trans_diff, 'item') else trans_diff) if 'trans_diff' in locals() else 0.0

np.save(f'{seq_folder}/camera.npy', camera)

# 单独保存VGGT内参矩阵（如果存在）
if vggt_intrinsics is not None:
    np.save(f'{seq_folder}/vggt_intrinsics.npy', vggt_intrinsics.cpu().numpy())
    print(f'   💾 VGGT intrinsics also saved separately to: vggt_intrinsics.npy')

print(f'✅ VGGT camera estimation completed!')
print(f'   Method: {"VGGT only" if BA_STEPS == 0 else "VGGT + BA"}')
print(f'   Frames: {len(imgfiles)}')
print(f'   Estimated poses: {len(cam_R_cpu)}')
if BA_STEPS > 0:
    print(f'   BA refinement: {BA_STEPS} steps {"(SUCCESS)" if ba_success else "(FAILED)"}')
    if ba_success and 'rot_diff' in locals() and 'trans_diff' in locals():
        print(f'   Improvement: rot={rot_diff:.6f}, trans={trans_diff:.6f}')
else:
    print(f'   BA refinement: DISABLED')
print(f'   Intrinsics: {"Per-frame (VGGT)" if vggt_intrinsics is not None else "Fixed (calibrated)"}')
if vggt_intrinsics is not None:
    print(f'   Intrinsics shape: {vggt_intrinsics.shape}')
print(f'   Tracking: Real (ViTDet+SAM+DEVA)')
print(f'   Results saved to: {seq_folder}/')
print(f'     - camera.npy (camera parameters)')
if vggt_intrinsics is not None:
    print(f'     - camera.npy[vggt_intrinsics] (per-frame intrinsics)')
    print(f'     - vggt_intrinsics.npy (per-frame intrinsics separate file)')
print(f'     - tracks.npy (tracking data)')
print(f'     - boxes.npy (detection boxes)')
print(f'     - masks.npy (segmentation masks)')

if args.debug:
    print(f'   Debug info saved to: {seq_folder}/vggt_debug')

##### Save Tracking Results #####
print('Saving tracking results...')

# 直接保存真实的tracking结果
np.save(f'{seq_folder}/tracks.npy', tracks_)
np.save(f'{seq_folder}/boxes.npy', boxes_)
np.save(f'{seq_folder}/masks.npy', masks_)

print(f'✅ Saved tracking data:')
print(f'   tracks.npy: {len(tracks_.item()) if hasattr(tracks_, "item") else len(tracks_)} tracks')
print(f'   boxes.npy: {len(boxes_)} frames')
print(f'   masks.npy: {len(masks_)} frames')

# 统计信息
if hasattr(tracks_, 'item'):
    tracks_dict = tracks_.item()
    total_detections = sum(len(track_data) for track_data in tracks_dict.values())
    print(f'   Total detections: {total_detections}')
    print(f'   Track IDs: {list(tracks_dict.keys())}')
elif hasattr(tracks_, 'values'):
    total_detections = sum(len(track_data) for track_data in tracks_.values())
    print(f'   Total detections: {total_detections}')
    print(f'   Track IDs: {list(tracks_.keys())}')
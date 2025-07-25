import numpy as np
import cv2
import math
import torch
import os
from thirdparty.camcalib.model import CameraRegressorNetwork


def run_spec(img):
    # Predict gravity direction + fov using SPEC
    
    # 使用相对导入path_utils
    from ..utils.path_utils import get_pretrain_path
    
    spec_checkpoint = get_pretrain_path('camcalib_sa_biased_l2.ckpt')
    
    spec = CameraRegressorNetwork()
    spec = spec.load_ckpt(spec_checkpoint).to('cuda')

    with torch.no_grad():
        if isinstance(img, str):
            img = cv2.imread(img)[:,:,::-1]

        preds = spec(img, transform_data=True)
        vfov, pitch, roll = preds
        f_pix = img.shape[0] / (2 * np.tan(vfov / 2.))
        
    return [f_pix, pitch, roll]


def cam_wrt_gravity(pitch, roll):
    # Convert pitch-roll from SPEC to cam pose wrt to gravity direction
    Rpitch = rotation_about_x(-pitch)[:3, :3]
    Rroll = rotation_about_y(roll)[:3, :3]
    R_gc = Rpitch @ Rroll
    return R_gc


def cam_wrt_world(pitch, roll):
    # Cam from gravity frame to world frame
    R_gc = cam_wrt_gravity(pitch, roll)
    R_wg = torch.Tensor([[1,0,0],
                         [0,-1,0],
                         [0,0,-1]])
    # 确保R_wg与R_gc在同一设备上
    if hasattr(R_gc, 'device'):
        R_wg = R_wg.to(R_gc.device)
    R_wc = R_wg @ R_gc
    return R_wc


def align_cam_to_world(img, cam_R, cam_T):
    f_pix, pitch, roll = run_spec(img)
    R_wc = cam_wrt_world(pitch, roll)
    
    # 确保R_wc与输入张量在同一设备上
    if hasattr(cam_R, 'device'):
        R_wc = R_wc.to(cam_R.device)

    world_cam_R = torch.einsum('ij,bjk->bik', R_wc, cam_R)
    world_cam_T = torch.einsum('ij,bj->bi', R_wc, cam_T)

    return world_cam_R, world_cam_T, f_pix


def rotation_about_x(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def rotation_about_y(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]])


def rotation_about_z(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

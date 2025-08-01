import torch
import torch.nn.functional as F
# from lietorch import SO3

from lib.utils.geometry import batch_rodrigues
from lib.utils import rotation_conversions as geo

def compute_l2_loss(batch):
    x2 = batch["x2"]
    output = batch["output"]
    
    loss = F.mse_loss(x2, output, reduction='mean')
    return loss


def keypoint_loss(batch, openpose_weight=0., gt_weight=1.):
    """ Compute 2D reprojection loss on the keypoints.
    The loss is weighted by the confidence.
    The available keypoints are different for each dataset.
    """
    pred_keypoints_2d = batch['pred_keypoints_2d']
    gt_keypoints_2d = batch['keypoints']

    conf = gt_keypoints_2d[:, :, [-1]].clone()
    conf[:, :25] *= openpose_weight
    conf[:, 25:] *= gt_weight

    mse  = F.mse_loss(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1], reduction='none')
    loss = (conf * mse).mean()
    return loss


def keypoint_3d_loss(batch):
    """Compute 3D keypoint loss for the examples that 3D keypoint annotations are available.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = batch['pred_keypoints_3d']
    gt_keypoints_3d = batch['pose_3d']
    has_pose_3d = batch['has_pose_3d']
    device = pred_keypoints_3d.device

    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
    conf = conf[has_pose_3d == 1]
    pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]

    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
        pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

        mse  = F.mse_loss(pred_keypoints_3d, gt_keypoints_3d, reduction='none')
        loss = (conf * mse).mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)

    return loss

def acceleration_loss(batch):
    """Compute 3D keypoint acceleration loss.
    The loss is weighted by the confidence.
    """
    pred_keypoints_3d = batch['pred_keypoints_3d']
    pred_keypoints_3d = pred_keypoints_3d[:, 25:, :]

    conf = batch['pose_3d'][:, :, [-1]].clone()
    gt_keypoints_3d = batch['pose_3d'][:, :, :-1].clone()

    # center at root
    gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
    pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]

    # compute accel
    num_j = 24
    joints_gt = gt_keypoints_3d.reshape(-1, 16, num_j, 3)
    joints_pred = pred_keypoints_3d.reshape(-1, 16, num_j, 3)
    conf = conf.reshape(-1, 16, 24, 1)

    accel_gt = joints_gt[:, :-2] - 2 * joints_gt[:, 1:-1] + joints_gt[:, 2:]
    accel_pred = joints_pred[:, :-2] - 2 * joints_pred[:, 1:-1] + joints_pred[:, 2:]

    mse  = F.mse_loss(accel_gt, accel_pred, reduction='none')
    loss = (conf[:,1:-1] * mse).mean()

    return loss


def smpl_losses(batch, pose_weight=1., beta_weight=0.001):
    pred_rotmat = batch['pred_rotmat']
    pred_betas  = batch['pred_betas']
    gt_pose  = batch['pose']
    gt_betas = batch['betas']
    has_smpl = batch['has_smpl']
    beta_weight = batch['beta_weight']
    device = pred_rotmat.device

    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]
    if len(pred_rotmat_valid) > 0:
        loss_regr_pose  = F.mse_loss(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_betas = F.mse_loss(pred_betas_valid, gt_betas_valid)
    else:
        loss_regr_pose  = torch.FloatTensor(1).fill_(0.).mean().to(device)
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).mean().to(device)

    loss = pose_weight*loss_regr_pose + beta_weight*loss_regr_betas
    return loss

def smpl_losses_plus(batch, pose_weight=1., beta_weight=0.001, init_w=1.0):
    pred_rotmat_0 = batch['pred_rotmat_0']
    pred_rotmat = batch['pred_rotmat']
    pred_betas  = batch['pred_betas']
    gt_pose  = batch['pose']
    gt_betas = batch['betas']
    has_smpl = batch['has_smpl']
    beta_weight = batch['beta_weight']
    device = pred_rotmat.device

    pred_rotmat_0_valid = pred_rotmat_0[has_smpl == 1]
    pred_rotmat_valid = pred_rotmat[has_smpl == 1]
    gt_rotmat_valid = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)[has_smpl == 1]
    pred_betas_valid = pred_betas[has_smpl == 1]
    gt_betas_valid = gt_betas[has_smpl == 1]

    if len(pred_rotmat_valid) > 0:
        loss_regr_betas = F.mse_loss(pred_betas_valid, gt_betas_valid)
        loss_regr_pose  = F.mse_loss(pred_rotmat_valid, gt_rotmat_valid)
        loss_regr_pose += F.mse_loss(pred_rotmat_0_valid, gt_rotmat_valid) * init_w
        
    else:
        loss_regr_betas = torch.FloatTensor(1).fill_(0.).mean().to(device)
        loss_regr_pose  = torch.FloatTensor(1).fill_(0.).mean().to(device)  

    loss = pose_weight*loss_regr_pose + beta_weight*loss_regr_betas
    return loss

def vertice_loss(batch):
    pred_rotmat = batch['pred_rotmat']
    pred_betas  = batch['pred_betas']
    has_smpl = batch['has_smpl']
    smpl = batch['smpl']
    device = pred_rotmat.device

    # pred vertices
    pred_out = smpl(global_orient=pred_rotmat[:,[0]],
                    body_pose=pred_rotmat[:,1:],
                    betas=pred_betas, 
                    pose2rot=False)
    pred_vert = pred_out.vertices

    # gt vertices
    if 'gt_vert' not in batch:
        gt_pose  = batch['pose']
        gt_betas = batch['betas']
        gt_rotmat = batch_rodrigues(gt_pose.view(-1,3)).view(-1, 24, 3, 3)

        gt_out = smpl(global_orient=gt_rotmat[:,[0]],
                      body_pose=gt_rotmat[:,1:],
                      betas=gt_betas,
                      pose2rot=False)
        gt_vert = gt_out.vertices
        batch['gt_vert'] = gt_vert
    else:
        gt_vert = batch['gt_vert']

    gt_vert = gt_vert[has_smpl == 1]
    pred_vert = pred_vert[has_smpl == 1]

    if len(gt_vert) > 0:
        loss  = F.l1_loss(pred_vert, gt_vert)
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)

    return loss


def cam_depth_loss(batch):
    # The last component is a loss that forces the network to predict positive depth values
    pred_camera = batch['pred_cam']
    loss = ((torch.exp(-pred_camera[:,0]*10)) ** 2 ).mean()

    return loss.clamp(min=None, max=10.0)

def camera_loss(batch):
    """
    VGGT相机参数损失函数
    相机参数格式: [translation(3) + quaternion_rotation(4) + field_of_view(2)] = 9维
    """
    if 'pred_camera_params' not in batch or 'camera' not in batch:
        device = next(iter(batch.values())).device
        return torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    pred_camera = batch['pred_camera_params']
    gt_camera = batch['camera']
    has_camera = batch.get('has_camera', torch.ones(gt_camera.shape[0]))
    device = pred_camera.device
    
    # 处理不同维度的数据
    if gt_camera.dim() == 3:  # [B, T, 9]
        if has_camera.dim() == 1:
            has_camera = has_camera.unsqueeze(1).expand(-1, gt_camera.shape[1])
        valid_mask = has_camera == 1
        pred_camera_valid = pred_camera[valid_mask]
        gt_camera_valid = gt_camera[valid_mask]
    else:  # [B, 9]
        valid_mask = has_camera == 1
        pred_camera_valid = pred_camera[valid_mask]
        gt_camera_valid = gt_camera[valid_mask]
    
    if len(pred_camera_valid) > 0:
        loss = F.mse_loss(pred_camera_valid, gt_camera_valid)
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    return loss

def camera_translation_loss(batch):
    """相机平移参数损失 (前3个参数)"""
    if 'pred_camera_params' not in batch or 'camera' not in batch:
        device = next(iter(batch.values())).device
        return torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    pred_camera = batch['pred_camera_params']
    gt_camera = batch['camera']
    has_camera = batch.get('has_camera', torch.ones(gt_camera.shape[0]))
    device = pred_camera.device
    
    # 只使用平移部分
    pred_translation = pred_camera[..., :3]
    gt_translation = gt_camera[..., :3]
    
    # 处理有效数据
    if gt_translation.dim() == 3:
        if has_camera.dim() == 1:
            has_camera = has_camera.unsqueeze(1).expand(-1, gt_translation.shape[1])
        valid_mask = has_camera == 1
        pred_translation_valid = pred_translation[valid_mask]
        gt_translation_valid = gt_translation[valid_mask]
    else:
        valid_mask = has_camera == 1
        pred_translation_valid = pred_translation[valid_mask]
        gt_translation_valid = gt_translation[valid_mask]
    
    if len(pred_translation_valid) > 0:
        loss = F.mse_loss(pred_translation_valid, gt_translation_valid)
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    return loss

def camera_rotation_loss(batch):
    """相机旋转参数损失 (3-7参数: 四元数)"""
    if 'pred_camera_params' not in batch or 'camera' not in batch:
        device = next(iter(batch.values())).device
        return torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    pred_camera = batch['pred_camera_params']
    gt_camera = batch['camera']
    has_camera = batch.get('has_camera', torch.ones(gt_camera.shape[0]))
    device = pred_camera.device
    
    # 只使用旋转部分（四元数）
    pred_quat = pred_camera[..., 3:7]
    gt_quat = gt_camera[..., 3:7]
    
    # 处理有效数据
    if gt_quat.dim() == 3:
        if has_camera.dim() == 1:
            has_camera = has_camera.unsqueeze(1).expand(-1, gt_quat.shape[1])
        valid_mask = has_camera == 1
        pred_quat_valid = pred_quat[valid_mask]
        gt_quat_valid = gt_quat[valid_mask]
    else:
        valid_mask = has_camera == 1
        pred_quat_valid = pred_quat[valid_mask]
        gt_quat_valid = gt_quat[valid_mask]
    
    if len(pred_quat_valid) > 0:
        # 四元数损失：考虑q和-q的等价性
        loss_pos = F.mse_loss(pred_quat_valid, gt_quat_valid, reduction='none').sum(dim=-1)
        loss_neg = F.mse_loss(pred_quat_valid, -gt_quat_valid, reduction='none').sum(dim=-1)
        loss = torch.min(loss_pos, loss_neg).mean()
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    return loss

def camera_fov_loss(batch):
    """相机视场角损失 (7-9参数)"""
    if 'pred_camera_params' not in batch or 'camera' not in batch:
        device = next(iter(batch.values())).device
        return torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    pred_camera = batch['pred_camera_params']
    gt_camera = batch['camera']
    has_camera = batch.get('has_camera', torch.ones(gt_camera.shape[0]))
    device = pred_camera.device
    
    # 只使用视场角部分
    pred_fov = pred_camera[..., 7:9]
    gt_fov = gt_camera[..., 7:9]
    
    # 处理有效数据
    if gt_fov.dim() == 3:
        if has_camera.dim() == 1:
            has_camera = has_camera.unsqueeze(1).expand(-1, gt_fov.shape[1])
        valid_mask = has_camera == 1
        pred_fov_valid = pred_fov[valid_mask]
        gt_fov_valid = gt_fov[valid_mask]
    else:
        valid_mask = has_camera == 1
        pred_fov_valid = pred_fov[valid_mask]
        gt_fov_valid = gt_fov[valid_mask]
    
    if len(pred_fov_valid) > 0:
        loss = F.mse_loss(pred_fov_valid, gt_fov_valid)
    else:
        loss = torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    return loss

def camera_depth_loss(batch):
    """
    VGGT相机深度约束损失 - 强制预测正深度值
    针对VGGT相机参数格式: [translation(3) + quaternion(4) + fov(2)]
    约束translation的Z分量（深度）为正值
    """
    if 'pred_camera_params' not in batch:
        device = next(iter(batch.values())).device
        return torch.FloatTensor(1).fill_(0.).mean().to(device)
    
    pred_camera = batch['pred_camera_params']  # [B, T, 9] 或 [B, 9]
    
    # 提取Z轴平移分量（深度）
    if pred_camera.dim() == 3:  # [B, T, 9]
        depth = pred_camera[:, :, 2]  # Z轴平移
        depth = depth.view(-1)  # 展平为 [B*T]
    else:  # [B, 9]
        depth = pred_camera[:, 2]  # Z轴平移
    
    # 使用与原始cam_depth_loss相同的公式
    # 强制深度为正值
    loss = ((torch.exp(-depth * 10)) ** 2).mean()
    
    return loss.clamp(min=None, max=10.0)

collection = {'KPT2D': keypoint_loss, 'KPT3D': keypoint_3d_loss, 'SMPL':  smpl_losses,
              # 'CAM_S': cam_depth_loss,     # 移除原有的深度损失（被VGGT替换）
              'V3D': vertice_loss, 'ACCEL': acceleration_loss,
              'SMPL_PLUS': smpl_losses_plus,
              # 完整的VGGT相机损失系统（替换所有原有相机损失）
              'CAMERA_LOSS': camera_loss,
              'CAMERA_TRANS': camera_translation_loss,
              'CAMERA_ROT': camera_rotation_loss,
              'CAMERA_FOV': camera_fov_loss,
              'CAMERA_DEPTH': camera_depth_loss} 


def compile_criterion(cfg):
    MixLoss = BaseLoss()
    for t, w in cfg.LOSS.items():
        MixLoss.weights[t] = w
        MixLoss.functions[t] = collection[t]

    return MixLoss


class BaseLoss(torch.nn.Module):
    def __init__(self,):
        super(BaseLoss, self).__init__()
        self.weights = {}
        self.functions = {}

    def forward(self, batch):
        losses = {}
        mixes_loss = 0
        for t, w in self.weights.items():
            loss = self.functions[t](batch)
            mixes_loss += w * loss
            losses[t] = loss.item()

        losses['mixed'] = mixes_loss.item()
        return mixes_loss, losses

    def report(self, ):
        return


class KptsMSELoss(torch.nn.Module):
    def __init__(self, use_vis=False):
        super(KptsMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.use_vis = use_vis

    def forward(self, output, target, vis):
        '''
        output: (BN, K, w, h)
        target: (BN, K, w, h)
        vis: (BN, K)
        '''
        batch_size = output.size(0)
        num_kpts = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_kpts, -1))
        heatmaps_gt = target.reshape((batch_size, num_kpts, -1))
        vis = vis.reshape((batch_size, num_kpts, 1))

        if self.use_vis:
            loss = self.criterion(
                heatmaps_pred.mul(vis),
                heatmaps_gt.mul(vis)
                )
        else:
            loss = self.criterion(heatmaps_pred, heatmaps_gt)

        return loss 


import numpy as np
import random
import torch
import torch.nn.functional as F

def augment_lidar(points):
    def random_rotation(points, angles_range=(-torch.pi, torch.pi)):
        """Randomly rotate the point cloud around the z-axis."""
        device = points.device
        batch_size = points.shape[0]
        theta = torch.rand(batch_size, device=device) * (angles_range[1] - angles_range[0]) + angles_range[0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrices = torch.stack([
            cos_theta, -sin_theta, torch.zeros_like(theta),
            sin_theta, cos_theta, torch.zeros_like(theta),
            torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)
        ], dim=1).reshape(-1, 3, 3).to(device)  # Ensure it's on the same device
        
        # Expand rotation matrices to match the shape of the points
        rotation_matrices = rotation_matrices.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, 3, 3)
        points_rotated = torch.einsum('bijmn,bijkm->bijkn', rotation_matrices, points[..., :3].unsqueeze(-2))
        points[..., :3] = points_rotated.squeeze(-2)
    
        return points
    def random_scaling(points, scale_low=0.8, scale_high=1.2):
        """Randomly scale the point cloud."""
        device = points.device
        scales = torch.rand(points.shape[0], 1, 1, 1, device=device) * (scale_high - scale_low) + scale_low
        points[:, :, :, :] *= scales
        return points

    def random_translation(points, translate_range=0.1):
        """Randomly translate the point cloud."""
        device = points.device
        translations = (torch.rand(points.shape[0], 1, 1, 3, device=device) * 2 - 1) * translate_range
        points[:, :, :, :3] += translations
        return points

    def random_jitter(points, sigma=0.01, clip=0.05):
        """Randomly jitter the point cloud."""
        jitter = torch.clamp(sigma * torch.randn_like(points[:, :, :, :]), -clip, clip)
        points[:, :, :, :] += jitter
        return points

    def random_dropout(points, max_dropout_ratio=0.3):
        """Randomly drop out points in the point cloud."""
        device = points.device
        batch_size, seq_len, num_points, _ = points.shape
        dropout_ratios = torch.rand(batch_size, device=device) * max_dropout_ratio
        for i in range(batch_size):
            dropout_mask = torch.rand(num_points, device=device) > dropout_ratios[i]
            points[i, :, dropout_mask, :] = points[i, :, dropout_mask, :]
            points[i, :, ~dropout_mask, :] = 0
        return points

    if torch.rand(1).item() < 0.6:
        points = random_rotation(points)
    if torch.rand(1).item() < 0.6:
        points = random_scaling(points)
    if torch.rand(1).item() < 0.6:
        points = random_translation(points)
    if torch.rand(1).item() < 0.6:
        points = random_jitter(points)
    if torch.rand(1).item() < 0.6:
        points = random_dropout(points)

    return points

def augment_mmwave(points):
    def random_rotation(points, angles_range=(-torch.pi, torch.pi)):
        """Randomly rotate the point cloud around the z-axis."""
        device = points.device
        batch_size = points.shape[0]
        theta = torch.rand(batch_size, device=device) * (angles_range[1] - angles_range[0]) + angles_range[0]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrices = torch.stack([
            cos_theta, -sin_theta, torch.zeros_like(theta),
            sin_theta, cos_theta, torch.zeros_like(theta),
            torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)
        ], dim=1).reshape(-1, 3, 3).to(device)  # Ensure it's on the same device
        
        # Expand rotation matrices to match the shape of the points
        rotation_matrices = rotation_matrices.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, 3, 3)
        points_rotated = torch.einsum('bijmn,bijkm->bijkn', rotation_matrices, points[..., :3].unsqueeze(-2))
        points[..., :3] = points_rotated.squeeze(-2)
    
        return points
    def random_scaling(points, scale_low=0.8, scale_high=1.2):
        """Randomly scale the point cloud."""
        device = points.device
        scales = torch.rand(points.shape[0], 1, 1, 1, device=device) * (scale_high - scale_low) + scale_low
        points[:, :, :, :] *= scales
        return points

    def random_translation(points, translate_range=0.1):
        """Randomly translate the point cloud."""
        device = points.device
        translations = (torch.rand(points.shape[0], 1, 1, 5, device=device) * 2 - 1) * translate_range
        points[:, :, :, :] += translations
        return points

    def random_jitter(points, sigma=0.01, clip=0.05):
        """Randomly jitter the point cloud."""
        jitter = torch.clamp(sigma * torch.randn_like(points[:, :, :, :]), -clip, clip)
        points[:, :, :, :] += jitter
        return points

    def random_dropout(points, max_dropout_ratio=0.3):
        """Randomly drop out points in the point cloud."""
        device = points.device
        batch_size, seq_len, num_points, _ = points.shape
        dropout_ratios = torch.rand(batch_size, device=device) * max_dropout_ratio
        for i in range(batch_size):
            dropout_mask = torch.rand(num_points, device=device) > dropout_ratios[i]
            points[i, :, dropout_mask, :] = points[i, :, dropout_mask, :]
            points[i, :, ~dropout_mask, :] = 0
        return points

    if torch.rand(1).item() < 0.6:
        points = random_rotation(points)
    if torch.rand(1).item() < 0.6:
        points = random_scaling(points)
    if torch.rand(1).item() < 0.6:
        points = random_translation(points)
    if torch.rand(1).item() < 0.6:
        points = random_jitter(points)
    if torch.rand(1).item() < 0.6:
        points = random_dropout(points)

    return points


def augment_rgb(data):
    M, N = random.choice([[1, 0], [2, 1], [3, 0], [3, 1]])
    batch_size, timesteps, num_points, coords = data.shape
    orig_steps = torch.arange(timesteps, device=data.device).float()
    interp_steps = torch.arange(0, orig_steps[-1].item() + 0.001, 1 / (M + 1), device=data.device).float()

    # Reshape data to (batch_size, num_points*coords, timesteps) for interpolation
    data_reshaped = data.permute(0, 2, 3, 1).reshape(batch_size, num_points * coords, timesteps)

    # Perform interpolation
    interp_data = F.interpolate(data_reshaped, size=len(interp_steps), mode='linear', align_corners=False)

    # Reshape back to original format (batch_size, new_timesteps, num_points, coords)
    interp_data = interp_data.reshape(batch_size, num_points, coords, len(interp_steps)).permute(0, 3, 1, 2)

    length_inserted = interp_data.shape[1]
    start = random.randint(0, length_inserted - timesteps * (N + 1))
    index_selected = torch.arange(start, start + timesteps * (N + 1), N + 1, device=data.device)
    
    out = interp_data[:, index_selected, :, :]
    
    return out
    

def augment_wifi_csi(data):
    M, N = random.choice([[1, 0], [2, 1], [3, 0], [3, 1]])
    batch_size, timesteps, num_points, coords = data.shape
    orig_steps = torch.arange(timesteps, device=data.device).float()
    interp_steps = torch.arange(0, orig_steps[-1].item() + 0.001, 1 / (M + 1), device=data.device).float()

    # Reshape data to (batch_size, num_points*coords, timesteps) for interpolation
    data_reshaped = data.permute(0, 2, 3, 1).reshape(batch_size, num_points * coords, timesteps)

    # Perform interpolation
    interp_data = F.interpolate(data_reshaped, size=len(interp_steps), mode='linear', align_corners=False)

    # Reshape back to original format (batch_size, new_timesteps, num_points, coords)
    interp_data = interp_data.reshape(batch_size, num_points, coords, len(interp_steps)).permute(0, 3, 1, 2)

    length_inserted = interp_data.shape[1]
    start = random.randint(0, length_inserted - timesteps * (N + 1))
    index_selected = torch.arange(start, start + timesteps * (N + 1), N + 1, device=data.device)
    
    out = interp_data[:, index_selected, :, :]
    
    return out

def augment_data(data, mod):
    if mod == 'mmwave':
        return augment_mmwave(data)
    elif mod == 'lidar':
        return augment_lidar(data)
    elif mod == 'rgb' or mod == 'infra1' or mod == 'infra2':
        return augment_rgb(data)
    elif mod == 'wifi-csi':
        return augment_wifi_csi(data)
    else:
        raise ValueError(f"Unknown modality {mod}")
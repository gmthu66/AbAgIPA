import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def sample_and_group(num_points, radius, nsample, xyz, points):
    B, _, N = xyz.size()

    # Sample points
    fps_idx = torch.randperm(N)[:num_points]
    new_xyz = index_points(xyz, fps_idx)

    # Grouping
    idx, grouped_xyz, grouped_points = query_ball_point(radius, nsample, xyz, new_xyz, points)

    return grouped_xyz, grouped_points


def index_points(points, idx):
    return points[:, :, idx]


def query_ball_point(radius, nsample, xyz1, xyz2, points=None):
    B, _, N1 = xyz1.size()
    _, _, N2 = xyz2.size()

    xyz1 = xyz1.view(B, 3, 1, N1)
    xyz2 = xyz2.view(B, 1, 3, N2)

    dist = torch.norm(xyz1 - xyz2, 2, dim=2)
    idx = dist < radius

    grouped_xyz = index_points(xyz2, idx)
    grouped_points = None
    if points is not None:
        grouped_points = index_points(points, idx)

    idx = idx.long().sum(dim=2, keepdim=True)

    return idx, grouped_xyz, grouped_points


class SetAbstraction(nn.Module):
    def __init__(self, in_channels, out_channels, num_points, radius, nsample):
        super(SetAbstraction, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv1d(out_channels, out_channels, 1)
        self.num_points = num_points
        self.radius = radius
        self.nsample = nsample

    def forward(self, xyz, points):
        B, _, N = xyz.size()
        _, C, _ = points.size()

        # Sample and group
        new_xyz = sample_and_group(self.num_points, self.radius, self.nsample, xyz, points)

        # Forward through MLP
        new_points = F.relu(self.conv1(new_xyz))
        new_points = F.relu(self.conv2(new_points))
        new_points = F.relu(self.conv3(new_points))
        new_points = F.max_pool1d(new_points, new_points.size(-1)).squeeze(-1)

        return new_xyz, new_points


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeaturePropagation, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: high-level sampled point set
        xyz2: low-level sampled point set
        points1: high-level feature set
        points2: low-level feature set
        """
        B, _, N1 = xyz1.size()
        _, _, N2 = xyz2.size()

        # Interpolate features
        dist, idx = query_ball_point(0.2, 32, xyz2, xyz1)
        grouped_points = index_points(points2, idx)
        grouped_points = F.relu(self.conv1(grouped_points))

        # Feature concatenation
        new_points = torch.cat([points1, grouped_points], dim=1)
        new_points = F.relu(self.conv2(new_points))

        return new_points


class PointNetPPDownsample(nn.Module):
    def __init__(self):
        super(PointNetPPDownsample, self).__init__()
        self.sa1 = SetAbstraction(3, 64, 1024, radius=4, nsample=32)
        self.sa2 = SetAbstraction(64, 128, 128, radius=8, nsample=32)
        self.fp1 = FeaturePropagation(128 + 64, 64)
        self.fp2 = FeaturePropagation(64 + 3, 64)

    def forward(self, xyz, points=None):
        # points = xyz  # assuming xyz is the input point cloud
        points = deepcopy(xyz) if points is None else points
        xyz1, points1 = self.sa1(xyz, points)  # points是每个点的特征信息， 是点的属性(颜色, 氨基酸类型等等)
        xyz2, points2 = self.sa2(xyz1, points1)
        points1_upsampled = self.fp1(xyz2, xyz1, points2, points1)
        new_points = self.fp2(xyz, xyz1, points1_upsampled, points)

        return new_points


class GlobalFeatureExtraction(nn.Module):
    def __init__(self):
        super(GlobalFeatureExtraction, self).__init__()
        self.downsample_module = PointNetPPDownsample()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, xyz):
        global_features = self.downsample_module(xyz)
        global_features = F.relu(self.fc1(global_features.squeeze(2)))
        global_features = F.relu(self.fc2(global_features))
        return global_features


if __name__ == "__main__":
    # Example usage
    global_feature_extractor = GlobalFeatureExtraction()
    # Assuming input point cloud has shape [batch_size, 3, num_points]
    xyz = torch.randn(16, 3, 1024)
    # Forward pass
    global_features = global_feature_extractor(xyz)

    print("Global features size:", global_features.size())

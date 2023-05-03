import open3d as o3d
import numpy as np
import os
import sys
import time

sys.path.append('./lidar_detector')
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

# Set parameters
VOXEL_GRID_SIZE = 0.05
ROI_MIN_POINT = np.array([-10, -10, -2])
ROI_MAX_POINT = np.array([10, 10, 2])
GROUND_THRESH = 0.1
CLUSTER_THRESH = 0.5
CLUSTER_MIN_SIZE = 30
CLUSTER_MAX_SIZE = 1000

# Define 3D detector class
class Detector3D:
    def __init__(self):
        self.results = []
        self.folder_path = 'lidar_detector/PCD'
        self.vis = o3d.visualization.Visualizer()

    
    def downsample(self, pcd, voxel_size):
        downpcd = pcd.voxel_down_sample(voxel_size)
        return downpcd
    
    def roi_filter(self, pcd, roi_min, roi_max):
        roi_filter = pcd.crop(
            o3d.geometry.AxisAlignedBoundingBox(roi_min, roi_max))
        return roi_filter
    
    def ground_segmentation(self, pcd, threshold):
        plane_model, inliers = pcd.segment_plane(threshold, 3, 1000)
        [a, b, c, d] = plane_model
        if d < 0:
            a, b, c, d = -a, -b, -c, -d
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud
    
    def cluster(self, pcd, cluster_threshold, min_size, max_size):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(pcd.cluster_dbscan(eps=cluster_threshold, min_points=min_size, print_progress=False))
        max_label = labels.max()
        clusters = []
        for i in range(max_label + 1):
            cluster = pcd.select_by_index(np.where(labels == i)[0])
            if len(cluster.points) > max_size:
                continue
            clusters.append(cluster)
        return clusters
    
    # Process the original point cloud
    def process_cloud(self, pcd):
        downsampled_pcd = self.downsample(pcd, VOXEL_GRID_SIZE)
        roi_pcd = self.roi_filter(downsampled_pcd, ROI_MIN_POINT, ROI_MAX_POINT)
        inlier_pcd, outlier_pcd = self.ground_segmentation(roi_pcd, GROUND_THRESH)
        clusters = self.cluster(outlier_pcd, CLUSTER_THRESH, CLUSTER_MIN_SIZE, CLUSTER_MAX_SIZE)

        return clusters

    def create_bounding_box(self, clusters):
        for cluster in clusters:
            # 获得聚类的点云
            points = np.asarray(cluster.points)
            # 计算点云的中心点
            center = points.mean(axis=0)
            # 计算点云的协方差矩阵
            cov = np.cov(points.T)
            # 使用特征值分解求解长方体的3个半轴的长度和方向
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            # 对特征值进行排序，从大到小排序
            sorted_idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[sorted_idx]
            eigenvectors = eigenvectors[:, sorted_idx]
            # 长方体的3个半轴的长度
            length = np.sqrt(eigenvalues) * 2
            # 长方体的3个半轴的方向
            direction = eigenvectors
            # 计算长方体的8个顶点坐标
            corner_points = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        x = center[0] + length[0]/2 * (-1)**i
                        y = center[1] + length[1]/2 * (-1)**j
                        z = center[2] + length[2]/2 * (-1)**k
                        point = np.array([x, y, z])
                        corner_points.append(point)
            # 输出长方体的8个顶点坐标
            # print("Bounding box corner points:")
            # for i, point in enumerate(corner_points):
                # print(f"Corner {i+1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
            # print("")

            self.results.append(corner_points)

    def run(self):
        # Sort them by filename
        dir_path = self.folder_path
        pcd_files = sorted([filename for filename in os.listdir(dir_path) if filename.endswith('.pcd')])

        # Read files in a loop
        for filename in pcd_files:
            self.vis.create_window()
            self.vis.close()
            file_path = os.path.join(self.folder_path, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            clusters = self.process_cloud(pcd)
            for cluster in clusters:
                self.create_bounding_box([cluster])
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
            for i in range(len(clusters)):
                cluster = clusters[i]
                color = colors[i % len(colors)]
                cluster.paint_uniform_color(color)
                self.vis.add_geometry(cluster)
            view_control = self.vis.get_view_control()
            view_control.set_front([0.0, 1.0, 0.0])
            view_control.set_up([0.0, 0.0, 1.0]) 
            self.vis.run()
            self.vis.clear_geometries()

import open3d as o3d
import numpy as np
import os

# 设置参数
VOXEL_GRID_SIZE = 0.05
ROI_MIN_POINT = np.array([-10, -10, -2])
ROI_MAX_POINT = np.array([10, 10, 2])
GROUND_THRESH = 0.2
CLUSTER_THRESH = 0.5
CLUSTER_MIN_SIZE = 10
CLUSTER_MAX_SIZE = 10000

# 定义障碍物检测类
class ObstacleDetector:
    def __init__(self):
        self.labels = []
    
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
        print(f"Model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        inlier_cloud = pcd.select_by_index(inliers)
        outlier_cloud = pcd.select_by_index(inliers, invert=True)
        return inlier_cloud, outlier_cloud
    
    def cluster(self, pcd, cluster_threshold, min_size, max_size):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcd.cluster_dbscan(eps=cluster_threshold, min_points=min_size, print_progress=False))
        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        clusters = []
        for i in range(max_label + 1):
            cluster = pcd.select_by_index(np.where(labels == i)[0])
            if len(cluster.points) > max_size:
                continue
            clusters.append(cluster)
        return clusters
    
    # 对原始点云进行处理
    def process_cloud(self, pcd):
        downsampled_pcd = self.downsample(pcd, VOXEL_GRID_SIZE)
        roi_pcd = self.roi_filter(downsampled_pcd, ROI_MIN_POINT, ROI_MAX_POINT)
        inlier_pcd, outlier_pcd = self.ground_segmentation(roi_pcd, GROUND_THRESH)
        clusters = self.cluster(outlier_pcd, CLUSTER_THRESH, CLUSTER_MIN_SIZE, CLUSTER_MAX_SIZE)

        return clusters

# 遍历文件夹中的所有pc
if __name__ == '__main__':
    detector = ObstacleDetector()
    folder_path = './PCD'
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcd"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file {file_path}")
            pcd = o3d.io.read_point_cloud(file_path)
            clusters = detector.process_cloud(pcd)
            # 可视化
            colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]]
            for i in range(len(clusters)):
                cluster = clusters[i]
                color = colors[i % len(colors)]
                cluster.paint_uniform_color(color)
            o3d.visualization.draw_geometries(clusters)

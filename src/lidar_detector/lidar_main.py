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

    def create_bounding_box(self, clusters):
        for cluster in clusters:
            # 获得聚类的点云
            points = np.asarray(cluster.points)
            # 获取外接长方体的八个顶点坐标
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(cluster.points)
            corners = bbox.get_box_points()
            # 输出长方体的8个顶点坐标
            print("Bounding box corner points:")
            for i, point in enumerate(corners):
                print(f"Corner {i+1}: ({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
            print("")
            

# 遍历文件夹中的所有pcd
if __name__ == '__main__':
    detector = ObstacleDetector()
    folder_path = './PCD'
    
    vis = o3d.visualization.Visualizer()
    #创建播放窗口
    vis.create_window()
    pointcloud = o3d.geometry.PointCloud()
    to_reset = True
    vis.add_geometry(pointcloud)
    for filename in os.listdir(folder_path):
        if filename.endswith(".pcd"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file {file_path}")
            pcd = o3d.io.read_point_cloud(file_path)
            # 输出八个顶点
            clusters = detector.process_cloud(pcd)
            for cluster in clusters:
                detector.create_bounding_box([cluster])  # 调用 create_bounding_box()
            
            # 可视化
            pcd = np.asarray(pcd.points).reshape((-1, 3))
            pointcloud.points = o3d.utility.Vector3dVector(pcd)  # 如果使用numpy数组可省略上两行
            vis.update_geometry()
            if to_reset:
                vis.reset_view_point(True)
                to_reset = False
            vis.poll_events()
            vis.update_renderer()


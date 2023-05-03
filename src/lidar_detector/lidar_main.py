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
            # Get the coordinates of the eight vertices
            bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(cluster.points)
            corner_points = np.asarray(bbox.get_box_points())

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

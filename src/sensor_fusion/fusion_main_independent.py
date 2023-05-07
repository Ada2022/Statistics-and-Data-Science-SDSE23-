import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class Matcher:
    def __init__(self, camera_results, lidar_results, lidar_to_camera_transform, camera_intrinsic_parameters):
        # Initialize
        self.camera_results = camera_results
        self.lidar_results = lidar_results
        self.lidar_to_camera_transform = lidar_to_camera_transform
        self.camera_intrinsic_parameters = camera_intrinsic_parameters

    def transform_vertices_to_camera_coordinates(self, lidar_box):
        # Transform lidar bounding box vertices to camera coordinate system
        camera_vertices = []
        for vertex in lidar_box:
            coordinate_transform = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
            vertex = coordinate_transform @ vertex
            transformed_vertex = self.lidar_to_camera_transform @ np.append(vertex, 1).reshape(4, 1)
            transformed_vertex = transformed_vertex[:3] / transformed_vertex[3]
            camera_vertices.append(transformed_vertex)
        return np.array(camera_vertices)

    def project_vertices_to_pixel_coordinates(self, camera_vertices):
        # Project camera bounding box vertices to pixel coordinates
        pixel_vertices = []
        for vertex in camera_vertices:
            pixel_vertex = self.camera_intrinsic_parameters @ np.append(vertex, 1).reshape(4, 1)
            pixel_vertex = pixel_vertex[:2] / pixel_vertex[2]
            pixel_vertices.append(pixel_vertex)
        return np.array(pixel_vertices)

    def compute_iou(self, bbox1, bbox2):
        # Compute IoU between two bounding boxes
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x1 >= x2 or y1 >= y2:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        iou = intersection / (area1 + area2 - intersection)
        return iou

    def run(self):
        # Match camera and lidar detection results
        cost_matrix = np.zeros((len(self.camera_results), len(self.lidar_results)))
        candidates = []
        for i, camera_result in enumerate(self.camera_results):
            for j, lidar_result in enumerate(self.lidar_results):
                
                # Transform lidar bounding box to camera coordinates
                lidar_vertices = self.transform_vertices_to_camera_coordinates(lidar_result)
                pixel_vertices = self.project_vertices_to_pixel_coordinates(lidar_vertices)
                pixel_vertices = np.squeeze(pixel_vertices).astype(np.float32)
                x, y, w, h = cv2.boundingRect(pixel_vertices)
                bbox_lidar = np.array([x, y, x + w, y + h])
                candidates.append(bbox_lidar)

                # Compute IoU between camera and lidar bounding boxes
                bbox_camera = camera_result
                iou = self.compute_iou(bbox_camera, bbox_lidar)
                cost_matrix[i, j] = 1 - iou

        # Perform linear assignment using Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        fused_results = []
        fused_results = candidates[col_ind[0]]

        return fused_results
    
if __name__ == '__main__':
    lidar_results_transform = np.array([[ 9.99635086e-01,  2.69526266e-02, -1.80319433e-03, -6.48575600e-02],
                                        [-2.70109374e-02,  9.98133934e-01, -5.47636676e-02, -5.68320100e-02],
                                        [ 3.23804765e-04,  5.47923895e-02,  9.98497716e-01, -1.04481150e-01],
                                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

    camera_intrinsic_parameters = np.array([[ 611.524169921875,  0.00000000e+00, 321.3505554199219, 0.00000000e+00],
                                    [0.00000000e+00,  609.79541015625, 249.3844451904297, 0.00000000e+00],
                                    [0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.00000000e+00]])

    camera_demo_result = [np.array([481.87503, 146.162  , 534.0785 , 329.9514 ])]

    lidar_demo_result = [np.array([[ 0.06879278, -6.1072984 , -0.65451008],
    [ 0.36503592, -6.1072984 , -0.65451008],
    [ 0.06879278, -5.8985167 , -0.65451008],
    [ 0.06879278, -6.1072984 ,  0.66487181],
    [ 0.36503592, -5.8985167 ,  0.66487181],
    [ 0.06879277, -5.8985167 ,  0.66487181],
    [ 0.36503592, -6.1072984 ,  0.66487181],
    [ 0.36503592, -5.8985167 , -0.65451008]]), np.array([[-2.80742945, -6.8049841 , -0.63871539],
    [-2.0980308 , -6.8049841 , -0.63871539],
    [-2.80742945, -6.2983217 , -0.63871539],
    [-2.80742945, -6.8049841 ,  1.1471205 ],
    [-2.0980308 , -6.2983217 ,  1.1471205 ],
    [-2.80742945, -6.2983217 ,  1.1471205 ],
    [-2.0980308 , -6.8049841 ,  1.1471205 ],
    [-2.0980308 , -6.2983217 , -0.63871539]]), np.array([[-8.1947193 , -9.99965   , -0.55875289],
    [-5.8383069 , -9.99965   , -0.55875289],
    [-8.1947193 , -9.2074757 , -0.55875289],
    [-8.1947193 , -9.99965   ,  1.8592333 ],
    [-5.8383069 , -9.2074757 ,  1.8592333 ],
    [-8.1947193 , -9.2074757 ,  1.8592333 ],
    [-5.8383069 , -9.99965   ,  1.8592333 ],
    [-5.8383069 , -9.2074757 , -0.55875289]])]

    # sensor fusion 
    matcher = Matcher(camera_demo_result, lidar_demo_result, lidar_results_transform, camera_intrinsic_parameters)
    fused_results = matcher.run()

    color = (0, 0, 255) 
    thickness = 3 
    demo = cv2.imread("C:/Users/yuqing/Desktop/test.png")
    cv2.rectangle(demo, (fused_results[0], fused_results[1]), (fused_results[2], fused_results[3]), color, thickness)
    cv2.putText(demo, "lidar result", (int(fused_results[0] + 2), int(fused_results[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Result", demo)
    cv2.waitKey(0)

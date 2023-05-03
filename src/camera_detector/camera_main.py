import argparse
import os
import cv2
import time
import numpy as np
import torch
import sys
sys.path.append('./camera_detector')

from camera_detector.config.yolo_config import yolo_config
from camera_detector.data.coco import coco_class_labels, coco_class_index
from camera_detector.data.transforms import ValTransforms
from camera_detector.models.yolo import build_model


class Detector2D:
    def __init__(self) -> None:
        self.person_bboxes = []

    def parse_args(self):
        parser = argparse.ArgumentParser(description='YOLO Demo Detection')

        # basic
        parser.add_argument('--mode', default='video',
                            type=str, help='Use the data from image, video or camera')
        parser.add_argument('--cuda', action='store_true', default=False,
                            help='Use cuda')
        parser.add_argument('--path_to_img', default='data/demo/images/',
                            type=str, help='The path to image files')
        parser.add_argument('--path_to_vid', default='camera_detector/data/demo/videos/output.avi',
                            type=str, help='The path to video files')
        parser.add_argument('--path_to_save', default='camera_detector/det_results/images/',
                            type=str, help='The path to save the detection results')
        parser.add_argument('--path_to_saveVid', default='data/video/result.avi',
                            type=str, help='The path to save the detection results video')
        parser.add_argument('-vs', '--visual_threshold', default=0.4,
                            type=float, help='visual threshold')

        # model
        parser.add_argument('-m', '--model', default='yolo_nano',
                            help='yolov1, yolov2, yolov3, yolov3_spp, yolov3_de, '
                                'yolov4, yolo_tiny, yolo_nano')
        parser.add_argument('--num_queries', type=int, default=4, 
                            help='number of queris of YOLOQ')
        parser.add_argument('--weight', default='camera_detector/weights/yolo_nano_22.4_40.7.pth',
                            type=str, help='Trained state_dict file path to open')
        parser.add_argument('-size', '--img_size', default=640, type=int,
                            help='img_size')
        parser.add_argument('--conf_thresh', default=0.1, type=float,
                            help='NMS threshold')
        parser.add_argument('--nms_thresh', default=0.45, type=float,
                            help='NMS threshold')
        parser.add_argument('--center_sample', action='store_true', default=False,
                            help='center sample trick.')
        
        return parser.parse_args()
                        

    def plot_bbox_labels(self, img, bbox, label, cls_color, test_scale=0.4):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
        # plot bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        return img


    def visualize(self, img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
        ts = 0.4
        for i, bbox in enumerate(bboxes):
            if scores[i] > vis_thresh and coco_class_index[int(cls_inds[i])] == 1:
                cls_color = class_colors[int(cls_inds[i])]
                cls_id = coco_class_index[int(cls_inds[i])]
                mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
                img = self.plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

        return img


    def detect(self, net, 
            device, 
            transform, 
            vis_thresh, 
            mode='image', 
            path_to_img=None, 
            path_to_vid=None, 
            path_to_save=None):
        # class color
        class_colors = [(np.random.randint(255),
                        np.random.randint(255),
                        np.random.randint(255)) for _ in range(80)]
        save_path = os.path.join(path_to_save, mode)
        os.makedirs(save_path, exist_ok=True)

        if mode == 'video':
            video = cv2.VideoCapture(path_to_vid)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            save_size = (640, 480)
            save_path = os.path.join(save_path, 'det.avi')
            fps = 15.0
            out = cv2.VideoWriter(save_path, fourcc, fps, save_size)

            while(True):
                ret, frame = video.read()

                if ret:
                    # ------------------------- Detection ---------------------------
                    self.img  = frame
                    img_h, img_w = frame.shape[:2]
                    size = np.array([[img_w, img_h, img_w, img_h]])
                    # prepare
                    x, _, _, scale, offset = transform(frame)
                    x = x.unsqueeze(0).to(device)
                    # inference
                    t0 = time.time()
                    bboxes, scores, cls_inds = net(x)
                    
                    t1 = time.time()
                    # print("detection time used ", t1-t0, "s")

                    # rescale
                    bboxes -= offset
                    bboxes /= scale
                    bboxes *= size

                    self.person_bboxes = []
                    for i, bbox in enumerate(bboxes):
                            if coco_class_index[int(cls_inds[i])] == 1:
                                self.person_bboxes.append(bbox)

                    # print('person bboxes are:', self.person_bboxes)
                    
                    frame_processed = self.visualize(img=frame, 
                                                bboxes=bboxes,
                                                scores=scores, 
                                                cls_inds=cls_inds,
                                                class_colors=class_colors,
                                                vis_thresh=vis_thresh)

                    frame_processed_resize = cv2.resize(frame_processed, save_size)
                    out.write(frame_processed_resize)
                    cv2.imshow('detection', frame_processed)
                    cv2.waitKey(1)
                else:
                    break
            video.release()
            out.release()
            cv2.destroyAllWindows()
    

    def run(self):
        args = self.parse_args()

        # use cuda
        if args.cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # YOLO Config
        cfg = yolo_config[args.model]
        # build model
        model = build_model(args=args, 
                            cfg=cfg, 
                            device=device, 
                            num_classes=80, 
                            trainable=False)

        # load weight
        model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
        model = model.to(device).eval()
        print('Finished loading model!')

        # run
        self.detect(net=model, 
                device=device,
                transform=ValTransforms(args.img_size),
                mode=args.mode,
                path_to_img=args.path_to_img,
                path_to_vid=args.path_to_vid,
                path_to_save=args.path_to_save,
                vis_thresh=args.visual_threshold)
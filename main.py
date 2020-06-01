import os
import cv2
import time
import torch
import random
import argparse
import mot.utils
import mot.encode
import mot.metric
import mot.predict
import numpy as np
import torchvision
import mot.associate
from mot.detection import Centernet_tensorrt
import tensorrt as trt
from mot.tracker import Tracker
from action import TSM

class IoUTracker(Tracker):
    def __init__(self, detector, sigma_conf=0.4):
        metric = mot.metric.IoUMetric(use_prediction=True)
        encoder = mot.encode.ImagePatchEncoder(resize_to=(32, 32))
        matcher = mot.associate.HungarianMatcher(metric, sigma=0.1)
        predictor = mot.predict.KalmanPredictor()
        # predictor=None
        super().__init__(detector, [encoder], matcher, predictor)
        self.sigma_conf = sigma_conf






# def ramdom_sample(images, num_segments):
#     total_images = len(images)
#     image_inds = []
#     segment_length = int(total_images / num_segments)
#     for i in range(num_segments):
#         image_inds.append(random.randint(segment_length * i, segment_length * i + segment_length - 1))
#     input_mean = [0.485, 0.456, 0.406]
#     input_std = [0.229, 0.224, 0.225]
#     transform = torchvision.transforms.Compose([
#         ToTorchFormatTensor(),
#         GroupNormalize(input_mean, input_std)])
#     # for ind in image_inds:
#     #     cv2.imshow('templates',images[ind])
#     #     cv2.waitKey(0)
#     images_list=[transform(images[ind]) for ind in image_inds]
#     image_tensor=torch.cat(images_list,0)
#     return image_tensor


def get_video_writer(save_video_path, width, height):
    if save_video_path != '':
        return cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 14, (int(width), int(height)))
    else:
        class MuteVideoWriter():
            def write(self, *args, **kwargs):
                pass

            def release(self):
                pass

        return MuteVideoWriter()



def track_and_recognize(tracker, recognizer, args):
    capture = cv2.VideoCapture(args.video_path)
    # video_writer = get_video_writer(args.save_video, capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                                    # capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=0
    while True:
        start_time=time.time()
        ret, frame = capture.read()
        # if count<900:
        #     count+=1
        #     continue
        if not ret:
            break
        frame = frame[80:, :640]
        tracker.tick(frame)
        frame = mot.utils.visualize_snapshot(frame, tracker)

        # Perform action recognition each second
        recognizer(tracker.tracklets_active)
        for tracklet in tracker.tracklets_active:
            if tracklet.action[-1] == 0:
                box = tracklet.last_detection.box
                frame = cv2.putText(frame, 'walking', (int(box[0] + 4), int(box[1]) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
            elif tracklet.action[-1] == 1:
                box = tracklet.last_detection.box
                frame = cv2.putText(frame, 'standing', (int(box[0] + 4), int(box[1]) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
            end_time=time.time()
            print(end_time-start_time)
            cv2.imshow('Demo', frame)
            # video_writer.write(frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

    # video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dw', '--detector_model_path', required=True, help='root path of object detector models')
    parser.add_argument( '--action_path', required=True, help='root path of object detector models')

    parser.add_argument('-rw', '--recognizer_checkpoint', required=False,
                        help='checkpoint file of TSN action recognizer')
    parser.add_argument('-i', '--video_path', default='', required=False,
                        help='Path to the test video file or directory of test images. Leave it blank to use webcam.')
    parser.add_argument('-o', '--save_video', default='', required=False,
                        help='Path to the output video file. Leave it blank to disable.')
    parser.add_argument('-s', '--save_result', default='', required=False,
                        help='Path to the output track result file. Leave it blank to disable.')
    parser.add_argument('--num_segments', default=4, help='set segments num for action part')
    parser.add_argument('--vis_thres', default=0.3, type=float,help='threshold of detection')
    parser.add_argument('--max_per_image', default=20, help='max objects in per image')

    args = parser.parse_args()

    detector = Centernet_tensorrt(args)
    tracker = IoUTracker(detector, sigma_conf=0.3)
    recognizer = TSM(args)
    track_and_recognize(tracker,recognizer, args)

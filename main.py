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
from action import MobileNetV2,GroupNormalize,ToTorchFormatTensor
from mot.tracker import Tracker


class IoUTracker(Tracker):
    def __init__(self, detector, sigma_conf=0.4):
        metric = mot.metric.IoUMetric(use_prediction=True)
        encoder = mot.encode.ImagePatchEncoder(resize_to=(32, 32))
        matcher = mot.associate.HungarianMatcher(metric, sigma=0.1)
        predictor = mot.predict.KalmanPredictor()
        # predictor=None
        super().__init__(detector, [encoder], matcher, predictor)
        self.sigma_conf = sigma_conf




class TSM:
    def __init__(self, checkpoint,num_segments):
        self.model = MobileNetV2()
        self.num_segments=num_segments
        if checkpoint is not None:
            sd = torch.load(checkpoint)
            sd = sd['state_dict']
            model_dict = self.model.state_dict()
            if isinstance(self.model, MobileNetV2):

                for k in list(sd.keys()):
                    if 'base_model' in k:
                        sd[k.replace('base_model.', '')] = sd.pop(k)
                for k in list(sd.keys()):
                    if 'module' in k:
                        sd[k.replace('module.', '')] = sd.pop(k)

                for k in list(sd.keys()):
                    if '.net' in k:
                        sd[k.replace('.net', '')] = sd.pop(k)
                for k in list(sd.keys()):
                    if 'new_fc' in k:
                        sd[k.replace('new_fc', 'classifier')] = sd.pop(k)
            model_dict.update(sd)
            self.model.load_state_dict(model_dict)

        self.model.eval()

    def __call__(self, images):
        # images = np.array(images)
        # images = images.transpose((0, 3, 1, 2))
        # images = np.expand_dims(images, 0)
        # images = images.astype(np.float32) - 128

        return self.model(images)


def ramdom_sample(images, num_segments):
    total_images = len(images)
    image_inds = []
    segment_length = int(total_images / num_segments)
    for i in range(num_segments):
        image_inds.append(random.randint(segment_length * i, segment_length * i + segment_length - 1))
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose([
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)])
    # for ind in image_inds:
    #     cv2.imshow('templates',images[ind])
    #     cv2.waitKey(0)
    images_list=[transform(images[ind]) for ind in image_inds]
    image_tensor=torch.cat(images_list,0)
    return image_tensor


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
        # for tracklet in tracker.tracklets_active:
        #     if tracklet.is_confirmed() and tracklet.is_detected() and len(tracklet.feature_history) >= args.num_segments:
        #         samples = ramdom_sample([feature[1]['patch'] for feature in tracklet.feature_history], args.num_segments)
        #         prediction = recognizer(samples)
        #         action = np.argmax(prediction.detach().cpu())
        #         if action == 0:
        #             box = tracklet.last_detection.box
        #             frame = cv2.putText(frame, 'walking', (int(box[0] + 4), int(box[1]) - 8),
        #                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
        #         elif action == 1:
        #             box = tracklet.last_detection.box
        #             frame = cv2.putText(frame, 'standing', (int(box[0] + 4), int(box[1]) - 8),
        #                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
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
    #
    #
    # recognizer = TSM(args.recognizer_checkpoint,args.num_segments)
    track_and_recognize(tracker,None, args)
